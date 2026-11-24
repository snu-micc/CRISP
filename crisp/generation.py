"""Build a new formation-energy catalog using the study request templates."""
import ast
from datetime import datetime
import json
from pathlib import Path
import re
import numpy as np
from .semantic import cluster_embeddings, top_keywords, parse_rules

ROOT = Path(__file__).resolve().parents[1]


def _write(folder, name, value):
    (folder / name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf8")


def _fill(value, fields):
    if isinstance(value, str):
        return re.sub(r"<<(\w+)>>", lambda m: fields[m[1]], value)
    if isinstance(value, dict):
        return {k: _fill(v,fields) for k,v in value.items()}
    if isinstance(value, list):
        return [_fill(v,fields) for v in value]
    return value


def _reply(client, request, fields=None):
    body = _fill(request["body"], {} if fields is None else fields)
    if request["endpoint"] == "responses":
        return client.responses.create(**body).output_text
    return client.chat.completions.create(**body).choices[0].message.content


def _numbered(text):
    rules = parse_rules(text)
    if len(rules) != 10 or not all(rules):
        raise ValueError("The generator did not return ten numbered rules.")
    return rules


def build_formation_catalog(client, folder, *, full=False, progress=print):
    """Make fresh proposals; do not read any of the distributed catalogs or model weights."""
    settings = json.loads((ROOT/"prompts"/"formation_energy.json").read_text())
    requests, cfg = settings["requests"], settings["profile"].copy()
    if not full:
        cfg.update(discovery_calls=3, clusters=3, pca_dimensions=10)
    dest = Path(folder) / ("formation_energy_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    dest.mkdir(parents=True,exist_ok=False)
    rows = []
    for i in range(cfg["discovery_calls"]):
        progress(f"Sampling rules: {i+1}/{cfg['discovery_calls']}")
        for text in _numbered(_reply(client,requests["discovery"])):
            rows.append(dict(rule_uid=f"rule_{len(rows):05d}",rule_text=text))
    _write(dest,"proposals.json", rows)
    keyed = {}
    for i in range(0,len(rows),cfg["keyword_batch_size"]):
        progress(f"Preparing rule texts: {i+1}–{min(i+cfg['keyword_batch_size'],len(rows))}/{len(rows)}")
        batch = rows[i:i+cfg["keyword_batch_size"]]
        reply = json.loads(_reply(client,requests["keywords"],{"rules_json":json.dumps(batch,ensure_ascii=False)}))
        items = reply["items"]
        if len(items)!=len(batch) or {r["rule_uid"] for r in items}!={r["rule_uid"] for r in batch}:
            raise ValueError("Returned keywords do not match the requested rules.")
        for r in items:
            keyed[r["rule_uid"]] = re.sub(r"\s+"," ",r["processed_rule"]).strip()
    texts = [keyed[r["rule_uid"]] for r in rows]
    if not all(texts): raise ValueError("Empty keyword text.")
    vec = []
    for i in range(0,len(texts),128):
        progress(f"Embedding rules: {i+1}–{min(i+128,len(texts))}/{len(texts)}")
        batch = texts[i:i+128]
        result = sorted(client.embeddings.create(model=cfg["embedding_model"],input=batch).data,key=lambda r:r.index)
        if [r.index for r in result] != list(range(len(batch))): raise ValueError("Missing embedding.")
        vec.extend(r.embedding for r in result)
    assigned = cluster_embeddings(np.array(vec,dtype=np.float32),cfg)
    group = {}
    for i,a in enumerate(assigned):
        group.setdefault(a["cluster_id"],[]).append(i)
    if len(group)!=cfg["clusters"]: raise ValueError("Fewer distinct clusters than requested.")
    keywords = {}
    for rid,ix in group.items():
        try: keywords[rid] = top_keywords(texts, np.array(ix))
        except ValueError:
            if full: raise
            keywords[rid] = ""
    map_text = "\n".join(f"{rid} | n={len(group[rid])} | {keywords[rid]}" for rid in sorted(group))
    _write(dest,"clusters.json", [dict(r, processed_rule=t, **a) for r,t,a in zip(rows,texts,assigned)])
    reps=[]
    for rid in sorted(group):
        progress(f"Consolidating {rid}")
        ix = sorted(group[rid],key=lambda i:assigned[i]["distance_to_centroid"])
        members=[dict(index=j,rule_uid=rows[i]["rule_uid"],processed_rule=texts[i],
                      original_rule=rows[i]["rule_text"]) for j,i in enumerate(ix,1)]
        fields=dict(cluster_id=rid,cluster_size=str(len(ix)),top_keywords=keywords[rid],
                    global_map=map_text,members_json=json.dumps(members,ensure_ascii=False))
        req=requests["curation"]
        if not full:
            req=json.loads(json.dumps(req).replace("25 clusters","3 clusters").replace("25 outputs","3 outputs")
                           .replace("25 CLUSTERS","3 CLUSTERS"))
        item=json.loads(_reply(client,req,fields))
        if item["cluster_id"]!=rid or not item["representative_rule"].strip(): raise ValueError("Invalid representative.")
        reps.append(item)
    _write(dest,"rules.json",reps)
    sources=dest/"functions"; sources.mkdir()
    for item in reps:
        rid=item["cluster_id"]; text=item["representative_rule"]
        stop={"the","and","with","from","that","this","into","for","its","are","one"}
        words=[w for w in re.findall(r"[a-zA-Z]+",text.lower()) if w not in stop][:5]
        name=f"rule_{rid[1:]}_"+("_".join(words) or f"descriptor_{rid[1:]}")
        progress(f"Writing {rid}")
        output=json.loads(_reply(client,requests["codegen"],
                                dict(cluster_id=rid,rule_text=text,function_name=name)))
        if output["cluster_id"]!=rid: raise ValueError("Mismatched code response.")
        code=output["function_code"].strip()
        # Preserve the draft for the researcher; never import or run generated source.
        (sources/(rid+".py")).write_text(code+"\n",encoding="utf8")
        tree=ast.parse(code)
        if sum(isinstance(n,ast.FunctionDef) and n.name==name for n in tree.body)!=1:
            raise ValueError(f"{rid} did not contain the requested function; the draft is retained.")
    return dest
