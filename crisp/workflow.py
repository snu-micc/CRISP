"""Dataset-blind OpenAI API workflow for compiling chemical rules into code."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

from .audit import audit_generated_code
from .prompts import code_prompt, get_task_prompts


NUMBERED_RULE = re.compile(
    r"(?:^|\n|,\s+)(\d{1,2})\s*[.:)]\s*(.+?)(?=(?:\n|,\s+)\d{1,2}\s*[.:)]\s*|\Z)",
    flags=re.DOTALL,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_numbered_rules(text: str, expected: int = 10) -> list[str]:
    """Parse the numbered list format used by the frozen discovery prompts."""
    parsed: dict[int, str] = {}
    for match in NUMBERED_RULE.finditer(text.strip()):
        index = int(match.group(1))
        rule = re.sub(r"\s+", " ", match.group(2)).strip(" -*\t\r\n")
        if 1 <= index <= expected and rule:
            parsed[index] = rule
    if len(parsed) == expected:
        return [parsed[index] for index in range(1, expected + 1)]

    fallback: dict[int, str] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\d{1,2})\s*[.:)]\s*(.+?)\s*$", line)
        if match:
            index = int(match.group(1))
            if 1 <= index <= expected:
                fallback[index] = re.sub(r"\s+", " ", match.group(2)).strip()
    return [fallback[index] for index in range(1, expected + 1) if index in fallback]


def strip_code_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:python)?\s*", "", value, count=1, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value, count=1)
    return value.strip()


def slugify(text: str, fallback: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:6]
    value = "_".join(words).strip("_")
    return value[:56] or fallback


def _response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    return json.loads(response.json())


def _request_text(
    client: OpenAI,
    *,
    model: str,
    user: str,
    instructions: str | None,
    temperature: float | None,
    max_output_tokens: int,
    attempts: int = 4,
) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": user,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if temperature is not None:
        kwargs["temperature"] = temperature

    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.responses.create(**kwargs)
            text = (response.output_text or "").strip()
            if not text:
                raise RuntimeError("API response contained no output text")
            return text, _response_dict(response)
        except Exception as exc:  # API/network errors are retried and preserved by caller.
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"OpenAI request failed after {attempts} attempts: {error}") from error


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "task",
        "discovery_model",
        "embedding_model",
        "consolidation_model",
        "code_generation_model",
        "discovery_calls",
        "rules_per_call",
        "clusters",
        "pca_dimensions",
        "kmeans_n_init",
        "kmeans_max_iter",
        "random_seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")
    if not bool(config.get("dataset_blind", False)):
        raise ValueError("Public workflow requires dataset_blind=true")
    if int(config["rules_per_call"]) != 10:
        raise ValueError("Frozen public prompts require rules_per_call=10")
    if int(config["discovery_calls"]) < 1 or int(config["clusters"]) < 1:
        raise ValueError("discovery_calls and clusters must be positive")
    get_task_prompts(str(config["task"]))


def _discover(
    client: OpenAI,
    config: dict[str, Any],
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = get_task_prompts(str(config["task"]))
    raw_dir = output / "raw" / "discovery"
    raw_dir.mkdir(parents=True, exist_ok=True)
    calls = int(config["discovery_calls"])
    workers = max(1, int(config.get("workers", 4)))

    def one(call_index: int) -> tuple[int, dict[str, Any]]:
        path = raw_dir / f"call_{call_index:04d}.json"
        if path.exists():
            return call_index, load_json(path)
        text, response = _request_text(
            client,
            model=str(config["discovery_model"]),
            user=spec.discovery,
            instructions=None,
            temperature=config.get("discovery_temperature"),
            max_output_tokens=int(config.get("discovery_max_output_tokens", 1400)),
        )
        rules = parse_numbered_rules(text, int(config["rules_per_call"]))
        record = {
            "call_index": call_index,
            "response_id": response.get("id"),
            "model": response.get("model", config["discovery_model"]),
            "output_text": text,
            "parsed_rules": rules,
            "parsed_rule_count": len(rules),
            "usage": response.get("usage"),
            "created_utc": utc_now(),
        }
        write_json(path, record)
        return call_index, record

    records: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, index): index for index in range(1, calls + 1)}
        for future in as_completed(futures):
            index, record = future.result()
            records[index] = record

    ordered = [records[index] for index in range(1, calls + 1)]
    incomplete = [row["call_index"] for row in ordered if row["parsed_rule_count"] != config["rules_per_call"]]
    if incomplete:
        raise RuntimeError(
            "Discovery parsing did not recover exactly ten rules for calls: "
            + ", ".join(map(str, incomplete[:20]))
        )

    rules: list[dict[str, Any]] = []
    for row in ordered:
        for position, text in enumerate(row["parsed_rules"], start=1):
            rules.append(
                {
                    "rule_index": len(rules) + 1,
                    "call_index": row["call_index"],
                    "position": position,
                    "text": text,
                }
            )
    write_json(output / "stages" / "01_discovered_rules.json", rules)
    return rules, ordered


def _embed_and_cluster(
    client: OpenAI,
    rules: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    stage_dir = output / "stages"
    embedding_path = stage_dir / "02_embeddings.npy"
    usage_path = stage_dir / "02_embedding_usage.json"
    if embedding_path.exists():
        vectors = np.load(embedding_path)
        embedding_usage = load_json(usage_path) if usage_path.exists() else []
    else:
        vectors_list: list[list[float]] = []
        embedding_usage: list[dict[str, Any]] = []
        batch_size = int(config.get("embedding_batch_size", 128))
        texts = [row["text"] for row in rules]
        for start in range(0, len(texts), batch_size):
            response = client.embeddings.create(
                model=str(config["embedding_model"]),
                input=texts[start : start + batch_size],
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors_list.extend(item.embedding for item in ordered)
            payload = _response_dict(response)
            embedding_usage.append(
                {
                    "batch_start": start,
                    "response_model": payload.get("model"),
                    "usage": payload.get("usage"),
                }
            )
        vectors = np.asarray(vectors_list, dtype=np.float32)
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, vectors)
        write_json(usage_path, embedding_usage)

    if vectors.shape[0] != len(rules):
        raise RuntimeError(f"Embedding row mismatch: {vectors.shape[0]} != {len(rules)}")
    clusters = int(config["clusters"])
    if clusters > len(rules):
        raise ValueError("clusters cannot exceed the number of discovered rules")
    normalized = normalize(vectors, norm="l2")
    components = min(int(config["pca_dimensions"]), normalized.shape[0] - 1, normalized.shape[1])
    reduced = PCA(n_components=components, random_state=int(config["random_seed"])).fit_transform(normalized)
    reduced = normalize(reduced, norm="l2")
    labels = KMeans(
        n_clusters=clusters,
        n_init=int(config["kmeans_n_init"]),
        max_iter=int(config["kmeans_max_iter"]),
        random_state=int(config["random_seed"]),
    ).fit_predict(reduced)

    counts = Counter(int(value) for value in labels)
    ordered_labels = sorted(counts, key=lambda label: (-counts[label], label))
    public_id = {label: index + 1 for index, label in enumerate(ordered_labels)}
    assignments: list[dict[str, Any]] = []
    for row, label in zip(rules, labels, strict=True):
        assignments.append({**row, "cluster_id": f"R{public_id[int(label)]:02d}"})
    summaries = [
        {"cluster_id": f"R{public_id[label]:02d}", "member_count": counts[label]}
        for label in ordered_labels
    ]
    write_json(stage_dir / "03_cluster_assignments.json", assignments)
    write_json(stage_dir / "03_cluster_summary.json", summaries)
    return assignments, summaries


def _consolidate(
    client: OpenAI,
    assignments: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
) -> list[dict[str, Any]]:
    spec = get_task_prompts(str(config["task"]))
    raw_dir = output / "raw" / "consolidation"
    max_members = int(config.get("max_cluster_prompt_rules", 2000))
    consolidated: list[dict[str, Any]] = []
    for summary in summaries:
        cluster_id = summary["cluster_id"]
        members = [row["text"] for row in assignments if row["cluster_id"] == cluster_id]
        if len(members) > max_members:
            raise RuntimeError(f"{cluster_id} exceeds max_cluster_prompt_rules={max_members}")
        path = raw_dir / f"{cluster_id}.json"
        if path.exists():
            record = load_json(path)
        else:
            prompt = spec.consolidation_user.format(
                rule_texts="\n".join(f"- {text}" for text in members)
            )
            text, response = _request_text(
                client,
                model=str(config["consolidation_model"]),
                instructions=spec.consolidation_system,
                user=prompt,
                temperature=config.get("consolidation_temperature", 0),
                max_output_tokens=int(config.get("consolidation_max_output_tokens", 900)),
            )
            record = {
                "cluster_id": cluster_id,
                "member_count": len(members),
                "representative_rule": text,
                "response_id": response.get("id"),
                "model": response.get("model", config["consolidation_model"]),
                "usage": response.get("usage"),
                "created_utc": utc_now(),
            }
            write_json(path, record)
        consolidated.append(record)
    write_json(output / "stages" / "04_consolidated_rules.json", consolidated)
    return consolidated


def _generate_code(
    client: OpenAI,
    consolidated: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
) -> list[dict[str, Any]]:
    spec = get_task_prompts(str(config["task"]))
    raw_dir = output / "raw" / "code_generation"
    code_dir = output / "generated_descriptors"
    manifest: list[dict[str, Any]] = []
    for rule_index, rule in enumerate(consolidated, start=1):
        cluster_id = str(rule["cluster_id"])
        slug = slugify(str(rule["representative_rule"]), f"descriptor_{rule_index:02d}")
        expected_name = f"rule_{rule_index:02d}_{slug}"
        raw_path = raw_dir / f"{cluster_id}.json"
        code_path = code_dir / f"{expected_name}.py"
        if raw_path.exists() and code_path.exists():
            record = load_json(raw_path)
            code = code_path.read_text(encoding="utf-8").strip()
        else:
            prompt = code_prompt(spec, rule_index, slug, str(rule["representative_rule"]))
            text, response = _request_text(
                client,
                model=str(config["code_generation_model"]),
                instructions=spec.code_system,
                user=prompt,
                temperature=config.get("code_generation_temperature", 0),
                max_output_tokens=int(config.get("code_generation_max_output_tokens", 2600)),
            )
            code = strip_code_fences(text)
            record = {
                "cluster_id": cluster_id,
                "expected_function_name": expected_name,
                "response_id": response.get("id"),
                "model": response.get("model", config["code_generation_model"]),
                "usage": response.get("usage"),
                "created_utc": utc_now(),
            }
            write_json(raw_path, record)
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code.rstrip() + "\n", encoding="utf-8")

        audit = audit_generated_code(code, expected_name)
        manifest.append(
            {
                "rule_id": f"R{rule_index:02d}",
                "cluster_id": cluster_id,
                "representative_rule": rule["representative_rule"],
                "function_name": expected_name,
                "code_path": str(code_path.relative_to(output)),
                "code_sha256": sha256_text(code),
                "static_audit": audit.to_dict(),
                "response_id": record.get("response_id"),
                "model": record.get("model"),
            }
        )
    write_json(output / "stages" / "05_generated_code_manifest.json", manifest)
    return manifest


def compile_catalog(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run or resume the public dataset-blind CRISP compilation protocol."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    config_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    config = load_json(config_path)
    _validate_config(config)
    output.mkdir(parents=True, exist_ok=True)

    spec = get_task_prompts(str(config["task"]))
    frozen_path = output / "config_frozen.json"
    if frozen_path.exists() and load_json(frozen_path) != config:
        raise RuntimeError("Output directory contains a different frozen configuration")
    write_json(frozen_path, config)
    write_json(
        output / "provenance.json",
        {
            "created_or_resumed_utc": utc_now(),
            "config_source": str(config_path),
            "config_sha256": sha256_bytes(config_path.read_bytes()),
            "task": config["task"],
            "dataset_blind": True,
            "dataset_paths_read": [],
            "models": {key: value for key, value in config.items() if key.endswith("_model")},
            "prompt_sha256": {
                "discovery": sha256_text(spec.discovery),
                "consolidation_system": sha256_text(spec.consolidation_system),
                "consolidation_user": sha256_text(spec.consolidation_user),
                "code_system": sha256_text(spec.code_system),
                "code_user": sha256_text(spec.code_user),
            },
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("openai", "numpy", "scikit-learn")
            },
        },
    )

    client = OpenAI()
    rules, discovery_records = _discover(client, config, output)
    assignments, summaries = _embed_and_cluster(client, rules, config, output)
    consolidated = _consolidate(client, assignments, summaries, config, output)
    code_manifest = _generate_code(client, consolidated, config, output)
    passed = sum(bool(row["static_audit"]["passed"]) for row in code_manifest)
    usage = {
        "discovery_calls": len(discovery_records),
        "discovered_rules": len(rules),
        "clusters": len(summaries),
        "generated_functions": len(code_manifest),
        "static_audit_passed": passed,
        "static_audit_failed": len(code_manifest) - passed,
    }
    write_json(output / "run_summary.json", usage)
    return usage
