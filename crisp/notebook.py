"""Controls for the two CRISP notebooks."""
from datetime import datetime
from html import escape
from pathlib import Path
import os
import ipywidgets as w
from IPython.display import display, Markdown, FileLink, clear_output

from .material import describe, sample, ROOT
from .scoring import model_folder, predict_frame

TASKS = [("Formation energy","formation_energy"),("Synthesizability","synthesizability"),
         ("Room-temperature ionic conductivity","ionic_conductivity")]
STYLE = {"description_width":"initial"}


def prediction_panel():
    edition=w.Dropdown(options=[("Original CRISP · prediction","original"),
                                ("Original CRISP · descriptors only","original_descriptors"),
                                ("Qwen · descriptors only","qwen")],
                       description="Use:",style=STYLE,layout=w.Layout(width="auto"))
    task=w.Dropdown(options=TASKS,description="Property:",style=STYLE)
    upload=w.FileUpload(accept=".cif",multiple=False,description="Choose CIF")
    default=w.Button(description="Use sample CIF")
    filename=w.HTML("Using BaTe (mp-1000).")
    mpid=w.Text(description="Materials Project ID:",placeholder="mp-…",style=STYLE)
    mpkey=w.Password(description="Materials Project API key:",style=STYLE)
    gap=w.Text(description="Band gap (eV):",style=STYLE)
    hull=w.Text(description="Energy above hull (eV/atom):",style=STYLE)
    extras=w.VBox()
    results=w.Output()
    go=w.Button(description="Predict",button_style="primary",icon="play")
    def uploaded(value=None):
        saved=upload.value
        if not saved: return None
        item=next(iter(saved.values())) if isinstance(saved,dict) else saved[0]
        return item
    def refresh(change=None):
        item=uploaded()
        label="ideal cubic Li₃OCl" if task.value=="ionic_conductivity" else "BaTe (mp-1000)"
        filename.value="Using "+escape(item.get("name","your CIF"))+"." if item else "Using "+label+"."
        extras.children=()
        if task.value=="synthesizability" and item:
            extras.children=(gap,hull) if edition.value=="qwen" else (mpid,mpkey)
        go.description="Predict" if edition.value=="original" else "Calculate descriptors"
        go.layout.width="195px"
        results.clear_output()
    def change_edition(change):
        previous=task.value
        options=TASKS[:2] if edition.value=="qwen" else TASKS
        task.options=options
        if previous in dict(options).values(): task.value=previous
        refresh()
    def reset(_):
        upload.value={} if isinstance(upload.value,dict) else ()
        mpid.value=mpkey.value=gap.value=hull.value=""
        refresh()
    def run(_):
        item=uploaded()
        go.disabled=True
        results.clear_output()
        with results:
            try:
                if edition.value=="original":
                    # Check for the weights before a CIF or Materials Project lookup is needed.
                    with model_folder(task.value):
                        pass
                data=bytes(item["content"]) if item else sample(task.value)
                catalog="qwen" if edition.value=="qwen" else "original"
                formula,tab,frame=describe(data,task.value,catalog,
                    mp_id=mpid.value.strip(),mp_key=mpkey.value.strip() or os.environ.get("MP_API_KEY",""),
                    band_gap=float(gap.value) if item and gap.value.strip() else None,
                    above_hull=float(hull.value) if item and hull.value.strip() else None)
                if edition.value=="original":
                    scores,threshold=predict_frame(frame,task.value); score=float(scores[0])
                    if threshold is not None:
                        label="above threshold" if score>=threshold else "below threshold"
                        message=f"PU score **{score:.4f}** ({label}; threshold {threshold:.4f})"
                    elif task.value=="formation_energy": message=f"Formation energy **{score:.4f} eV/atom**"
                    else: message=f"Room-temperature ionic conductivity **{10**score:.3g} S/cm** (log₁₀ σ = {score:.3f})"
                    display(Markdown(f"**{formula}** · {message}"))
                else:
                    display(Markdown(f"**{formula}** · {len(tab)} {task.label.lower()} descriptors"))
                display(tab.style.hide(axis="index").format({"Value":"{:.6g}"},na_rep="not available"))
                folder=ROOT/"my_results";folder.mkdir(exist_ok=True)
                stem=task.value+"_"+edition.value+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                file=folder/(stem+".csv")
                tab.to_csv(file,index=False)
                display(FileLink(str(file.relative_to(ROOT)),result_html_prefix="Descriptor values: "))
            except Exception as exc:
                display(w.HTML("<b>Could not finish:</b> "+escape(str(exc))))
            finally:
                mpkey.value="";go.disabled=False
    edition.observe(change_edition,names="value");task.observe(refresh,names="value")
    upload.observe(refresh,names="value");default.on_click(reset);go.on_click(run)
    refresh()
    return w.VBox([edition,task,w.HBox([upload,default]),filename,extras,go,results])


def generation_panel():
    from .generation import build_formation_catalog
    size=w.Dropdown(options=[("30 proposals · 3 calls · 3 rules",False),
                             ("Study settings · 10,000 proposals · 1,000 calls · 25 rules",True)],
                    description="Catalog size:",style=STYLE,layout=w.Layout(width="auto"))
    key=w.Password(description="OpenAI API key:",style=STYLE)
    confirm=w.Checkbox(description="Run paid API requests with my account",value=False,indent=False)
    go=w.Button(description="Generate rules",button_style="primary",disabled=True)
    status=w.HTML();result=w.Output()
    confirm.observe(lambda change:setattr(go,"disabled",not change["new"]),names="value")
    size.observe(lambda change:setattr(confirm,"value",False),names="value")
    def run(_):
        if not confirm.value: return
        from openai import OpenAI
        token=key.value.strip() or os.environ.get("OPENAI_API_KEY","")
        if not token:
            status.value="Enter an API key or set OPENAI_API_KEY.";return
        go.disabled=True;confirm.disabled=True;size.disabled=True
        result.clear_output()
        try:
            with OpenAI(api_key=token,max_retries=0,timeout=180.) as client:
                out=build_formation_catalog(client,ROOT/"my_catalogs",full=size.value,
                    progress=lambda s:setattr(status,"value",escape(s)))
            status.value="Catalog saved."
            with result:
                display(FileLink(str((out/"rules.json").relative_to(ROOT)),result_html_prefix="Rules: "))
                display(Markdown("Python drafts are in `"+str((out/"functions").relative_to(ROOT))+"`."))
        except Exception as exc:
            # Provider messages can echo credentials. Never persist or display them.
            status.value="Generation stopped ("+escape(type(exc).__name__)+"). Completed stages remain in my_catalogs/."
        finally:
            key.value="";confirm.value=False;confirm.disabled=False;size.disabled=False;go.disabled=True
    go.on_click(run)
    return w.VBox([size,key,confirm,go,status,result])
