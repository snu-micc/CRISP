"""Read a CIF and calculate a frozen descriptor catalog."""
import hashlib
import importlib.util
import inspect
import json
import math
import re
import warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = {"R01", "R06", "R10", "R19", "R23", "R48"}


@lru_cache(maxsize=5)
def rules(task, edition):
    if (edition, task) not in {
        ("original","synthesizability"), ("original","formation_energy"),
        ("original","ionic_conductivity"), ("qwen","synthesizability"), ("qwen","formation_energy")
    }:
        raise ValueError("Unknown catalog.")
    return json.loads((ROOT / "catalogs" / edition / (task + ".json")).read_text(encoding="utf8"))


@lru_cache(maxsize=5)
def _functions(task, edition):
    rules(task, edition)
    source = ROOT / "catalogs" / edition / (task + ".py")
    spec = importlib.util.spec_from_file_location(f"crisp_{edition}_{task}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(task=None):
    name = "Li3OCl.cif" if task=="ionic_conductivity" else "BaTe.cif"
    return (ROOT / "examples" / name).read_bytes()


def example_values(data):
    ref = json.loads((ROOT / "examples" / "BaTe.json").read_text())
    return ref["descriptors"] if hashlib.sha256(data).hexdigest() == ref["cif_sha256"] else {}


@contextmanager
def _mp_session(module, key, material_id, structure):
    from mp_api.client import MPRester
    if not key or not re.fullmatch(r"mp-[0-9]+", material_id or ""):
        raise ValueError("Enter the Materials Project ID for this CIF and an API key.")
    before = module.API_KEY
    cache = [getattr(module, n) for n in ("_summary_doc","_chemsys_count","_prototype_structures")]
    try:
        module.API_KEY = key
        for obj in cache: obj.cache_clear()
        with MPRester(key) as client:
            docs = client.materials.summary.search(material_ids=[material_id],
                fields=["structure","energy_above_hull","band_gap","formation_energy_per_atom"])
        if not docs or any(getattr(docs[0], prop, None) is None for prop in
                           ("structure","energy_above_hull","band_gap","formation_energy_per_atom")):
            raise ValueError("The Materials Project entry lacks the required reference fields.")
        from pymatgen.analysis.structure_matcher import StructureMatcher
        if not StructureMatcher().fit(structure, docs[0].structure):
            raise ValueError("The CIF does not match the supplied Materials Project ID.")
        yield
    finally:
        module.API_KEY = before
        for obj in cache: obj.cache_clear()


def describe(data, task, edition="original", *, mp_id="", mp_key="", band_gap=None, above_hull=None):
    """Return a human-readable table and the ordered numerical columns."""
    from contextlib import nullcontext, redirect_stdout
    from io import StringIO
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifParser
    text = data.decode("utf-8-sig")
    rows = rules(task, edition)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        struct = (Structure.from_str(text, fmt="cif") if edition=="qwen" else
                  CifParser.from_str(text, occupancy_tolerance=1.0).parse_structures(primitive=False)[0])
    module = _functions(task, edition)
    fixed = example_values(data)
    live = edition == "original" and task == "synthesizability" and not fixed
    if task == "synthesizability" and edition == "qwen":
        if fixed and band_gap is None and above_hull is None:
            above_hull, band_gap = fixed["R01"], fixed["R06"]
        if any(v is None or not math.isfinite(float(v)) for v in (band_gap, above_hull)):
            raise ValueError("Enter the band gap (eV) and energy above hull (eV/atom) for this structure.")
        struct.properties.update(band_gap=float(band_gap), energy_above_hull=float(above_hull))
    session = _mp_session(module, mp_key, mp_id, struct) if live else nullcontext()
    values = []
    with session, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in rows:
            rid = row["rule"]
            if edition == "original" and task == "synthesizability" and rid in fixed:
                values.append(fixed[rid])
                continue
            fn = getattr(module, row["function"])
            kwargs = {}
            if "mp_id" in inspect.signature(fn).parameters:
                kwargs["mp_id"] = mp_id
            if "api_key" in inspect.signature(fn).parameters:
                kwargs["api_key"] = mp_key
            messages = StringIO()
            try:
                with redirect_stdout(messages):
                    value = float(fn(struct.copy() if edition=="qwen" else struct, **kwargs))
                if live and rid in EXTERNAL and messages.getvalue().strip():
                    raise ValueError("Reference lookup did not complete.")
            except Exception as exc:
                if edition != "qwen":
                    raise ValueError(f"Could not compute {rid}: {row['name']}.") from exc
                value = math.nan
            values.append(value if math.isfinite(value) else math.nan)
    display = pd.DataFrame({"Rule":[r["rule"] for r in rows], "Descriptor":[r["name"] for r in rows],
                            "Value":values})
    frame = pd.DataFrame([values],columns=[r["column"] for r in rows])
    return struct.composition.reduced_formula, display, frame
