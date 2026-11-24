"""Evaluate a fitted CRISP forest on named descriptor columns."""
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import BadZipFile, Path as ZipPath, ZipFile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = "CRISP_RF_models.zip"


@contextmanager
def model_folder(task):
    if task not in ("formation_energy", "ionic_conductivity", "synthesizability"):
        raise ValueError("Unknown model.")
    archive = ROOT / WEIGHTS
    if archive.is_file():
        try:
            with ZipFile(archive) as bundle:
                folder = ZipPath(bundle, at=f"models/{task}/")
                if not (folder / "model.json").is_file():
                    raise ValueError(f"{WEIGHTS} does not contain the {task} model.")
                yield folder
        except BadZipFile as exc:
            raise ValueError(f"Cannot read {WEIGHTS}. Download a fresh copy.") from exc
        return
    folder = ROOT / "models" / task
    if not (folder / "model.json").is_file():
        raise FileNotFoundError(f"Place {WEIGHTS} beside Predict.ipynb and click Predict again. No extraction is needed.")
    yield folder


def predict_frame(frame, task):
    with model_folder(task) as folder:
        return _predict_frame(frame, folder)


def _predict_frame(frame, folder):
    spec = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if frame.columns.has_duplicates or set(frame.columns) != set(spec["columns"]):
        raise ValueError("The descriptor columns do not match this model.")
    x = frame[spec["columns"]].to_numpy(dtype=np.float32 if spec["kind"] == "pu" else np.float64,
                                         copy=True)
    if np.isinf(x).any():
        raise ValueError("Infinite descriptor value.")
    p = spec["preprocessing"]
    if p["medians"] is not None:
        x = np.where(np.isnan(x), np.asarray(p["medians"], dtype=x.dtype), x)
    if p["keep"] is not None:
        x = x[:, p["keep"]]
    if p["mean"] is not None:
        x = (x - np.asarray(p["mean"])) / np.asarray(p["scale"])
    if not np.isfinite(x).all():
        raise ValueError("The model requires finite input values.")
    x = x.astype(np.float32)
    groups, counts = {}, {}
    rows = np.arange(len(x))[:, None]
    for chunk in spec["chunks"]:
        name = chunk["file"]
        if not isinstance(name, str) or not re.fullmatch(r"trees_[0-9]{3}\.npz", name):
            raise ValueError("Unrecognized forest parameter file.")
        member = folder / name
        if not member.is_file():
            raise ValueError(f"The model is missing {name}. Download a fresh copy of {WEIGHTS}.")
        with np.load(BytesIO(member.read_bytes()), allow_pickle=False) as z:
            start, feat, right, val = (z[k] for k in ("start", "feature", "right", "value"))
        nodes = np.broadcast_to(start, (len(x), len(start))).copy()
        while True:
            f = feat[nodes]
            branches = f >= 0
            if not branches.any():
                break
            to_right = x[rows, np.maximum(f, 0)] > val[nodes]
            nodes += np.where(branches, np.where(to_right, right[nodes], 1), 0).astype(nodes.dtype)
        group = chunk["group"]
        groups[group] = groups.get(group, 0.) + val[nodes].sum(axis=1)
        counts[group] = counts.get(group, 0) + len(start)
    result = np.zeros(len(x))
    for group in sorted(groups):
        result += groups[group] / counts[group]
    return result / len(groups), spec.get("threshold")
