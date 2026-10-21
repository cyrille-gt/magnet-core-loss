"""MagNet (Princeton) steady-state training-ready release: download and loading helpers.

Raw data live under DATA_ROOT (D:\\Claude\\magnet\\steady-state-training-ready):
    data/scalar/part-00000.parquet        operating conditions + measured loss (all 15 materials)
    data/sequence/part-000NN.parquet      1024-point B(t), H(t) waveforms, joined on measurement_id
    manifest.json, README.md, ...         release metadata (sha256 of every shard in manifest.json)

Source: https://huggingface.co/datasets/minjiechen/MagNet (CC BY 4.0).
Cite: Li et al., "How MagNet: Machine Learning Framework for Modeling Power Magnetic
Material Characteristics", IEEE Trans. Power Electron., 2023, doi:10.1109/TPEL.2023.3309232.

Verified field definitions (checked on N87):
    flux_density_t           = (max B - min B) / 2  (AC peak flux density)
    b_t                      = AC component only (zero mean)
    h_a_per_m                = includes the DC bias H_dc
    volumetric_loss_w_per_m3 = f * loop area of H dB, within 0.1 %
Trapezoidal waveforms: segment order rise / dwell / fall / dwell; the dwell
segments are NOT flat (non-zero dB/dt).

Note: in this environment pandas.read_parquet fails (pyarrow 25 vs pandas 2.3
extension-type conflict), so tables are read with pyarrow directly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

# Override with the MAGNET_DATA_DIR environment variable (parent of steady-state-training-ready\ and derived\).
MAGNET_DIR = Path(os.environ.get("MAGNET_DATA_DIR", r"D:\Claude\magnet"))
DATA_ROOT = MAGNET_DIR / "steady-state-training-ready"
DERIVED_ROOT = MAGNET_DIR / "derived"
HF_BASE = ("https://huggingface.co/datasets/minjiechen/MagNet/resolve/main/"
           "releases/magnet-steady-state-training-ready/")
# needs huggingface.co and us.aws.cdn.hf.co (redirect target) reachable

# Sequence shards holding each material (determined from measurement_id; N87 verified).
SEQUENCE_PARTS = {"N87": [9, 10, 11]}

NSAMP = 1024


def _local_path(release_path: str, root: Path = DATA_ROOT) -> Path:
    """Map a release path (data/scalar/full/part-00000.parquet) to the local layout."""
    return root / release_path.replace("/full/", "/").replace("/", os.sep)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(release_path: str, root: Path = DATA_ROOT, verify: bool = True) -> Path:
    """Download one release file if missing; verify its sha256 against manifest.json."""
    dst = _local_path(release_path, root)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with requests.get(HF_BASE + release_path, stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        os.replace(tmp, dst)
    if verify and release_path.endswith(".parquet"):
        man = root / "manifest.json"
        if not man.exists():
            download("manifest.json", root, verify=False)
        expected = {e["path"]: e["sha256"] for e in json.loads(man.read_text("utf-8"))["files"]}
        if release_path in expected and _sha256(dst) != expected[release_path]:
            raise IOError(f"sha256 mismatch for {dst}")
    return dst


def scalar_path(root: Path = DATA_ROOT) -> Path:
    return download("data/scalar/full/part-00000.parquet", root)


def sequence_path(part: int, root: Path = DATA_ROOT) -> Path:
    return download(f"data/sequence/full/part-{part:05d}.parquet", root)


def load_scalar(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Scalar table as a DataFrame (list column quality_rejection_reasons dropped)."""
    t = pq.read_table(scalar_path(root))
    cols = [c for c in t.column_names if c != "quality_rejection_reasons"]
    return pd.DataFrame({c: t.column(c).to_numpy(zero_copy_only=False) for c in cols})


def iter_waveforms(material: str, scalar: pd.DataFrame, parts=None, root: Path = DATA_ROOT,
                   with_h: bool = False):
    """Yield (measurement_ids, B[n,1024] float64[, H]) per parquet row group for one material."""
    parts = parts if parts is not None else SEQUENCE_PARTS[material]
    is_mat = set(scalar.loc[scalar.material == material, "measurement_id"])
    cols = ["measurement_id", "b_t"] + (["h_a_per_m"] if with_h else [])
    for part in parts:
        pf = pq.ParquetFile(sequence_path(part, root))
        for rg in range(pf.metadata.num_row_groups):
            tb = pf.read_row_group(rg, columns=cols)
            ids = tb.column(0).to_numpy(zero_copy_only=False)
            m = np.fromiter((x in is_mat for x in ids), bool, len(ids))
            if not m.any():
                continue
            B = np.stack(tb.column(1).to_numpy(zero_copy_only=False)[m]).astype(np.float64)
            if with_h:
                H = np.stack(tb.column(2).to_numpy(zero_copy_only=False)[m]).astype(np.float64)
                yield ids[m], B, H
            else:
                yield ids[m], B
