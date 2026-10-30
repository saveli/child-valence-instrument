"""Clip-level feature table and design matrices.

`clip_row` computes, for one clip, every candidate feature the nested
selection can use: P/W for each stream, Cf, Cb, and Cr at every candidate
window start. `design_matrix` cuts one design out of that wide table.

Wide-table column names:
    <stream>|<P or W column>     e.g. effnat|childp4_mean, effnat|e1.0_2.0
    cb_<...>                      Cf and Cb
    w<start>|fr_<cue>_<stat>      Cr at that window start
Design (head) feature names drop the prefixes, as in the paper code:
    P, W columns as in affect_blocks; fr_<cue>_<stat> for Cr.
Missing values (no face, window past clip end) become 0 in the design matrix,
exactly as in the paper.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import affect_blocks as ab
from . import cues as cu
from .constants import BLOCKS, FR_STARTS


def block_columns() -> dict[str, list[str]]:
    """Unprefixed column names per block, in the paper's order."""
    cf = sorted([f"cb_{c}_{s}" for c in cu.FACIAL_CUES for s in ("mean", "max")])
    cr = sorted([f"fr_{c}_{s}" for c in cu.FACIAL_CUES for s in ("mean", "max")])
    cb = sorted([f"cb_{c}" for c in cu.BODY_CUES])
    return {"P": list(ab.P_COLS), "W": list(ab.W_COLS), "Cf": cf, "Cr": cr, "Cb": cb}


def clip_row(args: tuple) -> dict:
    clip_id, export_dir, mp_dir, starts, streams, pt_path = args
    row: dict = {"clip_id": clip_id}
    csv = Path(export_dir) / f"{clip_id}.csv"
    npy = Path(export_dir) / f"{clip_id}_emotiefflib.npy"
    if csv.exists() and npy.exists():
        row.update(ab.affect_features(csv, npy, ab.shipped_classifier(pt_path), streams))
    mp = Path(mp_dir) / f"{clip_id}.npz"
    if mp.exists():
        z = cu.load_mediapipe(mp)
        row.update(cu.full_clip_cues(z))
        row.update(cu.first_reaction_cues(z, starts))
    return row


def build_table(clip_ids, export_dir: Path, mp_dir: Path, starts=FR_STARTS,
                streams=("effnat", "shipped"), workers: int = 4,
                pt_path: Path | None = None) -> pd.DataFrame:
    tasks = [(c, export_dir, mp_dir, tuple(starts), tuple(streams), pt_path) for c in clip_ids]
    if workers <= 1:
        rows = [clip_row(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(clip_row, tasks, chunksize=8))
    return pd.DataFrame(rows)


def wide_name(block: str, col: str, stream: str, start: float) -> str:
    if block in ("P", "W"):
        return f"{stream}|{col}"
    if block == "Cr":
        return f"w{float(start)}|{col}"
    return col


def design_matrix(df: pd.DataFrame, stream: str, blocks=BLOCKS, start: float = 1.0,
                  feature_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """X (n, p) and the unprefixed feature names.

    With `feature_names` (from a fitted head) the columns follow that order.
    """
    cols = block_columns()
    pairs = [(b, c) for b in blocks for c in cols[b]]
    if feature_names is not None:
        block_of = {c: b for b, cs in cols.items() for c in cs}
        pairs = [(block_of[c], c) for c in feature_names]
    wide = [wide_name(b, c, stream, start) for b, c in pairs]
    missing = [w for w in wide if w not in df.columns]
    if missing:
        import warnings
        warnings.warn(f"{len(missing)} design columns absent from the table and set to 0 "
                      f"(e.g. {missing[:3]}); expected only if no clip had a usable face")
    X = (df.reindex(columns=wide).replace([np.inf, -np.inf], np.nan)
         .fillna(0.0).to_numpy(float))
    return X, [c for _, c in pairs]
