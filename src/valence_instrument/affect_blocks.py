"""P block (21) and W block (19) from the EmotiEffLib per-frame stream.

Streams (the paper's nested selection chose among five; this release ships the
two that need no extra weights):
  effnat   EmotiEffLib's own 8-class softmax + its native valence  <- DEPLOYED
  shipped  the same softmax + the fixed categorical proxy
           P(happy) - P(sad) - P(anger)
The paper's other three streams (EmoNet; child-recalibrated and adult-only
refits of the EmotiEffLib head) are not shipped: nested selection never chose
them, and their training sets restrict derived models.

Class probabilities are recomputed per frame from the stored 1280-d embedding
with the package's own classifier rows 0-7 (the DISCOVER module stores only the
arg-max label). Column names keep the paper code's names: childp<k>_* is class
k in EmotiEffLib's package order (Anger, Contempt, Disgust, Fear, Happiness,
Neutral, Sadness, Surprise); child_proxy_* is the selected valence stream.

Pooling, unchanged from the paper code:
  P (21): over confident frames (BlazeFace >= .5): p0..7 mean/std +
          valence mean/std/min/max/last-minus-first. The confidence mask is
          applied only if the per-frame arrays have the CSV's length (as in the
          paper code; with this release's exporter the lengths always match).
          Needs >= 4 confident frames, else NaN.
  W (19): over ALL frames (no-face frames carry valence 0, as the module
          writes them): window means e<start>_<dur> for starts 0-2 s and
          durations 1/2 s (NaN iff the start is past the clip end, truncated
          otherwise) and min/max/signed extreme over the first 2/3/4 s of the
          loess-smoothed series (span 1, degree 2). Needs >= 4 frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (CONF, EMOTIEFFLIB_CLASSES, EMOTIEFFLIB_PT_SHA256, EXT_SPANS, FPS,
                        PROXY_IDX_SHIPPED, WIN_DURS, WIN_STARTS)

P_COLS = ([f"childp{k}_{s}" for k in range(8) for s in ("mean", "std")]
          + [f"child_proxy_{s}" for s in ("mean", "std", "min", "max", "delta")])
W_COLS = ([f"e{s}_{d}" for s in WIN_STARTS for d in WIN_DURS]
          + [f"{k}{s}" for s in EXT_SPANS for k in ("min", "max", "ext")])

_HEAD: dict = {}


def shipped_classifier(pt_path: Path | None = None) -> dict:
    """Rows 0-7 of EmotiEffLib's shipped Linear(1280, 10) (package order).

    Read from the weights file the `emotiefflib` package downloads to
    ~/.emotiefflib/ (instantiate EmotiEffLibRecognizer once to trigger it).
    Not redistributed here.
    """
    if not _HEAD:
        import hashlib
        import torch
        p = Path(pt_path) if pt_path else Path.home() / ".emotiefflib" / "enet_b0_8_va_mtl.pt"
        if not p.exists():
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            EmotiEffLibRecognizer(engine="torch", model_name="enet_b0_8_va_mtl", device="cpu")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != EMOTIEFFLIB_PT_SHA256:
            print(f"WARNING: {p} sha256 differs from the weights used for the paper")
        m = torch.load(p, map_location="cpu", weights_only=False)
        fc = m.classifier
        assert isinstance(fc, torch.nn.Linear) and tuple(fc.weight.shape) == (10, 1280), fc
        _HEAD.update(W=fc.weight.detach().numpy()[:8].astype(np.float32),
                     b=fc.bias.detach().numpy()[:8].astype(np.float32),
                     classes=EMOTIEFFLIB_CLASSES)
    return _HEAD


def head_probs(z: np.ndarray, head: dict) -> np.ndarray:
    lg = z @ head["W"].T + head["b"]
    lg -= lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def loess_smooth(v: np.ndarray) -> np.ndarray:
    from skmisc.loess import loess
    x = np.arange(1, len(v) + 1, dtype=float)
    lo = loess(x, v.astype(float), span=1.0, degree=2, family="gaussian")
    lo.fit()
    return lo.outputs.fitted_values


def pool(row: dict, h: str, p: np.ndarray, x: np.ndarray, ok: np.ndarray) -> None:
    """Fill '<h>|<col>' for the P and W blocks. p: (n, 8), x: (n,), ok: (n,) bool."""
    if ok.sum() >= 4:
        pm = p[ok] if len(p) == len(ok) else p
        px = x[ok] if len(x) == len(ok) else x
        for k in range(8):
            row[f"{h}|childp{k}_mean"] = float(pm[:, k].mean())
            row[f"{h}|childp{k}_std"] = float(pm[:, k].std())
        row.update({f"{h}|child_proxy_mean": float(px.mean()),
                    f"{h}|child_proxy_std": float(px.std()),
                    f"{h}|child_proxy_min": float(px.min()),
                    f"{h}|child_proxy_max": float(px.max()),
                    f"{h}|child_proxy_delta": float(px[-1] - px[0])})
    if len(x) >= 4:
        n = len(x)
        for s in WIN_STARTS:
            for d in WIN_DURS:
                a = int(s * FPS)
                b = a + int(d * FPS)
                row[f"{h}|e{s}_{d}"] = float(x[a:min(b, n)].mean()) if a < n else np.nan
        sm = loess_smooth(x)
        for sp in EXT_SPANS:
            seg = sm[:int(sp * FPS)]
            lo, hi = float(seg.min()), float(seg.max())
            row[f"{h}|min{sp}"], row[f"{h}|max{sp}"] = lo, hi
            row[f"{h}|ext{sp}"] = hi if abs(hi) >= abs(lo) else lo


def affect_features(frames_csv: Path, emb_npy: Path, head: dict,
                    streams=("effnat", "shipped")) -> dict:
    row: dict = {}
    f = pd.read_csv(frames_csv, usecols=["blazeface_confidence", "emotiefflib_valence"])
    z = np.load(emb_npy).astype(np.float32)
    ok = (f["blazeface_confidence"] >= CONF).to_numpy()
    p = head_probs(z, head)
    if "effnat" in streams:
        pool(row, "effnat", p, f["emotiefflib_valence"].to_numpy(float), ok)
    if "shipped" in streams:
        ih, isd, ian = PROXY_IDX_SHIPPED
        pool(row, "shipped", p, p[:, ih] - p[:, isd] - p[:, ian], ok)
    return row
