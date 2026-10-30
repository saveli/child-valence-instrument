"""Child-disjoint evaluation and nested design selection.

Protocol (as in the paper):
  * outer GroupKFold(10) by child (`group`), scaler + ridge fit on training
    folds only, r and CCC on the pooled out-of-fold predictions;
  * nested selection: inside each outer fold, GroupKFold(5) by child on the
    training children scores every candidate design
        stream x non-empty subset of {P, W, Cf, Cr, Cb} x Cr window start
    (designs without P and W enter once, designs without Cr only at 1.5 s, as
    in the paper code) and the best inner pooled r is refit on the outer
    training fold and scored on the held-out children;
  * 95% percentile intervals from 2,000 bootstrap resamples of children over
    the stored out-of-fold predictions.
GroupKFold is deterministic (no shuffling); fold membership depends on the row
order and the group sizes of your table.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from .constants import BLOCKS, FR_STARTS, N_INNER, N_OUTER, SEED_BOOT, STREAMS
from .features import design_matrix
from .model import fit_predict


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my, vx, vy = x.mean(), y.mean(), x.var(), y.var()
    return float(2 * ((x - mx) * (y - my)).mean() / (vx + vy + (mx - my) ** 2))


def metrics(y, yhat) -> dict:
    return {"r": float(np.corrcoef(y, yhat)[0, 1]),
            "rho": float(stats.spearmanr(y, yhat).statistic),
            "ccc": ccc(yhat, y)}


def grouped_oof(X, y, groups, n_splits=N_OUTER):
    yhat = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        yhat[te] = fit_predict(X[tr], y[tr], X[te])
    return yhat


def child_bootstrap(y, preds: dict, groups, n=2000, seed=SEED_BOOT) -> dict:
    """Per prediction column: (n, 2) array of (r, CCC) over child resamples."""
    rng = np.random.default_rng(seed)
    kids = pd.unique(groups)
    idx_by = {k: np.flatnonzero(groups == k) for k in kids}
    out = {k: [] for k in preds}
    for _ in range(n):
        ii = np.concatenate([idx_by[k] for k in rng.choice(kids, len(kids))])
        for k, p in preds.items():
            out[k].append((np.corrcoef(y[ii], p[ii])[0, 1], ccc(p[ii], y[ii])))
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- nested selection

G: dict = {}


def _init(Xblocks, y, groups):
    G.update(Xblocks=Xblocks, y=y, groups=groups)


def candidates(streams=STREAMS, starts=FR_STARTS, ref_start=1.5):
    """(blocks with stream-tagged P/W, start) and readable metadata."""
    # Stream-major order as in the paper code; np.argmax keeps the first of tied
    # candidates, so the first stream wins ties.
    cands, meta = [], []
    for h in tuple(streams) + ("-",):
        for k in range(1, len(BLOCKS) + 1):
            for sub in itertools.combinations(BLOCKS, k):
                if (h == "-") == (("P" in sub) or ("W" in sub)):
                    continue
                tagged = tuple(f"{b}:{h}" if b in ("P", "W") else b for b in sub)
                for s in (starts if "Cr" in sub else (ref_start,)):
                    cands.append((tagged, s))
                    meta.append({"stream": h, "blocks": "+".join(sub), "start": s})
    return cands, meta


def candidate_X(cand, idx=None):
    blocks, start = cand
    X = np.hstack([G["Xblocks"][f"Cr@{start}" if b == "Cr" else b] for b in blocks])
    return X if idx is None else X[idx]


def inner_scores(args):
    cand, tr = args
    y, groups = G["y"][tr], G["groups"][tr]
    X = candidate_X(cand, tr)
    p = np.full(len(tr), np.nan)
    for itr, ite in GroupKFold(N_INNER).split(X, y, groups):
        p[ite] = fit_predict(X[itr], y[itr], X[ite])
    return float(np.corrcoef(p, y)[0, 1]), ccc(p, y)


def build_blocks(df: pd.DataFrame, streams=STREAMS, starts=FR_STARTS) -> dict:
    Xb = {}
    for h in streams:
        Xb[f"P:{h}"], _ = design_matrix(df, h, ("P",))
        Xb[f"W:{h}"], _ = design_matrix(df, h, ("W",))
    Xb["Cf"], _ = design_matrix(df, streams[0], ("Cf",))
    Xb["Cb"], _ = design_matrix(df, streams[0], ("Cb",))
    for s in starts:
        Xb[f"Cr@{s}"], _ = design_matrix(df, streams[0], ("Cr",), start=s)
    return Xb


def nested_selection(df: pd.DataFrame, y, groups, streams=STREAMS, starts=FR_STARTS,
                     workers: int = 4):
    """Outer OOF predictions of the nested pick and the per-fold picks."""
    Xb = build_blocks(df, streams, starts)
    cands, meta = candidates(streams, starts)
    oof = np.full(len(y), np.nan)
    picks = []
    _init(Xb, y, groups)
    folds = list(GroupKFold(N_OUTER).split(df, y, groups))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(Xb, y, groups)) as ex:
        for f, (tr, te) in enumerate(folds):
            sc = list(ex.map(inner_scores, [(c, tr) for c in cands], chunksize=2))
            r_in = np.array([s[0] for s in sc])
            i = int(np.argmax(r_in))
            oof[te] = fit_predict(candidate_X(cands[i], tr), y[tr], candidate_X(cands[i], te))
            picks.append({"fold": f, **meta[i], "inner_r": float(r_in[i]),
                          "inner_ccc": float(sc[i][1])})
    return oof, pd.DataFrame(picks)
