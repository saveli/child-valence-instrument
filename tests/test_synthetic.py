"""Smoke test on synthetic arrays only (no video, no models, no real data).

Checks that the feature code runs end to end, produces 117 deployed features,
and that train / evaluate / nested selection / predict run. Says nothing about
correctness against the paper; that check is in the private export script.

    python -m pytest tests/   or   python tests/test_synthetic.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valence_instrument import affect_blocks as ab  # noqa: E402
from valence_instrument import cues as cu  # noqa: E402
from valence_instrument import cv  # noqa: E402
from valence_instrument.constants import BLENDS, FR_STARTS  # noqa: E402
from valence_instrument.features import block_columns, design_matrix  # noqa: E402
from valence_instrument.model import fit_head, load_head, predict, save_head  # noqa: E402

RNG = np.random.default_rng(0)
FAKE_HEAD = {"W": RNG.normal(size=(8, 1280)).astype(np.float32) * .01,
             "b": np.zeros(8, np.float32)}


def fake_clip(tmp: Path, cid: str, n: int) -> None:
    conf = np.where(RNG.random(n) < .9, RNG.uniform(.5, 1, n), 0.0)
    val = np.where(conf >= .5, RNG.uniform(-1, 1, n), 0.0)
    pd.DataFrame({"frame": np.arange(n), "blazeface_confidence": conf,
                  "emotiefflib_valence": val}).to_csv(tmp / f"{cid}.csv", index=False)
    emb = RNG.normal(size=(n, 1280)).astype(np.float32)
    emb[conf < .5] = 0
    np.save(tmp / f"{cid}_emotiefflib.npy", emb)
    pose = RNG.uniform(0, 1, (n, 33, 5)).astype(np.float32)
    head = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    head[:, :3, :3] += RNG.normal(0, .05, (n, 3, 3)).astype(np.float32)
    np.savez_compressed(tmp / f"{cid}.npz", pose=pose,
                        blend=RNG.uniform(0, 1, (n, 52)).astype(np.float32),
                        head_mat=head, detected=RNG.random((n, 2)) < .95,
                        blend_names=np.asarray(BLENDS, dtype=object), fps=30.0)


def table(tmp: Path, ids) -> pd.DataFrame:
    rows = []
    for c in ids:
        row = {"clip_id": c}
        row.update(ab.affect_features(tmp / f"{c}.csv", tmp / f"{c}_emotiefflib.npy", FAKE_HEAD))
        z = cu.load_mediapipe(tmp / f"{c}.npz")
        row.update(cu.full_clip_cues(z))
        row.update(cu.first_reaction_cues(z, FR_STARTS))
        rows.append(row)
    return pd.DataFrame(rows)


def test_pipeline():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ids = [f"c{i:03d}" for i in range(120)]
        for i, c in enumerate(ids):
            fake_clip(tmp, c, n=int(RNG.integers(40, 300)))
        df = table(tmp, ids)
        cols = block_columns()
        assert [len(cols[b]) for b in ("P", "W", "Cf", "Cr", "Cb")] == [21, 19, 34, 34, 9]
        X, names = design_matrix(df, "effnat", start=1.0)
        assert X.shape == (120, 117) and len(names) == 117 and np.isfinite(X).all()
        y = RNG.integers(1, 6, 120).astype(float)
        g = np.repeat(np.arange(30), 4).astype(str)
        head = fit_head(X, y, names, {"stream": "effnat", "blocks": list(cols),
                                      "fr_start": 1.0}, g)
        p = tmp / "head.json"
        save_head(head, p)
        h2 = load_head(p)
        X2, _ = design_matrix(df, "effnat", start=1.0, feature_names=h2["feature_names"])
        assert np.allclose(predict(h2, X2), predict(head, X))
        oof = cv.grouped_oof(X, y, g)
        assert np.isfinite(oof).all()
        # nested selection on a reduced grid (speed)
        oof_n, picks = cv.nested_selection(df, y, g, streams=("effnat",),
                                           starts=(1.0,), workers=2)
        assert np.isfinite(oof_n).all() and len(picks) == 10
        n_c, _ = cv.candidates(("effnat", "shipped"), FR_STARTS)
        assert len(n_c) == 2 * 96 + 31   # paper: 5 * 96 + 31 = 511
        print("ok:", cv.metrics(y, oof), picks[["stream", "blocks", "start"]].iloc[0].to_dict())


if __name__ == "__main__":
    test_pipeline()
