"""Ridge head: StandardScaler + RidgeCV, and its JSON artifact.

Same estimator as in the paper: features standardized on the training clips,
RidgeCV over 15 log-spaced penalties 1e-2..1e5 (scikit-learn's efficient
leave-one-out selection), target = raw 1-5 rating, prediction NOT clipped.

Artifact (`valence_head.json`), plain JSON, no pickle:
    format, design {stream, blocks, fr_start}, feature_names[117],
    scaler_mean[117], scaler_scale[117], coef[117], intercept, alpha,
    n_train_clips, n_train_groups, target, provenance {...}
prediction = ((x - scaler_mean) / scaler_scale) @ coef + intercept
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .constants import ALPHAS

FORMAT = "valence-ridge-head/1"


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def fit_head(X, y, feature_names, design: dict, groups=None, provenance=None) -> dict:
    sc = StandardScaler().fit(X)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X), y)
    return {"format": FORMAT, "design": design, "feature_names": list(feature_names),
            "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
            "coef": m.coef_.tolist(), "intercept": float(m.intercept_),
            "alpha": float(m.alpha_), "n_train_clips": int(len(y)),
            "n_train_groups": None if groups is None else int(len(np.unique(groups))),
            "target": "clip valence rating, 1 (very negative) .. 5 (very positive)",
            "provenance": provenance or {}}


def save_head(head: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(head, indent=1))


def load_head(path: Path) -> dict:
    head = json.loads(Path(path).read_text())
    if head.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} artifact")
    p = len(head["feature_names"])
    assert all(len(head[k]) == p for k in ("scaler_mean", "scaler_scale", "coef"))
    return head


def predict(head: dict, X: np.ndarray) -> np.ndarray:
    mu, sd = np.asarray(head["scaler_mean"]), np.asarray(head["scaler_scale"])
    return ((X - mu) / sd) @ np.asarray(head["coef"]) + head["intercept"]
