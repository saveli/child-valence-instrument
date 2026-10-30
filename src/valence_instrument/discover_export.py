"""Convert DISCOVER outputs of one clip to plain per-frame files.

Writes into <export_root>/:
    <clip_id>.csv                  frame, time_s, blazeface_* (5), emotiefflib_valence
    <clip_id>_emotiefflib.npy      (n, 1280) EmotiEffLib embedding (zeros where no face)

Adapted from the paper's export script; EmoNet columns removed.
DEVIATION (documented): in the paper the per-frame length was set by the EmoNet
valence stream and the other streams were linearly aligned to it. EmoNet is not
part of the instrument, so here the EmotiEffLib valence stream sets the length.
Both come from the same 30 fps video and the same box stream; we expect equal
lengths, but this was not verified clip by clip.

The embedding files are face-derived data of the people in your videos. Keep
them under the same protection as the videos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BBOX_COLS = ["blazeface_ymin", "blazeface_xmin", "blazeface_ymax", "blazeface_xmax",
             "blazeface_confidence"]


def _load(path: Path):
    from discover_utils.data.handler.file_handler import FileHandler
    return FileHandler().load(path)


def continuous_values(annotation) -> np.ndarray:
    if hasattr(annotation.data, "dtype") and annotation.data.dtype.names:
        return annotation.data["score"].astype(np.float32)
    return np.asarray(annotation.data, dtype=np.float32).flatten()


def align_numeric(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) == target_length:
        return values
    if len(values) == 0:
        return np.full(target_length, np.nan, dtype=np.float32)
    old = np.linspace(0, len(values) - 1, len(values))
    new = np.linspace(0, len(values) - 1, target_length)
    return np.interp(new, old, values).astype(np.float32)


def export_clip(clip_id: str, discover_root: Path, export_root: Path) -> str | None:
    """Returns an error string or None."""
    d = Path(discover_root) / clip_id
    cols: dict = {}
    try:
        cols["emotiefflib_valence"] = continuous_values(_load(d / "emotiefflib_valence.annotation"))
    except Exception as exc:  # noqa: BLE001
        return f"missing emotiefflib valence: {exc}"
    n = len(cols["emotiefflib_valence"])
    try:
        bb = np.asarray(_load(d / "face_bounding_box.stream").data, dtype=np.float32)
        for i, name in enumerate(BBOX_COLS):
            cols[name] = (align_numeric(bb[:, i], n) if bb.ndim == 2 and bb.shape[1] > i
                          else np.full(n, np.nan, np.float32))
    except Exception:  # noqa: BLE001
        for name in BBOX_COLS:
            cols[name] = np.full(n, np.nan, np.float32)
    try:
        emb = np.asarray(_load(d / "emotiefflib_embedding.stream").data, dtype=np.float32)
        if emb.ndim == 2 and emb.shape[1] == 1280:
            np.save(Path(export_root) / f"{clip_id}_emotiefflib.npy", emb)
        else:
            return f"unexpected embedding shape {emb.shape}"
    except Exception as exc:  # noqa: BLE001
        return f"missing embedding: {exc}"
    df = pd.DataFrame(cols)
    df.insert(0, "frame", np.arange(n, dtype=np.int32))
    df.insert(1, "time_s", df["frame"] / 30.0)
    df.to_csv(Path(export_root) / f"{clip_id}.csv", index=False)
    return None
