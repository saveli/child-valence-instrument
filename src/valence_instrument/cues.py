"""The 26 codebook cue definitions on MediaPipe outputs (77 features).

17 facial cues from the blendshapes, each summarized by mean and 95th
percentile ("_max") over
    Cf  the full clip                       (cb_<cue>_mean / cb_<cue>_max, 34)
    Cr  the first-reaction window, 3 s long (fr_<cue>_mean / fr_<cue>_max, 34)
        starting `start` s after clip start (deployed: 1.0 s)
and 9 head / gaze / upper-body cues once per clip (Cb, 9):
    cb_gaze_wander, cb_head_tilt, cb_head_back, cb_nod_energy,
    cb_shoulder_trend, cb_slump_trend, cb_fidget, cb_arm_raised, cb_arm_motion

The cues operationalize the visible indicators of the raters' coding scheme
(see the paper's supplement for the indicator -> cue table). Formulas are
copied unchanged from the code that produced the paper's numbers; only file
handling changed.

Statistics use frames with a detected face (MediaPipe). A window with fewer
than 4 such frames falls back to all detected frames. Clips with fewer than 8
detected faces get no cue features (NaN; set to 0 before the ridge head).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import BLENDS, FPS, FR_DUR

BI = {n: i for i, n in enumerate(BLENDS)}

FACIAL_CUES = ("smile", "frown", "grin", "brow_down", "brow_up", "eye_squint",
               "eye_wide", "eye_closed", "mouth_press", "pucker", "jaw_forward",
               "shrug_lower", "nose_sneer", "cheek_puff", "mouth_stretch",
               "mouth_asym", "jaw_open")
BODY_CUES = ("gaze_wander", "head_tilt", "head_back", "nod_energy", "shoulder_trend",
             "slump_trend", "fidget", "arm_raised", "arm_motion")


def _pair(b, name):  # mean of the Left/Right channels
    return (b[:, BI[f"{name}Left"]] + b[:, BI[f"{name}Right"]]) / 2


def cue_dict(b: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame facial cue signals from the (n, 52) blendshape array."""
    return {
        "smile": _pair(b, "mouthSmile"),
        "frown": _pair(b, "mouthFrown"),
        "grin": _pair(b, "mouthSmile") * b[:, BI["jawOpen"]],
        "brow_down": _pair(b, "browDown"),
        "brow_up": (b[:, BI["browInnerUp"]] + _pair(b, "browOuterUp")) / 2,
        "eye_squint": _pair(b, "eyeSquint"),
        "eye_wide": _pair(b, "eyeWide"),
        "eye_closed": _pair(b, "eyeBlink"),
        "mouth_press": _pair(b, "mouthPress"),
        "pucker": b[:, BI["mouthPucker"]],
        "jaw_forward": b[:, BI["jawForward"]],
        "shrug_lower": b[:, BI["mouthShrugLower"]],
        "nose_sneer": _pair(b, "noseSneer"),
        "cheek_puff": b[:, BI["cheekPuff"]],
        "mouth_stretch": _pair(b, "mouthStretch"),
        "mouth_asym": np.abs(b[:, BI["mouthLeft"]] - b[:, BI["mouthRight"]]),
        "jaw_open": b[:, BI["jawOpen"]],
    }


def load_mediapipe(path: Path):
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z["blend_names"]] if "blend_names" in z.files else []
    if names and names != list(BLENDS):
        raise ValueError(f"{path}: unexpected blendshape order from MediaPipe")
    return z


def full_clip_cues(z) -> dict:
    """Cf (34) and Cb (9) for one clip."""
    row: dict = {}
    blend, pose, head = z["blend"], z["pose"], z["head_mat"]
    det_all = z["detected"].astype(bool)          # (n, 2) = [pose, face]
    pose_det, det = det_all[:, 0], det_all[:, 1]
    if det.sum() < 8 or blend.shape[1] != 52:
        return row
    n = len(det)
    b = blend

    for name, x in cue_dict(b).items():
        v = x[det]   # mask = det; det.sum() >= 8 here
        row[f"cb_{name}_mean"] = float(np.mean(v))
        row[f"cb_{name}_max"] = float(np.percentile(v, 95))

    # gaze wandering: eyeLook* variability
    look = b[:, [BI[c] for c in BLENDS if c.startswith("eyeLook")]]
    row["cb_gaze_wander"] = float(look[det].std(axis=0).mean())

    # head pose from the facial transformation matrix
    R = head[det][:, :3, :3]
    pitch = np.arctan2(-R[:, 2, 1], R[:, 2, 2])
    roll = np.arctan2(-R[:, 0, 1], R[:, 1, 1])
    row["cb_head_tilt"] = float(np.abs(roll).mean())
    row["cb_head_back"] = float(np.percentile(pitch, 95))
    row["cb_nod_energy"] = float(np.abs(np.diff(pitch)).mean() * FPS)

    # posture from pose landmarks (normalized xy, y grows downward)
    vis = pose[:, :, 3]
    sh_ok = pose_det & (vis[:, 11] > .5) & (vis[:, 12] > .5)
    if sh_ok.sum() >= 8:
        sh_y = pose[:, [11, 12], 1].mean(axis=1)
        t = np.arange(n)[sh_ok]
        row["cb_shoulder_trend"] = float(np.polyfit(t / FPS, sh_y[sh_ok], 1)[0])
        nose_sh = sh_y - pose[:, 0, 1]
        row["cb_slump_trend"] = float(-np.polyfit(t / FPS, nose_sh[sh_ok], 1)[0])
        row["cb_fidget"] = float(np.abs(np.diff(pose[sh_ok][:, [0, 11, 12], :2],
                                                axis=0)).mean() * FPS)
    wr_ok = pose_det & ((vis[:, 15] > .5) | (vis[:, 16] > .5))
    if wr_ok.sum() >= 8:
        wy = np.where(vis[:, 15] > vis[:, 16], pose[:, 15, 1], pose[:, 16, 1])
        wxy = np.where((vis[:, 15] > vis[:, 16])[:, None], pose[:, 15, :2], pose[:, 16, :2])
        shy = pose[:, [11, 12], 1].mean(axis=1)
        row["cb_arm_raised"] = float((wy[wr_ok] < shy[wr_ok]).mean())
        row["cb_arm_motion"] = float(np.percentile(
            np.abs(np.diff(wxy[wr_ok], axis=0)).sum(axis=1), 90) * FPS)
    return row


def first_reaction_cues(z, starts) -> dict:
    """Cr (34) for each window start; keys 'w<start>|fr_<cue>_<stat>'."""
    row: dict = {}
    blend, det_all = z["blend"], z["detected"].astype(bool)
    det = det_all[:, 1] if det_all.ndim == 2 else det_all
    if det.sum() < 8 or blend.shape[1] != 52:
        return row
    n = len(det)
    cues = cue_dict(blend)
    for s in starts:
        lo, hi = min(int(s * FPS), n - 1), min(int((s + FR_DUR) * FPS), n)
        fr = np.zeros(n, bool)
        fr[lo:hi] = True
        m = det & fr
        for name, x in cues.items():
            v = x[m] if m.sum() >= 4 else x[det]
            row[f"w{s}|fr_{name}_mean"] = float(np.mean(v))
            row[f"w{s}|fr_{name}_max"] = float(np.percentile(v, 95))
    return row
