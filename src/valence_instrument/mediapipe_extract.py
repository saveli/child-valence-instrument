"""MediaPipe Tasks on the full frame: pose, blendshapes, head pose.

One sequential pass per clip (CPU, VIDEO running mode) through
    PoseLandmarker (full): 33 landmarks x (x, y, z, visibility, presence)
    FaceLandmarker:        52 blendshapes, 4x4 facial transformation matrix
Output: <out_dir>/<clip_id>.npz with
    pose (n,33,5), blend (n,52), head_mat (n,4,4), detected (n,2) = [pose, face],
    blend_names, fps   [+ face (n,478,3) only with store_face_mesh=True]

Same extraction as for the paper. One change: the 478-point face mesh is NOT
stored by default (no feature uses it, and dense face landmarks are close to
biometric data). Pass store_face_mesh=True to get the paper's exact file layout.

Model files are downloaded at run time from Google's MediaPipe model bucket and
checked against the sha256 of the files used for the paper.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

from .constants import MEDIAPIPE_MODELS
from .reencode import VIDEO_NAME


def ensure_models(model_dir: Path, check_hash: bool = True) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for fname, (url, sha) in MEDIAPIPE_MODELS.items():
        p = model_dir / fname
        if not p.exists():
            print(f"downloading {fname}")
            urllib.request.urlretrieve(url, p)
        if check_hash:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h != sha:
                raise RuntimeError(f"{fname}: sha256 {h} differs from the paper's model "
                                   f"({sha}). Features may not match; pass "
                                   "check_hash=False to proceed anyway.")
    return model_dir


def extract_clip(task: tuple[str, str, str, str, bool]) -> str | None:
    """Worker: one clip through both landmarkers. Returns an error string or None."""
    clip_id, clips_root, out_dir, model_dir, store_face_mesh = task
    out_path = Path(out_dir) / f"{clip_id}.npz"
    if out_path.exists():
        return None

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # VIDEO-mode landmarkers need monotonically increasing timestamps over their
    # lifetime -> fresh instances per clip.
    pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(Path(model_dir) / "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.VIDEO))
    face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(Path(model_dir) / "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True))

    video_path = Path(clips_root) / clip_id / VIDEO_NAME
    if not video_path.exists():
        return f"video missing: {video_path}"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"cv2 cannot open {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose_rows, face_rows, blend_rows, mat_rows, det_rows = [], [], [], [], []
    blend_names = None
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = int(n * 1000 / fps)
        pr = pose_lm.detect_for_video(img, ts)
        fr = face_lm.detect_for_video(img, ts)
        if pr.pose_landmarks:
            pose_rows.append([[l.x, l.y, l.z, l.visibility, l.presence]
                              for l in pr.pose_landmarks[0]])
        else:
            pose_rows.append(np.zeros((33, 5), dtype=np.float32))
        if fr.face_landmarks:
            if store_face_mesh:
                face_rows.append([[l.x, l.y, l.z] for l in fr.face_landmarks[0]])
            blend_rows.append([b.score for b in fr.face_blendshapes[0]])
            mat_rows.append(np.asarray(fr.facial_transformation_matrixes[0]))
            if blend_names is None:
                blend_names = [b.category_name for b in fr.face_blendshapes[0]]
        else:
            if store_face_mesh:
                face_rows.append(np.zeros((478, 3), dtype=np.float32))
            blend_rows.append(np.zeros(52, dtype=np.float32))
            mat_rows.append(np.zeros((4, 4), dtype=np.float32))
        det_rows.append([bool(pr.pose_landmarks), bool(fr.face_landmarks)])
        n += 1
    cap.release()
    pose_lm.close()
    face_lm.close()

    arrays = dict(pose=np.asarray(pose_rows, dtype=np.float32),
                  blend=np.asarray(blend_rows, dtype=np.float32),
                  head_mat=np.asarray(mat_rows, dtype=np.float32),
                  detected=np.asarray(det_rows, dtype=bool),
                  blend_names=np.asarray(blend_names if blend_names else [], dtype=object),
                  fps=fps)
    if store_face_mesh:
        arrays["face"] = np.asarray(face_rows, dtype=np.float32)
    np.savez_compressed(out_path, **arrays)
    return None


def extract_all(clip_ids: list[str], clips_root: Path, out_dir: Path, model_dir: Path,
                workers: int = 4, store_face_mesh: bool = False) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ensure_models(model_dir)
    tasks = [(c, str(clips_root), str(out_dir), str(model_dir), store_face_mesh)
             for c in clip_ids]
    if workers <= 1:
        res = list(map(extract_clip, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(extract_clip, tasks, chunksize=4))
    return [{"clip": t[0], "error": e} for t, e in zip(tasks, res) if e]
