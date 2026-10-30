"""Face stage: BlazeFace + EmotiEffLib as DISCOVER file-mode jobs.

The face detector and the EmotiEffLib forward pass are NOT re-implemented
here. They run as the public DISCOVER modules `blazeface` (>= 1.1.1,
letterboxed detector input) and `emotiefflib` (1.0.x) on a DISCOVER server,
exactly as for the paper. A re-implementation would change the boxes and thus
the crops and every downstream number.

Requires a running DISCOVER server with both modules installed; see README.
Per clip the server writes into <out_root>/<clip_id>/:
    face_bounding_box.stream, facial_landmarks.stream,
    emotiefflib_emotion.annotation, emotiefflib_valence.annotation,
    emotiefflib_arousal.annotation, emotiefflib_embedding.stream

Adapted from the paper's job-submission script: paths and URL configurable;
EmoNet and pose stages removed (not part of the instrument).
UNVERIFIED in this release layout: the paper ran with a different directory
layout (same templates otherwise). Test on one clip first.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from .constants import BLAZEFACE_OPTIONS, EMOTIEFFLIB_OPTIONS
from .reencode import VIDEO_NAME

# AffectNet-8 + rest; the module writes discrete labels with these ids.
CLASSES = {"0": "neutral", "1": "happiness", "2": "sadness", "3": "surprise",
           "4": "fear", "5": "disgust", "6": "anger", "7": "contempt", "8": "rest"}


def _esc(path: Path) -> str:
    return str(Path(path).resolve()).replace("{", "{{").replace("}", "}}")


def _video(clips_root: Path) -> dict:
    return {"id": "video", "type": "input", "src": "file:stream:video",
            "uri_template": f"{_esc(clips_root)}/{{session}}/{VIDEO_NAME}", "active": True}


def _out(out_root: Path, fname: str) -> str:
    return f"{_esc(out_root)}/{{session}}/{fname}"


def job_blazeface(clips_root: Path, out_root: Path) -> list[dict]:
    return [_video(clips_root),
            {"id": "bounding_box", "type": "output", "src": "file:stream:ssistream",
             "uri_template": _out(out_root, "face_bounding_box.stream"), "active": True},
            {"id": "landmarks", "type": "output", "src": "file:stream:ssistream",
             "uri_template": _out(out_root, "facial_landmarks.stream"), "active": True}]


def job_emotiefflib(clips_root: Path, out_root: Path) -> list[dict]:
    cont = {"src": "file:annotation:continuous", "sample_rate": 30,
            "min_val": -1, "max_val": 1, "active": True, "type": "output"}
    return [_video(clips_root),
            {"id": "bounding_box", "type": "input", "src": "file:stream:ssistream",
             "uri_template": _out(out_root, "face_bounding_box.stream"), "active": True},
            {"id": "expression", "type": "output", "src": "file:annotation:discrete",
             "uri_template": _out(out_root, "emotiefflib_emotion.annotation"),
             "classes": CLASSES, "active": True},
            {"id": "valence", **cont,
             "uri_template": _out(out_root, "emotiefflib_valence.annotation")},
            {"id": "arousal", **cont,
             "uri_template": _out(out_root, "emotiefflib_arousal.annotation")},
            {"id": "embedding", "type": "output", "src": "file:stream:ssistream",
             "uri_template": _out(out_root, "emotiefflib_embedding.stream"), "active": True}]


def build_jobs(clips_root: Path, out_root: Path, tag: str = "") -> list[dict]:
    sfx = f"-{tag}" if tag else ""
    common = {"leftContext": "0", "rightContext": "0", "frameSize": "0"}
    return [
        {"jobID": f"valence-blazeface{sfx}", "trainerFilePath": "blazeface\\blazeface.trainer",
         **common, "data": json.dumps(job_blazeface(clips_root, out_root)),
         "options": json.dumps(BLAZEFACE_OPTIONS)},
        {"jobID": f"valence-emotiefflib{sfx}",
         "trainerFilePath": "emotiefflib\\emotiefflib.trainer",
         **common, "data": json.dumps(job_emotiefflib(clips_root, out_root)),
         "options": json.dumps(EMOTIEFFLIB_OPTIONS)},
    ]


def status_url(process_url: str) -> str:
    p = urlparse(process_url)
    return urlunparse(p._replace(path="/job_status", query="", fragment=""))


def post_job(url: str, payload: dict, verify_tls: bool) -> None:
    import requests
    r = requests.post(url, headers={"Content-type": "application/x-www-form-urlencoded"},
                      data=payload, verify=verify_tls)
    if r.status_code != 200:
        raise RuntimeError(f"Job {payload['jobID']} failed: {r.status_code} {r.text}")
    print(f"Submitted {payload['jobID']}")


def wait_for_job(url: str, job_id: str, poll: int, timeout: int, verify_tls: bool) -> None:
    import requests
    t0 = time.monotonic()
    while True:
        r = requests.post(url, data={"jobID": job_id}, verify=verify_tls)
        if r.status_code != 200:
            raise RuntimeError(f"Status request for {job_id} failed: {r.status_code} {r.text}")
        status = str(r.json()["status"])
        if status == "2":
            return
        if status == "3":   # finished with per-clip errors; other clips succeeded
            print(f"WARN: {job_id} completed with per-clip errors")
            return
        if status == "4":
            raise RuntimeError(f"Job {job_id} failed (status 4)")
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"Timed out waiting for {job_id} (last status {status})")
        time.sleep(poll)


def submit(clip_ids: list[str], clips_root: Path, out_root: Path, url: str,
           dataset: str = "valence", chunk_size: int = 150, poll: int = 30,
           timeout_h: float = 72.0, verify_tls: bool = True) -> None:
    """Run BlazeFace then EmotiEffLib on every clip, in chunks (the server leaks
    file handles on very large jobs)."""
    for c in clip_ids:
        (Path(out_root) / c).mkdir(parents=True, exist_ok=True)
    chunks = [clip_ids[i:i + chunk_size] for i in range(0, len(clip_ids), chunk_size)]
    st = status_url(url)
    for i, chunk in enumerate(chunks):
        tag = f"chunk{i + 1:03d}of{len(chunks):03d}" if len(chunks) > 1 else ""
        for job in build_jobs(clips_root, out_root, tag):
            payload = {**job, "sessions": json.dumps(chunk), "dataset": dataset}
            post_job(url, payload, verify_tls)
            wait_for_job(st, payload["jobID"], poll, int(timeout_h * 3600), verify_tls)
