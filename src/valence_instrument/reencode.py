"""Re-encode clips to constant 30 fps H.264.

Why: the source WebM recordings had irregular frame timing (dropped frames,
inconsistent or missing frame-rate metadata). All window features are timed in
frames at 30 fps, so every clip is resampled to a constant 30 fps first
(ffmpeg duplicates/drops frames). Same ffmpeg call as used for the paper.

Layout produced (one folder per clip, the layout the DISCOVER jobs expect):
    <out_root>/<clip_id>/video.mp4
<clip_id> = input file stem.
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

VIDEO_NAME = "video.mp4"
VIDEO_EXTS = (".webm", ".mp4", ".mkv", ".mov", ".avi")


def convert_video(args: tuple[Path, Path, bool]) -> Path:
    path, out_root, overwrite = args
    out = out_root / path.stem / VIDEO_NAME
    out.parent.mkdir(exist_ok=True, parents=True)
    if out.exists() and not overwrite:
        return out
    cmd = ["ffmpeg", "-y" if overwrite else "-n", "-hide_banner", "-loglevel", "error",
           "-i", str(path), "-vf", "fps=30", "-c:v", "libx264", "-profile:v", "high",
           str(out)]
    subprocess.run(cmd, check=True)
    return out


def reencode_dir(in_dir: Path, out_root: Path, workers: int = 4,
                 overwrite: bool = False) -> list[str]:
    videos = sorted(p for p in Path(in_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    jobs = [(v, Path(out_root), overwrite) for v in videos]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(convert_video, jobs))
    return [v.stem for v in videos]
