"""Command line entry point:  python -m valence_instrument <command> ...

Inference on new clips (each clip must START at the event onset, e.g. the
moment feedback appears; all windows are timed from clip start):
    reencode   videos -> <work>/clips/<clip_id>/video.mp4 at constant 30 fps
    face       BlazeFace + EmotiEffLib on a DISCOVER server
    export     DISCOVER outputs -> <work>/frames/<clip_id>.csv + _emotiefflib.npy
    mediapipe  MediaPipe pose + blendshapes -> <work>/mediapipe/<clip_id>.npz
    features   -> <work>/features.csv (wide table, all candidate features)
    predict    features.csv + valence_head.json -> per-clip valence (1-5 scale)
    infer      all of the above in sequence

Training on your own rated clips (ratings CSV: clip_id, group, rating;
group = the person, so that folds are person-disjoint):
    evaluate   person-grouped 10-fold CV of one fixed design
    nested-cv  nested design selection (streams x blocks x window start)
    train      fit a head on all rated clips -> valence_head.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import BLOCKS, DEPLOYED_FR_START, DEPLOYED_STREAM, FR_STARTS, N_BOOT, STREAMS


def _clip_ids(work: Path) -> list[str]:
    return sorted(p.name for p in (work / "clips").iterdir() if p.is_dir())


def _ratings(df: pd.DataFrame, path: Path):
    r = pd.read_csv(path, dtype={"clip_id": str, "group": str})
    m = df.merge(r[["clip_id", "group", "rating"]], on="clip_id", validate="one_to_one")
    m = m[m["rating"].notna()].reset_index(drop=True)
    return m, m["rating"].to_numpy(float), m["group"].to_numpy()


def _design_args(ap):
    ap.add_argument("--stream", default=DEPLOYED_STREAM, choices=STREAMS)
    ap.add_argument("--blocks", default="+".join(BLOCKS))
    ap.add_argument("--fr-start", type=float, default=DEPLOYED_FR_START)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="valence_instrument", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("reencode")
    a.add_argument("--videos", type=Path, required=True)
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--workers", type=int, default=4)

    a = sub.add_parser("face")
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--url", default="http://127.0.0.1:8080/process",
                   help="DISCOVER server /process endpoint")
    a.add_argument("--insecure", action="store_true", help="skip TLS verification")
    a.add_argument("--chunk-size", type=int, default=150)

    a = sub.add_parser("export")
    a.add_argument("--work", type=Path, required=True)

    a = sub.add_parser("mediapipe")
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--model-dir", type=Path, default=Path("models/mediapipe"))
    a.add_argument("--workers", type=int, default=4)

    a = sub.add_parser("features")
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--emotiefflib-pt", type=Path, default=None)

    a = sub.add_parser("predict")
    a.add_argument("--features", type=Path, required=True)
    a.add_argument("--head", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)

    a = sub.add_parser("infer")
    a.add_argument("--videos", type=Path, required=True)
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--head", type=Path, required=True)
    a.add_argument("--url", default="http://127.0.0.1:8080/process")
    a.add_argument("--insecure", action="store_true")
    a.add_argument("--model-dir", type=Path, default=Path("models/mediapipe"))
    a.add_argument("--workers", type=int, default=4)

    for name in ("evaluate", "nested-cv", "train"):
        a = sub.add_parser(name)
        a.add_argument("--features", type=Path, required=True)
        a.add_argument("--ratings", type=Path, required=True)
        a.add_argument("--out", type=Path, required=True)
        if name != "nested-cv":
            _design_args(a)
        else:
            a.add_argument("--streams", default=",".join(STREAMS))
            a.add_argument("--workers", type=int, default=4)
        if name != "train":
            a.add_argument("--boot", type=int, default=N_BOOT)

    args = ap.parse_args(argv)
    run(args)


def run(args) -> None:
    cmd = args.cmd
    if cmd in ("reencode", "infer"):
        from .reencode import reencode_dir
        reencode_dir(args.videos, args.work / "clips", args.workers)
    if cmd in ("face", "infer"):
        from .discover_jobs import submit
        submit(_clip_ids(args.work), args.work / "clips", args.work / "discover",
               args.url, verify_tls=not args.insecure,
               chunk_size=getattr(args, "chunk_size", 150))
    if cmd in ("export", "infer"):
        from .discover_export import export_clip
        out = args.work / "frames"
        out.mkdir(parents=True, exist_ok=True)
        for c in _clip_ids(args.work):
            err = export_clip(c, args.work / "discover", out)
            if err:
                print(f"{c}: {err}")
    if cmd in ("mediapipe", "infer"):
        from .mediapipe_extract import extract_all
        errs = extract_all(_clip_ids(args.work), args.work / "clips", args.work / "mediapipe",
                           args.model_dir, args.workers)
        for e in errs:
            print(e)
    if cmd in ("features", "infer"):
        from .features import build_table
        df = build_table(_clip_ids(args.work), args.work / "frames", args.work / "mediapipe",
                         FR_STARTS, STREAMS, args.workers,
                         getattr(args, "emotiefflib_pt", None))
        df.to_csv(args.work / "features.csv", index=False)
    if cmd in ("predict", "infer"):
        from .features import design_matrix
        from .model import load_head, predict
        feats = args.features if cmd == "predict" else args.work / "features.csv"
        out = args.out if cmd == "predict" else args.work / "valence.csv"
        df = pd.read_csv(feats, dtype={"clip_id": str})
        head = load_head(args.head)
        d = head["design"]
        X, _ = design_matrix(df, d["stream"], start=d["fr_start"],
                             feature_names=head["feature_names"])
        pd.DataFrame({"clip_id": df["clip_id"], "valence": predict(head, X)}) \
            .to_csv(out, index=False)
        print(f"-> {out}")
    if cmd in ("evaluate", "train"):
        from .features import design_matrix
        df, y, g = _ratings(pd.read_csv(args.features, dtype={"clip_id": str}), args.ratings)
        blocks = tuple(args.blocks.split("+"))
        X, names = design_matrix(df, args.stream, blocks, args.fr_start)
        design = {"stream": args.stream, "blocks": list(blocks), "fr_start": args.fr_start}
        if cmd == "train":
            from .model import fit_head, save_head
            save_head(fit_head(X, y, names, design, g), args.out)
            print(f"-> {args.out}  ({X.shape[1]} features, {len(y)} clips)")
        else:
            from .cv import child_bootstrap, grouped_oof, metrics
            oof = grouped_oof(X, y, g)
            res = {"design": design, "n_clips": len(y),
                   "n_groups": int(len(np.unique(g))), **metrics(y, oof)}
            if args.boot:
                B = child_bootstrap(y, {"oof": oof}, g, args.boot)["oof"]
                res["r_ci"] = np.percentile(B[:, 0], [2.5, 97.5]).tolist()
                res["ccc_ci"] = np.percentile(B[:, 1], [2.5, 97.5]).tolist()
            args.out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"clip_id": df["clip_id"], "group": g, "y": y, "oof": oof}) \
                .to_csv(args.out / "oof.csv", index=False)
            (args.out / "summary.json").write_text(json.dumps(res, indent=1))
            print(json.dumps(res, indent=1))
    if cmd == "nested-cv":
        from .cv import child_bootstrap, metrics, nested_selection
        df, y, g = _ratings(pd.read_csv(args.features, dtype={"clip_id": str}), args.ratings)
        oof, picks = nested_selection(df, y, g, tuple(args.streams.split(",")),
                                      FR_STARTS, args.workers)
        res = {"n_clips": len(y), "n_groups": int(len(np.unique(g))), **metrics(y, oof)}
        if args.boot:
            B = child_bootstrap(y, {"oof": oof}, g, args.boot)["oof"]
            res["r_ci"] = np.percentile(B[:, 0], [2.5, 97.5]).tolist()
            res["ccc_ci"] = np.percentile(B[:, 1], [2.5, 97.5]).tolist()
        args.out.mkdir(parents=True, exist_ok=True)
        picks.to_csv(args.out / "picks.csv", index=False)
        pd.DataFrame({"clip_id": df["clip_id"], "group": g, "y": y, "oof": oof}) \
            .to_csv(args.out / "oof.csv", index=False)
        (args.out / "summary.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        print(picks.to_string(index=False))


if __name__ == "__main__":
    main()
