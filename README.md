# Clip-level valence from frozen face and body features

Code for the paper *Validating Child Facial Valence in Classroom Video: Temporal
Pooling and Interpretable Features* (FG 2027, under review).

The fitted head is included (`models/valence_head.json`); see
[Model availability](#model-availability).

The instrument estimates a 1–5 valence rating for short video clips of a
reaction (for example, from the moment feedback appears on screen). It uses
only frozen, open-weight models that run locally, plus a ridge head with
117 inputs:

| block | features | source |
|---|---|---|
| P (21) | 8 class probabilities (mean, SD) + valence (mean, SD, min, max, last−first) over frames with a confident face | EmotiEffLib `enet_b0_8_va_mtl` on BlazeFace crops |
| W (19) | valence window means 0–2 s after onset (1 s and 2 s windows); min, max and signed extreme of the loess-smoothed valence over the first 2, 3 and 4 s | same stream, all frames |
| Cf (34) | 17 facial cues from MediaPipe blendshapes, mean and 95th percentile over the clip | MediaPipe FaceLandmarker |
| Cr (34) | the same 17 cues over a 3 s first-reaction window starting 1.0 s after onset | MediaPipe FaceLandmarker |
| Cb (9) | head, gaze and upper-body cues (head tilt, head back, nod energy, gaze wander, shoulder and slump trend, fidget, arm raised, arm motion) | MediaPipe Face/PoseLandmarker |

The cues operationalize the visible indicators of a human valence coding
scheme. The design (stream, blocks, window start) was chosen by nested
person-grouped cross-validation. In the paper, person-disjoint 10-fold CV
gives r = .53 and CCC = .45 against the human ratings.

## Intended use, and what not to do

This is research software for **aggregate, study-level** analyses of
recorded reactions, under ethical approval and with consent.
Its validation does **not** support decisions about individuals. Do not
use it to grade, select, diagnose or monitor anyone. Do not deploy it for
emotion recognition in schools or workplaces. In the EU this may be
prohibited (AI Act, Art. 5(1)(f)). Agreement with human ratings was
established on one population and one recording setup. Before using it on
your data, validate it against your own ratings (see *Train on your own
data*).

## Pipeline

```
video ──ffmpeg fps=30──▶ video.mp4 ─┬─▶ DISCOVER: BlazeFace ─▶ EmotiEffLib ─▶ P, W
                                    └─▶ MediaPipe Tasks (pose, blendshapes, head pose) ─▶ Cf, Cr, Cb
                                                                     117 features ─▶ ridge head ─▶ valence (1–5)
```

* **Re-encoding.** Every clip is resampled to a constant 30 fps (H.264).
  All windows are counted in frames at 30 fps, and recordings with irregular
  frame timing would otherwise shift them.
* **Face stage.** BlazeFace and EmotiEffLib run as modules of the DISCOVER
  orchestration server [DISCOVER citation], unchanged from the paper. They are
  not re-implemented here, because small changes in detection or cropping
  change the features. You need a DISCOVER server with the `blazeface`
  (≥ 1.1.1) and `emotiefflib` (1.0.x) modules (open source; repository
  links are added after review).
* **Clip start = event onset.** Window features are timed from the first
  frame. Cut your clips so they start at the event onset.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
# ffmpeg must be on PATH; a DISCOVER server must be reachable for the face stage
```

## Inference on new clips

```bash
python -m valence_instrument infer \
    --videos path/to/clips/ --work work/ \
    --head models/valence_head.json \
    --url http://127.0.0.1:8080/process
# -> work/valence.csv  (clip_id, valence on the 1–5 rating scale; not clipped)
```

The steps can also be run one by one: `reencode`, `face`, `export`,
`mediapipe`, `features`, `predict` (`--help` on each).

## Train on your own data

Prepare `ratings.csv` with columns `clip_id, group, rating`. `group`
identifies the person, so that every fold keeps a person's clips together.
Then:

```bash
python -m valence_instrument reencode --videos clips/ --work work/
python -m valence_instrument face     --work work/ --url ...
python -m valence_instrument export   --work work/
python -m valence_instrument mediapipe --work work/
python -m valence_instrument features --work work/

# person-grouped 10-fold CV of the paper's design
python -m valence_instrument evaluate  --features work/features.csv --ratings ratings.csv --out eval/
# nested design selection (stream x blocks x window start), as in the paper
python -m valence_instrument nested-cv --features work/features.csv --ratings ratings.csv --out nested/
# final head on all rated clips
python -m valence_instrument train     --features work/features.csv --ratings ratings.csv \
    --out models/my_head.json [--stream effnat --blocks P+W+Cf+Cr+Cb --fr-start 1.0]
```

Of the paper's five candidate per-frame streams, this release includes the
two that need no extra weights: `effnat` (deployed: EmotiEffLib softmax with
its native valence) and `shipped` (the same softmax with the fixed proxy
P(happy) − P(sad) − P(anger)).

## Model availability

`models/valence_head.json`: about 350 numbers (feature means and scales,
coefficients, intercept), no per-clip data, under `LICENSE-MODEL` (research
only, no individual decisions, no deployment in schools or workplaces).

## Third-party models (downloaded at run time, not redistributed)

| model | obtained from | licence |
|---|---|---|
| MediaPipe Face/Pose Landmarker `.task` | Google MediaPipe model bucket (sha256 checked) | Apache-2.0 (model cards) |
| EmotiEffLib `enet_b0_8_va_mtl` | `emotiefflib` package download | code Apache-2.0; weights trained on AffectNet (research-only terms) |
| BlazeFace (back model) | DISCOVER `blazeface` module, via BlazeFace-PyTorch | see module README |

## Data

No videos, extracted features, ratings or per-person outputs are part of
this repository. The study recordings are of minors and cannot be shared.

## Citation

```
<BibTeX — added after review>
```

## Licence

Code: MIT (`LICENSE`). Fitted parameters: `LICENSE-MODEL`.
