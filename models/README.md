# models/

`valence_head.json`: the deployed ridge head, refit on all 5,243 rated clips
(518 children) with the design selected by nested child-grouped CV.

Content: 117 feature names, 117 standardization means and scales, 117 ridge
coefficients, intercept, penalty, training counts, software versions. Plain
JSON (no pickle), about 10 kB. It contains no per-clip or per-person values.

To fit a head on your own rated clips instead: `python -m valence_instrument train ...`.

Not in this folder, and never redistributed here: the MediaPipe `.task`
files (downloaded at run time into `models/mediapipe/`, checked by sha256),
the EmotiEffLib weights (downloaded by the `emotiefflib` package), the
BlazeFace weights (downloaded by the DISCOVER `blazeface` module).
