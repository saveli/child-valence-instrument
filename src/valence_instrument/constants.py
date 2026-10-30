"""Fixed constants of the deployed instrument.

Every value here is the value that produced the paper's numbers. Change them
only for your own experiments, and then do not call the result "the
instrument".
"""
import numpy as np

# --- video -------------------------------------------------------------------
FPS = 30                      # clips are re-encoded to constant 30 fps first

# --- face stage (DISCOVER modules, see discover_jobs.py) ---------------------
CONF = 0.5                    # BlazeFace confidence for a "confident" frame
BLAZEFACE_OPTIONS = {         # blazeface module >= 1.1.1 (letterboxed input)
    "model": "back",
    "min_suppression_thresh": 0.3,
    "min_score_thresh": 0.5,
    "batch_size": 250,
    "repeat_last": True,
}
EMOTIEFFLIB_OPTIONS = {       # emotiefflib module 1.0.x
    "model_name": "enet_b0_8_va_mtl",
    "batch_size": 64,
    "face_detection_min_conf": 0.5,
    "margin": 0.1,
}

# EmotiEffLib package class order (alphabetical) for 8-class models. The P
# block columns childp0..childp7 follow THIS order (not AffectNet order).
EMOTIEFFLIB_CLASSES = ("Anger", "Contempt", "Disgust", "Fear",
                       "Happiness", "Neutral", "Sadness", "Surprise")
PROXY_IDX_SHIPPED = (4, 6, 0)  # happy, sad, anger -> P(h) - P(s) - P(a)

# --- window bank (W block, 19 features) --------------------------------------
WIN_STARTS = (0.0, 0.5, 1.0, 1.5, 2.0)
WIN_DURS = (1.0, 2.0)
EXT_SPANS = (2.0, 3.0, 4.0)

# --- first-reaction facial cues (Cr block) -----------------------------------
FR_STARTS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)   # candidate starts (nested CV)
FR_DUR = 3.0
DEPLOYED_FR_START = 1.0                           # picked by nested CV

# --- ridge head ----------------------------------------------------------------
ALPHAS = np.logspace(-2, 5, 15)
BLOCKS = ("P", "W", "Cf", "Cr", "Cb")             # hstack order of the 117 features
DEPLOYED_STREAM = "effnat"                        # shipped softmax + native valence
STREAMS = ("effnat", "shipped")                   # streams this release can build

# --- evaluation ----------------------------------------------------------------
N_OUTER = 10
N_INNER = 5
N_BOOT = 2000
SEED_BOOT = 20261026

# MediaPipe FaceLandmarker blendshape order (52, incl. "_neutral").
BLENDS = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp',
          'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
          'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
          'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
          'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
          'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward',
          'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft',
          'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
          'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
          'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
          'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
          'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
          'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft',
          'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

# MediaPipe Tasks model files: downloaded at run time, never vendored.
# sha256 = the files the paper's features were extracted with.
MEDIAPIPE_MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    # NOTE: only the float16/latest URL matched this hash on 2026-10-30; the
    # float16/1 URL serves a different file. "latest" can move -> hash check.
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"),
}
EMOTIEFFLIB_PT_SHA256 = "b7b9522ec6062d94919ef5cf620b561bc0320c8d03c88a807b95accb0da3d600"
