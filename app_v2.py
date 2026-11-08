"""CARE-PD gait score: a held-out UPDRS gait prediction for a CARE-PD walk, shown next to the clinician's score.

The one program of final_version (the experiment itself is final_experiment.ipynb). Run it from this folder:

    python app.py            # the page, on http://127.0.0.1:7860 (GRADIO_PORT changes the port)
    python app.py --check    # self-check: model files, held-out model selection, every walk's score vs the saved run

What it does with a walk from CARE-PD (chosen from the dataset in data/, or uploaded as a .pkl):
  1. shows it: an animated stick figure, the pelvis path, speed, and the 30 per-frame inputs the GRU receives;
  2. scores it with variant A's frozen ordinal GRU (A_LODO/models), five seeds, and the flat GRU for comparison;
  3. compares the score with the clinician's MDS-UPDRS item 3.10 score from the release.

A walk from a scored cohort is scored only by the five LODO models that never saw that cohort, so the comparison is
held out, as in the study (v3_plan). A walk that is not in a scored cohort is scored by the models trained on all four
cohorts and has no clinician score to compare with.

The scoring code below is copied unchanged from final_experiment.ipynb, whose run matches the primary run exactly.
Non-commercial research on CARE-PD. Not a diagnostic device and not validated for clinical use.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("CAREPD_DEVICE", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib import animation  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # noqa: E402

ROOT = Path(__file__).resolve().parent
VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "unversioned"

# =====================================================================================================================
# 1. Scoring code, copied unchanged from final_experiment.ipynb
# =====================================================================================================================

DEVICE = torch.device(os.environ.get("CAREPD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

TARGET_FPS = 30.0
PELVIS = 0
L_HIP, R_HIP = 1, 2
SPINE1, SPINE2, SPINE3 = 3, 6, 9
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
L_FOOT, R_FOOT = 10, 11
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19

SPEED_JOINTS = [
    PELVIS, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT,
    SPINE1, SPINE3, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
]


def resample(arr: np.ndarray, fps: float, target: float = TARGET_FPS) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n < 2 or fps <= 0:
        return arr
    duration = (n - 1) / float(fps)
    m = max(2, int(round(duration * target)) + 1)
    src = np.linspace(0.0, duration, n)
    dst = np.linspace(0.0, duration, m)
    out = np.empty((m, arr.shape[1]), dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(dst, src, arr[:, j])
    return out


WINDOW = 30
PAUSE_SPEED = 0.15

CHANNELS = (
    [f"ang_{idx}" for idx in SPEED_JOINTS]
    + ["pelvis_vx", "pelvis_vy", "pelvis_vz", "pelvis_ax", "pelvis_ay", "pelvis_az", "horiz_speed", "pelvis_y_disp"]
    + ["asym_hip", "asym_knee", "asym_ankle", "asym_shoulder", "arm_swing_l", "arm_swing_r", "pause"]
)


def joint_speed(rotvecs: np.ndarray, fps: float) -> np.ndarray:
    rotation = Rotation.from_rotvec(rotvecs)
    return (rotation[:-1].inv() * rotation[1:]).magnitude() * fps


def arm_swing(rotvecs: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(rotvecs)
    return (rotation.mean().inv() * rotation[:-1]).magnitude()


def plan_channels(pose: np.ndarray, trans: np.ndarray, fps: float) -> np.ndarray:
    pose = resample(np.asarray(pose, dtype=np.float64), fps)
    trans = resample(np.asarray(trans, dtype=np.float64), fps)
    n = min(pose.shape[0], trans.shape[0])
    pose, trans = pose[:n], trans[:n]
    if n < 3 or pose.shape[-1] != 72:
        return np.zeros((1, len(CHANNELS)), dtype=np.float32)
    joints = pose.reshape(n, 24, 3)
    speed = {j: joint_speed(joints[:, j], TARGET_FPS) for j in set(SPEED_JOINTS)}
    velocity = np.diff(trans, axis=0) * TARGET_FPS
    acceleration = np.vstack([np.zeros((1, 3)), np.diff(velocity, axis=0) * TARGET_FPS])
    horizontal = np.linalg.norm(velocity[:, [0, 2]], axis=1)
    vertical = trans[:-1, 1] - np.median(trans[:, 1])
    columns = (
        [speed[j] for j in SPEED_JOINTS]
        + [velocity[:, 0], velocity[:, 1], velocity[:, 2],
           acceleration[:, 0], acceleration[:, 1], acceleration[:, 2], horizontal, vertical]
        + [speed[L_HIP] - speed[R_HIP], speed[L_KNEE] - speed[R_KNEE],
           speed[L_ANKLE] - speed[R_ANKLE], speed[L_SHOULDER] - speed[R_SHOULDER]]
        + [arm_swing(joints[:, L_SHOULDER]), arm_swing(joints[:, R_SHOULDER]),
           (horizontal < PAUSE_SPEED).astype(np.float64)]
    )
    out = np.column_stack(columns).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def window_starts(length: int) -> list[int]:
    if length <= WINDOW:
        return [0]
    starts = list(range(0, length - WINDOW + 1, WINDOW))
    if starts[-1] != length - WINDOW:
        starts.append(length - WINDOW)
    return starts


@dataclass
class Windows:
    x: np.ndarray
    lengths: np.ndarray
    walk: np.ndarray
    n_walks: int

    def counts(self) -> np.ndarray:
        return np.bincount(self.walk, minlength=self.n_walks)


def build_windows(kin_list: list[np.ndarray]) -> Windows:
    n_windows = sum(len(window_starts(len(kin))) for kin in kin_list)
    x = np.zeros((n_windows, WINDOW, len(CHANNELS)), dtype=np.float32)
    lengths = np.zeros(n_windows, dtype=np.int64)
    walk = np.zeros(n_windows, dtype=np.int64)
    i = 0
    for index, kin in enumerate(kin_list):
        for start in window_starts(len(kin)):
            segment = kin[start : start + WINDOW]
            x[i, : len(segment)] = segment
            lengths[i], walk[i] = len(segment), index
            i += 1
    return Windows(x, lengths, walk, len(kin_list))


def _valid(lengths: np.ndarray) -> np.ndarray:
    return np.arange(WINDOW)[None, :] < lengths[:, None]


def to_walks(values: np.ndarray, windows: Windows) -> np.ndarray:
    return np.bincount(windows.walk, weights=values, minlength=windows.n_walks) / windows.counts()


class PlanNorm:
    def fit(self, windows: Windows) -> "PlanNorm":
        values = windows.x[_valid(windows.lengths)]
        self.mean = values.mean(axis=0).astype(np.float32)
        self.std = (values.std(axis=0) + 1e-4).astype(np.float32)
        return self

    def transform(self, windows: Windows) -> np.ndarray:
        x = (windows.x - self.mean) / self.std
        x[~_valid(windows.lengths)] = 0.0
        return x.astype(np.float32)


class PlanGRU(nn.Module):
    def __init__(self, n_out: int, ordered: bool = False, in_ch: int = len(CHANNELS),
                 proj: int = 64, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.ordered = ordered
        self.proj = nn.Linear(in_ch, proj)
        self.gru = nn.GRU(proj, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_out)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.proj(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        states, last = self.gru(packed)
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=x.shape[1])
        mask = (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (states * mask).sum(dim=1) / lengths[:, None].float()
        z = self.out(self.drop(torch.cat([last[-1], mean], dim=-1)))
        if self.ordered:
            z = torch.cat([z[:, :1], z[:, :1] - torch.cumsum(F.softplus(z[:, 1:]), dim=1)], dim=1)
        return z


def ordinal_cumulative(logits: torch.Tensor, const: dict | None = None) -> torch.Tensor:
    p = torch.sigmoid(logits)
    for k, v in (const or {}).items():
        p[..., k] = v
    return torch.cummin(p, dim=-1).values


def expected_flat(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1) @ torch.arange(4, dtype=logits.dtype, device=logits.device)


def window_logits(model: PlanGRU, x: np.ndarray, lengths: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = torch.from_numpy(x[s : s + batch]).to(DEVICE)
            lb = torch.from_numpy(lengths[s : s + batch]).to(DEVICE)
            parts.append(model(xb, lb).cpu())
    return torch.cat(parts).numpy()


def walk_scores(kind: str, logits: np.ndarray, windows: Windows, temperature: float = 1.0,
                const: dict | None = None) -> np.ndarray:
    z = torch.from_numpy(logits / temperature)
    per_window = expected_flat(z) if kind == "flat" else ordinal_cumulative(z, const).sum(dim=-1)
    return to_walks(per_window.numpy().astype(np.float64), windows)


class FittedPlanGRU:
    def __init__(self, kind: str, model: PlanGRU, norm: PlanNorm, temperature: float, iso, const: dict, info: dict):
        self.kind, self.model, self.norm, self.temperature, self.iso, self.const, self.info = (
            kind, model, norm, temperature, iso, const, info)

    def predict(self, windows: Windows) -> dict:
        logits = window_logits(self.model, self.norm.transform(windows), windows.lengths)
        raw = walk_scores(self.kind, logits, windows, 1.0, self.const)
        out = {
            "raw": raw,
            "temperature": walk_scores(self.kind, logits, windows, self.temperature, self.const),
            "isotonic": np.clip(self.iso.predict(raw), 0.0, 3.0) if self.iso is not None else raw,
        }
        if self.kind == "ordinal":
            cumulative = ordinal_cumulative(torch.from_numpy(logits), self.const).numpy().astype(np.float64)
            out["cumulative"] = np.column_stack([to_walks(cumulative[:, k], windows) for k in range(3)])
        return out


class IsotonicMap:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(values, dtype=np.float64), self.x, self.y)


def load_fitted(path: str) -> FittedPlanGRU:
    blob = torch.load(path, weights_only=True, map_location="cpu")
    ordered = blob["kind"] == "ordinal"
    model = PlanGRU(3 if ordered else 4, ordered=ordered, hidden=blob["hidden"], dropout=blob["dropout"]).to(DEVICE)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    norm = PlanNorm()
    norm.mean, norm.std = blob["scaler_mean"].numpy(), blob["scaler_std"].numpy()
    iso = None if blob["isotonic_x"] is None else IsotonicMap(blob["isotonic_x"].numpy(), blob["isotonic_y"].numpy())
    return FittedPlanGRU(blob["kind"], model, norm, blob["temperature"], iso,
                         dict(blob["constant_thresholds"]),
                         {"hidden": blob["hidden"], "dropout": blob["dropout"]})


# =====================================================================================================================
# 2. Data, saved results and which models score which walk
# =====================================================================================================================

COHORTS = ["PD-GaM", "BMCLab", "T-SDU-PD", "3DGait"]
SEEDS = [1000, 1001, 1002, 1003, 1004]
LEVELS = ["Normal", "Slight", "Mild", "Moderate"]
MODELS_DIR = ROOT / "A_LODO" / "models"
DATA_DIR = ROOT / "data" / "CARE-PD" / "Canonicalized_SMPL_pickles"
RUN_DIR = ROOT / "results" / "versions" / "v3_plan"

DISCLAIMER = (
    "Research prototype on CARE-PD, non-commercial. **Not a diagnosis and not validated for clinical use.** "
    "On cohorts it never saw, the ordinal GRU (each seed scored on its own, then averaged over five seeds) was off by "
    "0.46 to 0.76 of a point on average, and on T-SDU-PD and 3DGait it was no better than always answering the training median."
)


@lru_cache(maxsize=1)
def saved_predictions() -> dict:
    with open(RUN_DIR / "lodo_predictions.json") as handle:
        raw = json.load(handle)
    out = {}
    for cohort in COHORTS:
        block = raw[cohort]
        index = {(str(s), str(w)): i for i, (s, w) in enumerate(zip(block["subject_id"], block["walk_id"]))}
        out[cohort] = {"block": block, "index": index}
    return out


@lru_cache(maxsize=1)
def run_metrics() -> dict:
    with open(RUN_DIR / "metrics.json") as handle:
        return json.load(handle)


def training_median(cohort: str) -> float:
    preds = saved_predictions()
    y = np.concatenate([np.asarray(preds[c]["block"]["y"], dtype=float) for c in COHORTS if c != cohort])
    return float(np.median(y))


def _natural(text: str):
    import re
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def subjects(cohort: str) -> list[str]:
    block = saved_predictions()[cohort]["block"]
    return sorted({str(s) for s in block["subject_id"]}, key=_natural)


def walks_of(cohort: str, subject: str) -> list[str]:
    block = saved_predictions()[cohort]["block"]
    return [str(w) for s, w in zip(block["subject_id"], block["walk_id"]) if str(s) == str(subject)]


@lru_cache(maxsize=4)
def cohort_data(cohort: str) -> dict:
    with open(DATA_DIR / f"{cohort}_canonical.pkl", "rb") as handle:
        return pickle.load(handle)


def release_walk(cohort: str, subject: str, walk_id: str) -> dict:
    raw = cohort_data(cohort)
    for key in raw:
        if str(key) == str(subject):
            return raw[key][walk_id]
    raise KeyError(f"{cohort}: subject {subject} not found")


def _is_walk(obj) -> bool:
    return isinstance(obj, dict) and "pose" in obj and "trans" in obj


@lru_cache(maxsize=2)
def _read_upload(path: str, mtime: float) -> tuple:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    name = Path(path).stem
    if _is_walk(obj):
        return (("", name, obj),)
    if not isinstance(obj, dict):
        raise ValueError("Not a CARE-PD walk: expected a dict with 'pose' and 'trans', or the release's nested dict.")
    found = []
    for key, value in obj.items():
        if _is_walk(value):
            found.append(("", str(key), value))
        elif isinstance(value, dict):
            found.extend((str(key), str(walk_id), walk) for walk_id, walk in value.items() if _is_walk(walk))
    if not found:
        raise ValueError("No walk with 'pose' and 'trans' was found in this file.")
    return tuple(found)


def read_upload(path: str) -> tuple:
    return _read_upload(path, os.path.getmtime(path))


def content_key(walk: dict) -> str:
    pose = np.asarray(walk["pose"], dtype=np.float64)
    if pose.ndim == 3:
        pose = pose.reshape(pose.shape[0], -1)
    trans = np.asarray(walk["trans"], dtype=np.float64)
    digest = hashlib.sha256(f"{pose.shape}|{trans.shape}".encode())
    digest.update(np.ascontiguousarray(pose).tobytes())
    digest.update(np.ascontiguousarray(trans).tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def content_index() -> dict:
    index = {}
    for cohort in COHORTS:
        block = saved_predictions()[cohort]["block"]
        for subject, walk_id in zip(block["subject_id"], block["walk_id"]):
            index.setdefault(content_key(release_walk(cohort, subject, walk_id)), (cohort, str(subject), str(walk_id)))
    return index


def identify(walk: dict) -> tuple[str, str, str] | None:
    return content_index().get(content_key(walk))


def scored_subjects() -> dict:
    out = {}
    for cohort in COHORTS:
        for subject in subjects(cohort):
            out.setdefault(subject, []).append(cohort)
    return out


def upload_warnings(subject: str, walk: dict) -> list[str]:
    notes = []
    trans = np.asarray(walk["trans"], dtype=np.float64)
    if trans.ndim == 2 and trans.shape[0] and np.abs(trans[0, [0, 2]]).max() > 1e-3:
        notes.append(f"Its translation starts at x = {trans[0, 0]:.2f} m, z = {trans[0, 2]:.2f} m, not at 0: it is probably not "
                     "in CARE-PD's canonical frame. Use the files in `data/CARE-PD/Canonicalized_SMPL_pickles/`; the raw "
                     "release files beside them are in another frame, and the models expect the canonical one.")
    if subject and str(subject) in scored_subjects():
        notes.append(f"Its subject id `{subject}` is also a subject of {', '.join(scored_subjects()[str(subject)])}. Subject "
                     "ids are local to a cohort, but if this is the same person, the ranker models trained on that person's "
                     "walks and the score is not held out.")
    if walk.get("UPDRS_GAIT") is not None:
        notes.append(f"The file carries UPDRS_GAIT = {walk['UPDRS_GAIT']}. It is not used as a held-out comparison here, "
                     "because the walk could not be matched to a scored-cohort walk.")
    return notes


def check_walk(walk: dict) -> list[str]:
    problems = []
    pose, trans = np.asarray(walk.get("pose")), np.asarray(walk.get("trans"))
    if pose.ndim == 3:
        pose = pose.reshape(pose.shape[0], -1)
    if pose.ndim != 2 or pose.shape[-1] != 72:
        problems.append(f"pose must be frames x 72 SMPL axis-angle values, got shape {pose.shape}")
    if trans.ndim != 2 or trans.shape[-1] != 3:
        problems.append(f"trans must be frames x 3, got shape {trans.shape}")
    if not problems and pose.shape[0] < 8:
        problems.append(f"only {pose.shape[0]} frames; the study drops walks under 8 frames")
    if not problems and pose.shape[0] != trans.shape[0]:
        problems.append(f"pose has {pose.shape[0]} frames but trans has {trans.shape[0]}")
    if not problems and not (np.isfinite(pose.astype(np.float64)).all() and np.isfinite(trans.astype(np.float64)).all()):
        problems.append("pose or trans contains NaN or infinite values")
    try:
        fps = float(walk.get("fps", 0))
        if not fps > 0:
            problems.append("fps is missing")
        elif not problems and resample(np.zeros((pose.shape[0], 1)), fps).shape[0] < 3:
            problems.append(f"{pose.shape[0]} frames at {fps:g} Hz is under 3 frames at 30 Hz, too short for the model's inputs")
    except (TypeError, ValueError):
        problems.append("fps is not a number")
    return problems


OLD_TORCH = "weights_only" not in inspect.signature(torch.load).parameters


@lru_cache(maxsize=1)
def model_checksums() -> dict:
    sums = {}
    for line in (MODELS_DIR / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[Path(name.strip().lstrip("*")).name] = digest
    return sums


def _load_model(path: Path):
    if not OLD_TORCH:
        return load_fitted(str(path))
    expected = model_checksums().get(path.name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError(f"{path.name} does not match A_LODO/models/SHA256SUMS; refusing to load it with torch "
                           f"{torch.__version__}, which cannot restrict loading to weights.")
    original = torch.load
    torch.load = lambda *args, weights_only=None, **kwargs: original(*args, **kwargs)
    try:
        return load_fitted(str(path))
    finally:
        torch.load = original


@lru_cache(maxsize=16)
def model_set(family: str, kind: str) -> tuple:
    prefix = "ranker" if family == "ranker" else f"lodo_{family}"
    paths = [MODELS_DIR / f"{prefix}_seed{seed}_{kind}.pt" for seed in SEEDS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing model files in {MODELS_DIR}: {missing}")
    return tuple(_load_model(p) for p in paths)


def walk_kin(walk: dict) -> np.ndarray:
    pose = np.asarray(walk["pose"], dtype=np.float64)
    if pose.ndim == 3:
        pose = pose.reshape(pose.shape[0], -1)
    return plan_channels(pose, np.asarray(walk["trans"], dtype=np.float64), float(walk["fps"]))


def score_kins(kins: list, family: str) -> dict:
    windows = build_windows(kins)
    return {kind: np.column_stack([model.predict(windows)["raw"] for model in model_set(family, kind)])
            for kind in ("ordinal", "flat")}


def score_walk(walk: dict, cohort: str | None) -> dict:
    family = cohort if cohort in COHORTS else "ranker"
    kin = walk_kin(walk)
    scores = score_kins([kin], family)
    out = {"family": family, "kin": kin, "n_windows": len(window_starts(len(kin)))}
    for kind in ("ordinal", "flat"):
        seeds = scores[kind][0]
        out[kind] = {"seeds": seeds, "mean": float(seeds.mean())}
    return out


def level(score: float) -> str:
    return LEVELS[int(np.clip(np.rint(score), 0, 3))]


def saved_row(cohort: str, subject: str, walk_id: str) -> dict | None:
    pred = saved_predictions()[cohort]
    i = pred["index"].get((str(subject), str(walk_id)))
    if i is None:
        return None
    block = pred["block"]
    return {"y": int(block["y"][i]),
            "ordinal_gru": float(block["ordinal_gru"][i]),
            "flat_gru": float(block["flat_gru"][i]),
            "frank_hall": float(block["frank_hall"][i]),
            "ordinal_seeds": np.array([float(seed[i]) for seed in block["ordinal_gru_seed_scores"]])}


# =====================================================================================================================
# 3. Viewer skeleton and figures (heatmap removed)
# =====================================================================================================================

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

_REST = np.array([
    [0.00, 0.00, 0.00], [0.07, -0.08, 0.00], [-0.07, -0.08, 0.00], [0.00, 0.10, -0.02],
    [0.08, -0.46, 0.00], [-0.08, -0.46, 0.00], [0.00, 0.23, -0.01],
    [0.08, -0.86, -0.03], [-0.08, -0.86, -0.03], [0.00, 0.29, 0.00],
    [0.10, -0.92, 0.10], [-0.10, -0.92, 0.10], [0.00, 0.50, -0.02],
    [0.07, 0.41, -0.01], [-0.07, 0.41, -0.01], [0.00, 0.62, 0.03],
    [0.18, 0.44, -0.02], [-0.18, 0.44, -0.02], [0.43, 0.42, -0.03], [-0.43, 0.42, -0.03],
    [0.68, 0.43, -0.03], [-0.68, 0.43, -0.03], [0.76, 0.42, -0.04], [-0.76, 0.42, -0.04],
])

OFFSETS = np.array([_REST[j] - (_REST[PARENTS[j]] if PARENTS[j] >= 0 else 0.0) for j in range(24)])

LEFT_SIDE = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT_SIDE = {2, 5, 8, 11, 14, 17, 19, 21, 23}


def joints(walk: dict, fps_out: float = TARGET_FPS) -> tuple[np.ndarray, float]:
    pose = np.asarray(walk["pose"], dtype=np.float64)
    if pose.ndim == 3:
        pose = pose.reshape(pose.shape[0], -1)
    pose = resample(pose, float(walk["fps"]), fps_out)
    trans = resample(np.asarray(walk["trans"], dtype=np.float64), float(walk["fps"]), fps_out)
    n = min(len(pose), len(trans))
    local = Rotation.from_rotvec(pose[:n].reshape(-1, 3)).as_matrix().reshape(n, 24, 3, 3)
    world = np.zeros_like(local)
    pos = np.zeros((n, 24, 3))
    world[:, 0], pos[:, 0] = local[:, 0], trans[:n]
    for j in range(1, 24):
        p = PARENTS[j]
        world[:, j] = world[:, p] @ local[:, j]
        pos[:, j] = pos[:, p] + np.einsum("fij,j->fi", world[:, p], OFFSETS[j])
    pos[:, :, 1] -= np.median(pos[:, :, 1].min(axis=1)) - 0.03
    return pos, fps_out


def skeleton_html(walk: dict, title: str, max_seconds: float = 12.0, fps_out: float = 15.0) -> str:
    pos, fps = joints(walk, fps_out)
    note = ""
    if len(pos) / fps > max_seconds:
        note = f"First {max_seconds:.0f} s of {len(pos) / fps:.1f} s shown. "
        pos = pos[: int(max_seconds * fps)]
    fig = plt.figure(figsize=(4.4, 4.6), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    floor = pos[:, 0, [0, 2]]

    def draw(frame: int):
        ax.cla()
        cx, cz = pos[frame, 0, 0], pos[frame, 0, 2]
        ax.plot(floor[:, 0], floor[:, 1], np.zeros(len(floor)), color="0.78", lw=1)
        ax.plot(floor[: frame + 1, 0], floor[: frame + 1, 1], np.zeros(frame + 1), color="0.35", lw=1.6)
        for j in range(1, 24):
            p = PARENTS[j]
            color = "#1f6fb4" if j in LEFT_SIDE else "#d4552b" if j in RIGHT_SIDE else "0.15"
            seg = pos[frame, [p, j]]
            ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color=color, lw=2.4)
        ax.scatter([pos[frame, 15, 0]], [pos[frame, 15, 2]], [pos[frame, 15, 1]], s=60, color="0.15")
        ax.set_xlim(cx + 1.0, cx - 1.0)
        ax.set_ylim(cz - 1.0, cz + 1.0)
        ax.set_zlim(0, 2.0)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=12, azim=-60)
        ax.set_xlabel("x lateral (m)", fontsize=7, labelpad=-8)
        ax.set_ylabel("z forward (m)", fontsize=7, labelpad=-8)
        ax.set_zlabel("y up (m)", fontsize=7, labelpad=-8)
        ax.tick_params(labelsize=6, pad=-3)
        ax.set_title(f"{title}\n{frame / fps:4.1f} s    blue = left, orange = right", fontsize=8)

    anim = animation.FuncAnimation(fig, draw, frames=len(pos), interval=1000 / fps)
    folder = Path(os.environ.get("CAREPD_UI_TMP", tempfile.gettempdir())) / "carepd_final_ui"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"skeleton_{os.getpid()}.gif"
    try:
        anim.save(str(path), writer=animation.PillowWriter(fps=fps))
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        plt.close(fig)
        if path.exists():
            path.unlink()
    return (f'<div style="text-align:center"><img src="data:image/gif;base64,{data}" style="max-width:100%"/>'
            f'<div style="font-size:0.8em;opacity:0.75">{note}Viewer skeleton with generic bone lengths, '
            "driven by the walk's own joint rotations and pelvis translation; not the subject's body.</div></div>")


def motion_figure(walk: dict, kin: np.ndarray, title: str):
    """Pelvis path from above, horizontal speed, and ankle height. Heatmap removed."""
    pos, fps = joints(walk, TARGET_FPS)
    pelvis = pos[:, 0]
    t = np.arange(len(pelvis)) / fps
    fig = plt.figure(figsize=(12, 3.8), dpi=90)
    # 1x3 grid instead of 2x3
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 1.4], wspace=0.55, left=0.08, right=0.93)
    
    ax = fig.add_subplot(grid[0, 0])
    sc = ax.scatter(pelvis[:, 0], pelvis[:, 2], c=t, s=4, cmap="viridis")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_xaxis()
    ax.set_xlabel("x lateral (m, + = subject's left)")
    ax.set_ylabel("z forward (m)")
    ax.set_title("Pelvis path from above", fontsize=10)
    fig.colorbar(sc, ax=ax, label="time (s)", fraction=0.05)
    
    ax = fig.add_subplot(grid[0, 1])
    speed = kin[:, CHANNELS.index("horiz_speed")]
    ax.plot(np.arange(len(speed)) / TARGET_FPS, speed, color="#1f6fb4", lw=1.2)
    ax.axhline(PAUSE_SPEED, color="0.5", ls="--", lw=0.8)
    ax.text(0, PAUSE_SPEED + 0.01, f"pause below {PAUSE_SPEED} m/s", fontsize=7, color="0.4")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/s")
    ax.set_title("Horizontal pelvis speed (a model input)", fontsize=10)
    
    ax = fig.add_subplot(grid[0, 2])
    for joint, name, color in [(L_ANKLE, "left ankle", "#1f6fb4"), (R_ANKLE, "right ankle", "#d4552b")]:
        ax.plot(t, pos[:, joint, 1], lw=1.0, color=color, label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height (m)")
    ax.set_title("Ankle height (viewer skeleton)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    
    fig.suptitle(title, fontsize=11)
    return fig


def comparison_figure(result: dict, actual: int | None, saved: dict | None, median: float | None, title: str):
    rows = [("Ordinal GRU (proposed)\nmean of 5 seeds", result["ordinal"]["mean"], result["ordinal"]["seeds"]),
            ("Flat GRU\nmean of 5 seeds", result["flat"]["mean"], result["flat"]["seeds"])]
    if saved is not None:
        rows.append(("Frank-Hall\n(saved by the run)", saved["frank_hall"], None))
    if median is not None:
        rows.append(("Median baseline\n(training median)", median, None))
    fig, ax = plt.subplots(figsize=(8.5, 1.3 + 0.8 * len(rows)), dpi=90)
    for i, (name, value, seeds) in enumerate(rows):
        y = len(rows) - 1 - i
        if seeds is not None:
            ax.plot([seeds.min(), seeds.max()], [y, y], color="0.6", lw=7, solid_capstyle="round", alpha=0.5)
            ax.scatter(seeds, [y] * len(seeds), s=16, color="0.3", zorder=3)
        ax.scatter([value], [y], s=120, color="#1f6fb4", zorder=4, edgecolor="white")
        ax.text(value, y + 0.3, f"{value:.2f}", ha="center", fontsize=9)
    if actual is not None:
        ax.axvline(actual, color="#c0392b", lw=2.4)
        align = {0: ("left", 0.04), 3: ("right", -0.04)}.get(actual, ("center", 0.0))
        ax.text(actual + align[1], len(rows) - 0.3, f"clinician: {actual} ({LEVELS[actual]})", color="#c0392b",
                ha=align[0], fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-0.6, len(rows) + 0.05)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([f"{k} {LEVELS[k]}" for k in range(4)], fontsize=8)
    for edge in (0.5, 1.5, 2.5):
        ax.axvline(edge, color="0.85", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def study_rows() -> list[list]:
    lodo = run_metrics()["lodo"]
    rows = []
    for cohort in COHORTS:
        row = [cohort, f"{float(lodo['median'][cohort]['mae']):.3f}"]
        for key in ("frank_hall", "flat_gru_raw", "ordinal_gru_raw"):
            row.append(f"{float(lodo[key][cohort]['qwk']):.3f} / {float(lodo[key][cohort]['mae']):.3f}")
        rows.append(row)
    return rows


# =====================================================================================================================
# 4. Scoring one walk for the page
# =====================================================================================================================

SCORE_HEADERS = ["", "score (0-3)", "level", "|error| vs clinician", "source"]
DATASET_SOURCE, UPLOAD_SOURCE = "A walk from the dataset", "The uploaded file"


def evaluate(walk: dict, cohort: str | None, subject: str, walk_id: str) -> dict:
    problems = check_walk(walk)
    if problems:
        raise ValueError("This walk cannot be scored: " + "; ".join(problems))
    result = score_walk(walk, cohort)
    saved = saved_row(cohort, subject, walk_id) if cohort in COHORTS else None
    actual = saved["y"] if saved is not None else None
    if actual is None and "UPDRS_GAIT" in walk and cohort in COHORTS:
        actual = int(walk["UPDRS_GAIT"])
    median = training_median(cohort) if cohort in COHORTS else None
    label = f"{cohort or 'not a scored-cohort walk'} · {subject or '-'} · {walk_id}"
    pose = np.asarray(walk["pose"])
    frames, fps = pose.shape[0], float(walk["fps"])

    def err(value):
        return "" if actual is None else f"{abs(value - actual):.2f}"

    family_text = (f"`lodo_{cohort}_seed1000-1004`: trained on {', '.join(c for c in COHORTS if c != cohort)} only, "
                   f"so they never saw {cohort}. This is a held-out prediction, as in the study."
                   if cohort in COHORTS else
                   "`ranker_seed1000-1004`: trained on all 2,953 walks of the four scored cohorts. This walk matched none "
                   "of those walks by content, so it is not one of their training walks unless it is a CARE-PD walk "
                   "stored differently (another coordinate frame, or a copy in another cohort file). There is no clinician "
                   "score to compare with.")
    o, f = result["ordinal"], result["flat"]
    rows = [["Ordinal GRU (proposed), mean of 5 seeds", round(o["mean"], 3), level(o["mean"]), err(o["mean"]), "computed now"],
            ["  its five seeds", f"{o['seeds'].min():.2f} to {o['seeds'].max():.2f}", "", "", "computed now"],
            ["Flat GRU, mean of 5 seeds", round(f["mean"], 3), level(f["mean"]), err(f["mean"]), "computed now"]]
    if saved is not None:
        rows.append(["Frank-Hall", round(saved["frank_hall"], 3), level(saved["frank_hall"]), err(saved["frank_hall"]),
                     "saved by the primary run"])
    if median is not None:
        rows.append(["Median baseline", median, level(median), err(median), "training median"])
    if actual is not None:
        rows.append(["Clinician (MDS-UPDRS 3.10)", actual, LEVELS[actual], "", "CARE-PD release"])

    lines = [f"### {label}"]
    if actual is not None:
        lines.append(f"**Clinician: {actual} ({LEVELS[actual]})** · **Ordinal GRU: {o['mean']:.2f} ({level(o['mean'])})**"
                     f" · error {abs(o['mean'] - actual):.2f} of a point · five seeds {o['seeds'].min():.2f} to {o['seeds'].max():.2f}")
    else:
        lines.append(f"**Ordinal GRU: {o['mean']:.2f} ({level(o['mean'])})** · five seeds {o['seeds'].min():.2f} to "
                     f"{o['seeds'].max():.2f} · no clinician score for this walk")
    lines.append(f"Models: {family_text}")
    if saved is not None:
        diff = float(np.max(np.abs(o["seeds"] - saved["ordinal_seeds"])))
        lines.append(f"Check: the scores computed now match the primary run's saved scores for this walk "
                     f"(largest seed difference {diff:.1e}; float32 rounding between CPU and GPU reaches about 5e-4)." if diff < 1e-3 else
                     f"**Warning:** the scores computed now differ from the primary run's saved scores by {diff:.3f}.")
    lines.append(f"{frames} frames at {fps:g} Hz ({frames / fps:.1f} s), {len(result['kin'])} input frames at 30 Hz, "
                 f"{result['n_windows']} one-second window(s); the walk's score is the mean over its windows.")
    title = f"{label}"
    return {
        "summary": "\n\n".join(lines),
        "rows": rows,
        "skeleton": skeleton_html(walk, f"{cohort or 'upload'} {walk_id}"),
        "compare": comparison_figure(result, actual, saved, median, f"{label}: prediction vs clinician"),
        "motion": motion_figure(walk, result["kin"], title),
        "result": result,
        "actual": actual,
    }


# =====================================================================================================================
# 5. Self-check: python app.py --check
# =====================================================================================================================

def self_check(limit: int | None = None) -> int:
    results = []

    def check(name: str, ok: bool, detail: str = ""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)

    sums = {}
    for line in (MODELS_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*").lstrip("./")] = digest
    bad = [n for n, d in sums.items()
           if (MODELS_DIR / n).exists() and hashlib.sha256((MODELS_DIR / n).read_bytes()).hexdigest() != d]
    missing = [n for n in sums if not (MODELS_DIR / n).exists()]
    check("model files match A_LODO/models/SHA256SUMS", not bad and not missing,
          f"{len(sums)} files; bad {bad}; missing {missing}")

    folds = json.load(open(RUN_DIR / "fold_manifest.json"))
    splits = json.load(open(RUN_DIR / "split_manifest.json"))
    for cohort in COHORTS:
        trained_on = splits["lodo"][cohort]["train_subjects"]
        trained_on = trained_on if isinstance(trained_on, dict) else __import__("ast").literal_eval(trained_on)
        check(f"{cohort}: the LODO split trained on {sorted(trained_on)} only", cohort not in trained_on)
        worst = 0.0
        for seed, model in zip(SEEDS, model_set(cohort, "ordinal")):
            ref = folds["lodo"][cohort]["gru"][str(seed)]["scaler"]
            worst = max(worst, float(np.max(np.abs(model.norm.mean - np.asarray(ref["mean"])))),
                        float(np.max(np.abs(model.norm.std - np.asarray(ref["std"])))))
        check(f"{cohort}: lodo_{cohort} ordinal models carry that split's recorded scalers", worst < 1e-5,
              f"largest difference {worst:.1e}")

    total, worst_all = 0, 0.0
    for cohort in COHORTS:
        block = saved_predictions()[cohort]["block"]
        n = len(block["walk_id"]) if limit is None else min(limit, len(block["walk_id"]))
        kins = [walk_kin(release_walk(cohort, block["subject_id"][i], block["walk_id"][i])) for i in range(n)]
        scores = score_kins(kins, cohort)
        worst = 0.0
        for kind in ("ordinal", "flat"):
            saved = np.asarray(block[f"{kind}_gru_seed_scores"], dtype=float)[:, :n].T
            worst = max(worst, float(np.max(np.abs(scores[kind] - saved))))
        mean_diff = float(np.max(np.abs(scores["ordinal"].mean(axis=1) - np.asarray(block["ordinal_gru"][:n]))))
        check(f"{cohort}: {n} walks rescored now equal the run's saved held-out seed scores", worst < 1e-3,
              f"largest difference {worst:.1e} (seed means {mean_diff:.1e})")
        total, worst_all = total + n, max(worst_all, worst)
        y = np.asarray(block["y"], dtype=float)
        mae = float(np.mean(np.abs(y - training_median(cohort))))
        ref = float(run_metrics()["lodo"]["median"][cohort]["mae"])
        check(f"{cohort}: median baseline MAE {mae:.4f} equals metrics.json", abs(mae - ref) < 1e-9)

    index = content_index()
    probes = [(c, str(s), str(w)) for c in COHORTS
              for s, w in list(zip(saved_predictions()[c]["block"]["subject_id"],
                                    saved_predictions()[c]["block"]["walk_id"]))[:3]]
    renamed_ok = all(identify(dict(release_walk(*probe))) == probe for probe in probes)
    check("uploads are matched by content: all 2,953 scored walks have distinct fingerprints, and 12 walks are found "
          "whatever their name", len(index) == 2953 and renamed_ok, f"{len(index)} fingerprints")
    pos, _ = joints(release_walk("BMCLab", "SUB01", "SUB01_off_walk_1"))
    check("viewer skeleton is finite, 24 joints", bool(np.isfinite(pos).all()) and pos.shape[1] == 24, f"shape {pos.shape}")
    print(f"\n{sum(results)} of {len(results)} checks pass; {total} walks rescored, largest seed difference {worst_all:.1e}")
    return 0 if all(results) else 1


# =====================================================================================================================
# 6. The page
# =====================================================================================================================

ABOUT = f"""
**What this page does.** It takes one walk from CARE-PD, shows it, scores its MDS-UPDRS gait item (3.10, 0 to 3) with the
study's proposed model, and puts that score next to the clinician's. Package version {VERSION}.

**Which model scores which walk.** The study trained, for each scored cohort, five ordinal GRUs (seeds 1000 to 1004) on
the other three cohorts only (leave-one-dataset-out, LODO). A walk from PD-GaM is scored by `lodo_PD-GaM_*`, which never saw
PD-GaM, and so on. The page shows the mean of the five seeds' scores. An uploaded walk is matched to the release by
content (identical pose and translation), whatever its file or walk name. A walk that matches none of the 2,953
scored-cohort walks is scored by `ranker_*`, the models trained on all four cohorts, and has no clinician score here.

**What the numbers can and cannot say.**
- On held-out cohorts the mean error of a single seed, averaged over the five seeds, was 0.46 to 0.76 of a point. On
  T-SDU-PD and 3DGait no learned predictor beat always answering the training median.
- The seed-averaged score never rounds to 3 (Moderate) on a held-out walk, even when the clinician said 3.
- Frank-Hall scores come from the run's saved predictions; no Frank-Hall LODO model was saved.
- The stick figure uses generic bone lengths. The model never sees it; it sees joint rotations and pelvis translation.

**Data and licence.** CARE-PD, non-commercial: its Terms of Use state CC BY-NC 4.0, and its dataset card tags
CC BY-NC-ND 4.0 (`data/CARE-PD/README.md`). Cite CARE-PD and its source cohorts, make no attempt to identify participants,
and make no clinical claim. Not a diagnostic device.
"""


def build_page():
    import gradio as gr

    def choices_for(cohort):
        subs = subjects(cohort)
        walks = walks_of(cohort, subs[0]) if subs else []
        return (gr.update(choices=subs, value=subs[0] if subs else None),
                gr.update(choices=walks, value=walks[0] if walks else None), gr.update(value=DATASET_SOURCE))

    def walks_for(cohort, subject):
        walks = walks_of(cohort, subject) if subject else []
        return gr.update(choices=walks, value=walks[0] if walks else None), gr.update(value=DATASET_SOURCE)

    def random_walk(cohort):
        block = saved_predictions()[cohort]["block"]
        i = int(np.random.default_rng().integers(len(block["walk_id"])))
        subject = str(block["subject_id"][i])
        return (gr.update(value=subject),
                gr.update(choices=walks_of(cohort, subject), value=str(block["walk_id"][i])),
                gr.update(value=DATASET_SOURCE))

    def on_upload(path):
        if not path:
            return gr.update(choices=[], value=None), gr.update(value=DATASET_SOURCE)
        try:
            found = read_upload(path)
        except Exception as error:
            hint = (" The file was written with NumPy 2 and this environment has NumPy 1; use the package's environment "
                    "(environment.yml)." if "numpy._core" in str(error) else "")
            raise gr.Error(f"Could not read this file: {error}.{hint}")
        labels = [f"{s} / {w}" if s else w for s, w, _ in found]
        return gr.update(choices=labels, value=labels[0]), gr.update(value=UPLOAD_SOURCE)

    def run(source, cohort, subject, walk_id, path, upload_choice):
        try:
            if source == UPLOAD_SOURCE:
                if not path or not upload_choice:
                    raise ValueError("Upload a .pkl first, or choose 'A walk from the dataset'.")
                found = read_upload(path)
                labels = [f"{s} / {w}" if s else w for s, w, _ in found]
                if upload_choice not in labels:
                    raise ValueError(f"'{upload_choice}' is not a walk in the uploaded file.")
                subj, wid, walk = found[labels.index(upload_choice)]
                problems = check_walk(walk)
                if problems:
                    raise ValueError("This walk cannot be scored: " + "; ".join(problems))
                match = identify(walk)
                if match is None:
                    out = evaluate(walk, None, subj, wid)
                    notes = ["**This walk matched none of the 2,953 walks of CARE-PD's four scored cohorts** "
                             "(compared by content: pose and translation must be identical), so there is no clinician "
                             "score to compare with."]
                    notes += [f"- {note}" for note in upload_warnings(subj, walk)]
                    out["summary"] = "\n\n".join(notes) + "\n\n" + out["summary"]
                else:
                    fc, fs, fw = match
                    out = evaluate(walk, fc, fs, fw)
                    renamed = "" if (str(wid), str(subj or fs)) == (fw, fs) else \
                        f" (uploaded as `{(subj + ' / ') if subj else ''}{wid}`)"
                    out["summary"] = (f"Matched by content to a release walk: {fc}, subject {fs}, walk "
                                      f"{fw}{renamed}.\n\n" + out["summary"])
            else:
                if not walk_id or walk_id not in walks_of(cohort, subject):
                    raise ValueError(f"Choose a walk: '{walk_id}' is not a walk of subject {subject} in {cohort}.")
                out = evaluate(release_walk(cohort, subject, walk_id), cohort, subject, walk_id)
        except gr.Error:
            plt.close("all")
            raise
        except Exception as error:
            plt.close("all")
            raise gr.Error(str(error))
        for fig in (out["compare"], out["motion"]):
            if fig is not None:
                plt.close(fig)
        return out["summary"], out["rows"], out["skeleton"], out["compare"], out["motion"]

    first_cohort = "BMCLab"
    first_subjects = subjects(first_cohort)
    first_walks = walks_of(first_cohort, first_subjects[0])
    with gr.Blocks(title="CARE-PD gait score: held-out prediction vs clinician") as page:
        gr.Markdown(f"# CARE-PD gait score: held-out prediction vs clinician\n{DISCLAIMER}")
        with gr.Tab("Score a walk"):
            with gr.Row():
                source = gr.Radio([DATASET_SOURCE, UPLOAD_SOURCE], value=DATASET_SOURCE, label="Walk to score")
            with gr.Row():
                cohort = gr.Dropdown(COHORTS, value=first_cohort, label="Cohort")
                subject = gr.Dropdown(first_subjects, value=first_subjects[0], label="Subject", allow_custom_value=True)
                walk = gr.Dropdown(first_walks, value=first_walks[0], label="Walk", allow_custom_value=True)
                pick = gr.Button("Random walk")
            with gr.Row():
                upload = gr.File(label="Or upload a CARE-PD .pkl: one walk, or a canonical cohort file "
                                       "(data/CARE-PD/Canonicalized_SMPL_pickles/<cohort>_canonical.pkl). "
                                       "Only files you trust: a pickle can run code",
                                 file_types=[".pkl"], type="filepath")
                upload_walk = gr.Dropdown([], label="Walk in the uploaded file", allow_custom_value=True)
            go = gr.Button("Score", variant="primary")
            summary = gr.Markdown()
            table = gr.Dataframe(headers=SCORE_HEADERS, label="This walk", interactive=False)
            with gr.Row():
                skeleton = gr.HTML(label="The walk")
                compare = gr.Plot(label="Prediction vs clinician", format="png")
            # Cohort section removed, only motion plot remains
            motion = gr.Plot(label="Motion", format="png")
        with gr.Tab("Study results"):
            gr.Markdown("Leave-one-dataset-out, QWK / MAE of the expected score, each seed's value averaged over 5 seeds, raw scores "
                        "(`results/versions/v3_plan/metrics.json`). The report is `arxiv_proposal/versions/v4.3/main.pdf`.")
            gr.Dataframe(value=study_rows(),
                         headers=["held-out cohort", "median MAE", "Frank-Hall", "flat GRU", "ordinal GRU"],
                         interactive=False)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)

        cohort.input(choices_for, cohort, [subject, walk, source])
        subject.input(walks_for, [cohort, subject], [walk, source])
        walk.input(lambda: gr.update(value=DATASET_SOURCE), None, source)
        pick.click(random_walk, cohort, [subject, walk, source])
        upload.change(on_upload, upload, [upload_walk, source])
        outputs = [summary, table, skeleton, compare, motion]
        go.click(run, [source, cohort, subject, walk, upload, upload_walk], outputs)
        page.load(run, [source, cohort, subject, walk, upload, upload_walk], outputs)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="run the self-check and exit")
    parser.add_argument("--limit", type=int, default=None, help="with --check: walks per cohort (default all)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.check:
        sys.exit(self_check(args.limit))
    page = build_page()
    launch = {"server_name": "127.0.0.1", "server_port": int(os.environ.get("GRADIO_PORT", "7860")), "share": False}
    accepted = inspect.signature(page.launch).parameters
    page.queue().launch(**{k: v for k, v in launch.items() if k in accepted})


if __name__ == "__main__":
    main()