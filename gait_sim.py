"""Realistic 3D gait kinematics simulator.

Produces a (T, 12) array of XYZ positions for 4 IMU sensors:
  channels  0-2  : left  wrist  (x, y, z)
  channels  3-5  : right wrist
  channels  6-8  : left  ankle
  channels  9-11 : right ankle

Subjects vary in height, cadence, stride length, and sensor noise.
Pathology profiles modify amplitude / phase / drift of the affected limbs
in ways that mirror published kinematic signatures:

  healthy    - symmetric in-phase arm/leg swing
  hemi_left  - hypokinetic left arm + left foot drop (reduced toe clearance)
  hemi_right - hypokinetic right arm + right foot drop
  spastic    - circumduction + reduced range on one side, picked at random
  parkinson  - bradykinesia (low amplitude bilateral) + tremor (6 Hz)
  ataxic     - high lateral variability, irregular phase

`build_features` turns the 12-channel raw window into the 24-channel
biomarker vector the model is trained on.
"""
from __future__ import annotations
import numpy as np

PROFILES = ("healthy", "hemi_left", "hemi_right", "spastic", "parkinson", "ataxic")
LABEL_BINARY = {"healthy": 0, "hemi_left": 1, "hemi_right": 1,
                "spastic": 1, "parkinson": 1, "ataxic": 1}
N_RAW = 12
N_FEAT = 24
WINDOW = 20  # timesteps per model input


def _subject(rng: np.random.Generator) -> dict:
    """Sample a virtual subject's anthropometrics + sensor characteristics."""
    height = rng.uniform(1.55, 1.90)               # m
    cadence = rng.uniform(0.85, 1.25)              # gait cycles per second
    stride = rng.uniform(0.55, 0.80) * height      # m
    arm_swing = rng.uniform(0.25, 0.45) * height
    noise = rng.uniform(0.005, 0.025)              # IMU noise sigma (m)
    drift = rng.uniform(0.0, 0.01)                 # slow sensor drift
    return dict(height=height, cadence=cadence, stride=stride,
                arm_swing=arm_swing, noise=noise, drift=drift)


def simulate_window(profile: str, rng: np.random.Generator,
                    timesteps: int = WINDOW, dt: float = 0.075,
                    severity: float | None = None) -> np.ndarray:
    """Return one (timesteps, 12) raw-sensor window for the given profile.

    `dt` ~ 0.075 s ⇒ 20 timesteps ≈ 1.5 s, about one full gait cycle.
    `severity` in [0,1]; defaults to random per call.
    """
    s = _subject(rng)
    if severity is None:
        severity = rng.uniform(0.4, 1.0) if profile != "healthy" else 0.0

    t = np.arange(timesteps) * dt
    phase = 2 * np.pi * s["cadence"] * t + rng.uniform(0, 2 * np.pi)

    # --- baseline symmetric kinematics --------------------------------------
    # Wrists swing antiphase to ipsilateral leg (normal counter-rotation).
    l_arm_y = s["arm_swing"] * np.sin(phase + np.pi)
    r_arm_y = s["arm_swing"] * np.sin(phase)
    l_arm_x = -0.20 * s["height"] + 0.02 * np.sin(2 * phase)
    r_arm_x = +0.20 * s["height"] + 0.02 * np.sin(2 * phase)
    l_arm_z = 0.55 * s["height"] + 0.03 * np.sin(2 * phase)
    r_arm_z = 0.55 * s["height"] + 0.03 * np.sin(2 * phase + np.pi)

    l_leg_y = 0.5 * s["stride"] * np.sin(phase)
    r_leg_y = 0.5 * s["stride"] * np.sin(phase + np.pi)
    l_leg_x = -0.10 * s["height"] + 0.01 * np.sin(phase)
    r_leg_x = +0.10 * s["height"] + 0.01 * np.sin(phase + np.pi)
    # foot clearance: half-rectified cosine — foot lifts during swing phase
    l_leg_z = 0.05 + np.maximum(0, 0.12 * np.cos(phase))
    r_leg_z = 0.05 + np.maximum(0, 0.12 * np.cos(phase + np.pi))

    # --- pathology modifiers -------------------------------------------------
    sev = severity
    if profile == "hemi_left":
        l_arm_y *= (1 - 0.85 * sev)                # flexor synergy, reduced swing
        l_arm_x += 0.10 * sev                       # arm held closer to body
        l_arm_z += 0.08 * sev                       # elbow flexion lifts wrist
        l_leg_z = 0.05 + np.maximum(0, (0.12 - 0.10 * sev) * np.cos(phase))  # foot drop
        l_leg_y *= (1 - 0.4 * sev)                  # shorter step
    elif profile == "hemi_right":
        r_arm_y *= (1 - 0.85 * sev)
        r_arm_x -= 0.10 * sev
        r_arm_z += 0.08 * sev
        r_leg_z = 0.05 + np.maximum(0, (0.12 - 0.10 * sev) * np.cos(phase + np.pi))
        r_leg_y *= (1 - 0.4 * sev)
    elif profile == "spastic":
        side = rng.choice(("L", "R"))
        if side == "L":
            l_arm_y *= (1 - 0.6 * sev); l_arm_x += 0.06 * sev
            l_leg_x -= 0.08 * sev * np.abs(np.sin(phase))  # circumduction outward
            l_leg_y *= (1 - 0.3 * sev)
        else:
            r_arm_y *= (1 - 0.6 * sev); r_arm_x -= 0.06 * sev
            r_leg_x += 0.08 * sev * np.abs(np.sin(phase + np.pi))
            r_leg_y *= (1 - 0.3 * sev)
    elif profile == "parkinson":
        # bilateral bradykinesia: globally reduced amplitudes
        scale = 1 - 0.65 * sev
        l_arm_y *= scale; r_arm_y *= scale
        l_leg_y *= scale; r_leg_y *= scale
        l_leg_z = 0.05 + np.maximum(0, 0.04 * np.cos(phase))   # shuffling
        r_leg_z = 0.05 + np.maximum(0, 0.04 * np.cos(phase + np.pi))
        # ~5-6 Hz pill-rolling tremor on the wrists
        tremor = 0.015 * sev * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2*np.pi))
        l_arm_x += tremor; r_arm_x += tremor
    elif profile == "ataxic":
        # high lateral variability, irregular phase
        jitter = rng.normal(0, 0.08 * sev, timesteps)
        l_arm_x += jitter; r_arm_x -= jitter
        l_leg_y += rng.normal(0, 0.05 * sev, timesteps)
        r_leg_y += rng.normal(0, 0.05 * sev, timesteps)
        l_leg_x += rng.normal(0, 0.04 * sev, timesteps)
        r_leg_x += rng.normal(0, 0.04 * sev, timesteps)

    # --- sensor model: gaussian noise + slow drift ---------------------------
    raw = np.stack([l_arm_x, l_arm_y, l_arm_z,
                    r_arm_x, r_arm_y, r_arm_z,
                    l_leg_x, l_leg_y, l_leg_z,
                    r_leg_x, r_leg_y, r_leg_z], axis=1)
    raw += rng.normal(0, s["noise"], raw.shape)
    raw += np.linspace(0, s["drift"], timesteps)[:, None] * rng.normal(0, 1, (1, N_RAW))
    return raw


def build_features(raw: np.ndarray) -> np.ndarray:
    """Map (T, 12) raw sensor data to (T, 24) clinical biomarker channels."""
    l_arm, r_arm = raw[:, 0:3], raw[:, 3:6]
    l_leg, r_leg = raw[:, 6:9], raw[:, 9:12]
    arm_drift = l_arm - r_arm
    leg_drift = l_leg - r_leg
    la = np.linalg.norm(l_arm, axis=1, keepdims=True)
    ra = np.linalg.norm(r_arm, axis=1, keepdims=True)
    ll = np.linalg.norm(l_leg, axis=1, keepdims=True)
    rl = np.linalg.norm(r_leg, axis=1, keepdims=True)
    arm_asym = (la - ra) / (la + ra + 1e-6)
    leg_asym = (ll - rl) / (ll + rl + 1e-6)
    return np.hstack((l_arm, r_arm, l_leg, r_leg, arm_drift, leg_drift,
                      la, ra, ll, rl, arm_asym, leg_asym))


def make_dataset(n_per_class: int, seed: int = 0):
    """Generate a balanced multi-class dataset of (windows, raw, labels)."""
    rng = np.random.default_rng(seed)
    X, Xraw, y_bin, y_multi = [], [], [], []
    for cls_idx, prof in enumerate(PROFILES):
        for _ in range(n_per_class):
            raw = simulate_window(prof, rng)
            X.append(build_features(raw))
            Xraw.append(raw)
            y_bin.append(LABEL_BINARY[prof])
            y_multi.append(cls_idx)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(Xraw, dtype=np.float32),
            np.asarray(y_bin, dtype=np.int32),
            np.asarray(y_multi, dtype=np.int32))
