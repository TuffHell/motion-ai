"""Walking-down-the-street scenario engine.

Produces longer continuous sequences (multi-second) in which a subject
starts walking healthy and then suddenly develops a neurological event
(stroke, spasticity flare, Parkinsonian freeze, ataxic episode). Used
both for the animated visualizations and the step-by-step AI demo.

Public API:
    simulate_walk(profile, rng, n_frames, onset_frame, peak_severity)
        -> raw (T,12), severity_curve (T,), subject dict, phase (T,)
    build_skeleton_frames(raw, subject, phase, dt, walk_forward=True)
        -> list of dicts with full-body joint positions per frame
    generate_scenarios(n=300, seed=42)
        -> list of scenario metadata dicts
"""
from __future__ import annotations
import numpy as np
from gait_sim import PROFILES, N_RAW

# --------------------------------------------------------------------- subject
def _subject(rng: np.random.Generator) -> dict:
    height = rng.uniform(1.55, 1.90)
    cadence = rng.uniform(0.85, 1.25)
    stride = rng.uniform(0.55, 0.80) * height
    arm_swing = rng.uniform(0.25, 0.45) * height
    noise = rng.uniform(0.005, 0.025)
    drift = rng.uniform(0.0, 0.01)
    return dict(height=height, cadence=cadence, stride=stride,
                arm_swing=arm_swing, noise=noise, drift=drift)


# --------------------------------------------------------- clean kinematics
def _kinematics(s, t, phase, profile, severity, rng):
    """Noise-free 12-channel signal for the same subject/phase but a given
    profile + severity. Same biomechanical model as gait_sim.simulate_window,
    factored out so we can crossfade healthy<->pathology smoothly."""
    h = s["height"]
    l_arm_y = s["arm_swing"] * np.sin(phase + np.pi)
    r_arm_y = s["arm_swing"] * np.sin(phase)
    l_arm_x = -0.20 * h + 0.02 * np.sin(2 * phase)
    r_arm_x = +0.20 * h + 0.02 * np.sin(2 * phase)
    l_arm_z = 0.55 * h + 0.03 * np.sin(2 * phase)
    r_arm_z = 0.55 * h + 0.03 * np.sin(2 * phase + np.pi)

    l_leg_y = 0.5 * s["stride"] * np.sin(phase)
    r_leg_y = 0.5 * s["stride"] * np.sin(phase + np.pi)
    l_leg_x = -0.10 * h + 0.01 * np.sin(phase)
    r_leg_x = +0.10 * h + 0.01 * np.sin(phase + np.pi)
    l_leg_z = 0.05 + np.maximum(0, 0.12 * np.cos(phase))
    r_leg_z = 0.05 + np.maximum(0, 0.12 * np.cos(phase + np.pi))

    sev = severity
    if profile == "hemi_left":
        l_arm_y = l_arm_y * (1 - 0.85 * sev)
        l_arm_x = l_arm_x + 0.10 * sev
        l_arm_z = l_arm_z + 0.08 * sev
        l_leg_z = 0.05 + np.maximum(0, (0.12 - 0.10 * sev) * np.cos(phase))
        l_leg_y = l_leg_y * (1 - 0.4 * sev)
    elif profile == "hemi_right":
        r_arm_y = r_arm_y * (1 - 0.85 * sev)
        r_arm_x = r_arm_x - 0.10 * sev
        r_arm_z = r_arm_z + 0.08 * sev
        r_leg_z = 0.05 + np.maximum(0, (0.12 - 0.10 * sev) * np.cos(phase + np.pi))
        r_leg_y = r_leg_y * (1 - 0.4 * sev)
    elif profile == "spastic":
        side = s.get("spastic_side", "L")
        if side == "L":
            l_arm_y = l_arm_y * (1 - 0.6 * sev); l_arm_x = l_arm_x + 0.06 * sev
            l_leg_x = l_leg_x - 0.08 * sev * np.abs(np.sin(phase))
            l_leg_y = l_leg_y * (1 - 0.3 * sev)
        else:
            r_arm_y = r_arm_y * (1 - 0.6 * sev); r_arm_x = r_arm_x - 0.06 * sev
            r_leg_x = r_leg_x + 0.08 * sev * np.abs(np.sin(phase + np.pi))
            r_leg_y = r_leg_y * (1 - 0.3 * sev)
    elif profile == "parkinson":
        scale = 1 - 0.65 * sev
        l_arm_y = l_arm_y * scale; r_arm_y = r_arm_y * scale
        l_leg_y = l_leg_y * scale; r_leg_y = r_leg_y * scale
        l_leg_z = 0.05 + np.maximum(0, 0.04 * np.cos(phase))
        r_leg_z = 0.05 + np.maximum(0, 0.04 * np.cos(phase + np.pi))
        tremor_phase = s.get("tremor_phase", 0.0)
        tremor = 0.015 * sev * np.sin(2 * np.pi * 5.5 * t + tremor_phase)
        l_arm_x = l_arm_x + tremor; r_arm_x = r_arm_x + tremor
    elif profile == "ataxic":
        jitter = s.get("ataxic_jitter")
        if jitter is None:
            jitter = rng.normal(0, 1.0, len(t))
            s["ataxic_jitter"] = jitter
        j2 = s.get("ataxic_j2", rng.normal(0, 1.0, len(t)))
        j3 = s.get("ataxic_j3", rng.normal(0, 1.0, len(t)))
        j4 = s.get("ataxic_j4", rng.normal(0, 1.0, len(t)))
        s.setdefault("ataxic_j2", j2); s.setdefault("ataxic_j3", j3); s.setdefault("ataxic_j4", j4)
        l_arm_x = l_arm_x + 0.08 * sev * jitter
        r_arm_x = r_arm_x - 0.08 * sev * jitter
        l_leg_y = l_leg_y + 0.05 * sev * j2
        r_leg_y = r_leg_y + 0.05 * sev * j3
        l_leg_x = l_leg_x + 0.04 * sev * j4
        r_leg_x = r_leg_x + 0.04 * sev * jitter

    return np.stack([l_arm_x, l_arm_y, l_arm_z,
                     r_arm_x, r_arm_y, r_arm_z,
                     l_leg_x, l_leg_y, l_leg_z,
                     r_leg_x, r_leg_y, r_leg_z], axis=1)


# --------------------------------------------------------------- simulate_walk
def simulate_walk(profile: str, rng: np.random.Generator,
                  n_frames: int = 80, dt: float = 0.075,
                  onset_frame: int | None = None,
                  peak_severity: float = 0.85,
                  ramp_frames: int = 10):
    """Continuous walking sequence with optional pathology onset.

    Returns (raw, sev_curve, subject, phase)
      raw         (n_frames, 12) noisy sensor signal
      sev_curve   (n_frames,)    instantaneous severity in [0, peak]
      subject     dict           anthropometrics of the simulated subject
      phase       (n_frames,)    gait phase in radians
    """
    s = _subject(rng)
    if profile == "spastic":
        s["spastic_side"] = "L" if rng.random() < 0.5 else "R"
    if profile == "parkinson":
        s["tremor_phase"] = rng.uniform(0, 2 * np.pi)

    t = np.arange(n_frames) * dt
    phase = 2 * np.pi * s["cadence"] * t + rng.uniform(0, 2 * np.pi)

    healthy = _kinematics(s, t, phase, "healthy", 0.0, rng)
    if profile == "healthy" or onset_frame is None or onset_frame >= n_frames:
        raw = healthy.copy()
        sev_curve = np.zeros(n_frames)
    else:
        sick = _kinematics(s, t, phase, profile, peak_severity, rng)
        sev_curve = np.zeros(n_frames)
        ramp_end = min(onset_frame + ramp_frames, n_frames)
        ramp = np.linspace(0, 1, ramp_end - onset_frame, endpoint=True)
        sev_curve[onset_frame:ramp_end] = ramp
        sev_curve[ramp_end:] = 1.0
        w = sev_curve[:, None]
        raw = (1 - w) * healthy + w * sick
        sev_curve = sev_curve * peak_severity

    raw = raw + rng.normal(0, s["noise"], raw.shape)
    raw = raw + np.linspace(0, s["drift"], n_frames)[:, None] * rng.normal(0, 1, (1, N_RAW))
    return raw, sev_curve, s, phase


# --------------------------------------------------------- full skeleton
def build_skeleton_frames(raw, subject, phase, dt=0.075,
                          walk_forward=True):
    """Derive full-body joint positions for each frame.

    Returns list[dict] with keys: head, neck, pelvis, l_shoulder, r_shoulder,
    l_hip, r_hip, l_elbow, r_elbow, l_wrist, r_wrist, l_knee, r_knee,
    l_ankle, r_ankle, pelvis_y (forward translation), bounce.
    """
    n_frames = raw.shape[0]
    h = subject["height"]
    cadence = subject["cadence"]
    walk_speed = cadence * subject["stride"]  # m/s
    out = []
    # Standard anthropometric proportions (Drillis & Contini, simplified):
    #   ankle 0.04h  knee 0.28h  hip 0.55h  shoulder 0.82h  neck 0.87h  head 0.97h
    for i in range(n_frames):
        py = walk_speed * i * dt if walk_forward else 0.0
        bounce = 0.020 * np.sin(2 * phase[i])  # vertical bob, double cadence

        pelvis = np.array([0.0, py,             0.55 * h + bounce])
        neck   = np.array([0.0, py,             0.87 * h + bounce])
        head   = np.array([0.0, py + 0.02 * h,  0.97 * h + bounce])
        l_sh   = np.array([-0.12 * h, py,       0.82 * h + bounce])
        r_sh   = np.array([+0.12 * h, py,       0.82 * h + bounce])
        l_hip  = np.array([-0.09 * h, py,       0.55 * h + bounce])
        r_hip  = np.array([+0.09 * h, py,       0.55 * h + bounce])

        # Sensors: raw x and z are absolute (ground-frame); y is relative to
        # the subject's pelvis (forward swing), so add the walking translation.
        # Wrists bounce with the torso; ankles touch the ground (no bounce).
        l_wrist = np.array([raw[i, 0], py + raw[i, 1], raw[i, 2] + bounce])
        r_wrist = np.array([raw[i, 3], py + raw[i, 4], raw[i, 5] + bounce])
        l_ankle = np.array([raw[i, 6], py + raw[i, 7], raw[i, 8]])
        r_ankle = np.array([raw[i, 9], py + raw[i, 10], raw[i, 11]])

        def _elbow(sh, wr, sign):
            # Elbow ≈ midpoint(shoulder, wrist), tucked slightly toward midline
            # and dropped a bit (natural arm flexion).
            mid = 0.5 * (sh + wr)
            return mid + np.array([sign * -0.025 * h, 0.025 * h, -0.025 * h])
        l_elbow = _elbow(l_sh, l_wrist, -1)
        r_elbow = _elbow(r_sh, r_wrist, +1)

        def _knee(hp, an):
            mid = 0.5 * (hp + an)
            return mid + np.array([0.0, 0.03 * h, 0.0])  # knee slightly anterior
        l_knee = _knee(l_hip, l_ankle)
        r_knee = _knee(r_hip, r_ankle)

        out.append(dict(head=head, neck=neck, pelvis=pelvis,
                        l_shoulder=l_sh, r_shoulder=r_sh,
                        l_hip=l_hip, r_hip=r_hip,
                        l_elbow=l_elbow, r_elbow=r_elbow,
                        l_wrist=l_wrist, r_wrist=r_wrist,
                        l_knee=l_knee, r_knee=r_knee,
                        l_ankle=l_ankle, r_ankle=r_ankle,
                        pelvis_y=py, bounce=bounce))
    return out


# ---------------------------------------------------------- 300 scenarios
def generate_scenarios(n: int = 300, seed: int = 42, n_frames: int = 80):
    """Deterministic metadata list for a gallery of n scenarios.

    Composition: ~15% healthy (controls), ~85% pathology, balanced across
    the 5 non-healthy profiles. Onset is sampled in the first half of the
    walk so the AI has time to react before the clip ends.
    """
    rng = np.random.default_rng(seed)
    out = []
    pathology = [p for p in PROFILES if p != "healthy"]
    for i in range(n):
        is_healthy = rng.random() < 0.15
        if is_healthy:
            profile = "healthy"
            onset = None
            severity = 0.0
        else:
            profile = pathology[i % len(pathology)]
            onset = int(rng.integers(18, max(19, n_frames // 2)))
            severity = float(rng.uniform(0.55, 0.95))
        out.append(dict(
            id=i, profile=profile,
            severity=severity,
            onset_frame=onset,
            onset_seconds=None if onset is None else round(onset * 0.075, 2),
            subject_seed=int(rng.integers(0, 1_000_000)),
        ))
    return out


# --------------------------------------------------- live AI inference
def sliding_window_inference(raw, scaler, model, build_features_fn,
                             window: int = 20, stride: int = 1):
    """Run the binary screener over every sliding window of length `window`.

    Returns (window_centers, probabilities).
    """
    n = raw.shape[0]
    if n < window:
        return np.array([]), np.array([])
    feats = build_features_fn(raw)
    centers, probs = [], []
    batch_inputs = []
    batch_centers = []
    for start in range(0, n - window + 1, stride):
        chunk = feats[start:start + window]
        scaled = scaler.transform(chunk).reshape(1, window, -1)
        batch_inputs.append(scaled[0])
        batch_centers.append(start + window // 2)
    batch = np.stack(batch_inputs, axis=0)
    p = model.predict(batch, verbose=0).ravel()
    return np.asarray(batch_centers), p


def compute_saliency(scaled_window, model):
    """Gradient saliency for a single (1, 20, 24) window. Returns
    (per_timestep_24_saliency, per_sensor_saliency_4) — both normalized [0,1]."""
    import tensorflow as tf
    x = tf.convert_to_tensor(scaled_window, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x, training=False)
    g = tape.gradient(y, x).numpy()[0]                       # (20, 24)
    sal = np.abs(g)
    sal_norm = sal / (sal.max() + 1e-9)
    # 24 feature channels split into 4 sensors:
    #   ch 0-2  l_wrist xyz, 3-5 r_wrist, 6-8 l_ankle, 9-11 r_ankle
    #   ch 12-14 arm_drift(L-R wrist), 15-17 leg_drift(L-R ankle)
    #   ch 18 l_wrist mag, 19 r_wrist mag, 20 l_ankle mag, 21 r_ankle mag
    #   ch 22 upper asymmetry  ch 23 lower asymmetry
    per_sensor = np.zeros(4)
    weights = sal.sum(0)  # importance per channel summed over timesteps
    per_sensor[0] = weights[0:3].sum() + 0.5 * weights[12:15].sum() + weights[18] + 0.5 * weights[22]
    per_sensor[1] = weights[3:6].sum() + 0.5 * weights[12:15].sum() + weights[19] + 0.5 * weights[22]
    per_sensor[2] = weights[6:9].sum() + 0.5 * weights[15:18].sum() + weights[20] + 0.5 * weights[23]
    per_sensor[3] = weights[9:12].sum() + 0.5 * weights[15:18].sum() + weights[21] + 0.5 * weights[23]
    per_sensor = per_sensor / (per_sensor.max() + 1e-9)
    return sal_norm, per_sensor


def simulate_custom(
    arm_swing_L: float = 0.40,
    arm_swing_R: float = 0.40,
    leg_swing_L: float = 0.55,
    leg_swing_R: float = 0.55,
    foot_clear_L: float = 0.12,
    foot_clear_R: float = 0.12,
    tremor_amp: float = 0.0,
    lateral_jitter: float = 0.0,
    cadence_hz: float = 1.0,
    stride_m: float = 1.35,
    noise_sigma: float = 0.012,
    n_frames: int = 80,
    dt: float = 0.075,
    seed: int = 0,
):
    """Hand-crafted IMU stream for the AI sandbox.

    Each parameter is an independent biomechanical knob — the dashboard's
    'Test the AI' tab uses this to probe the model with bespoke deficits
    (e.g. "what if only the left arm swing is suppressed, but nothing else?").
    Returns (raw (T,12), subject_dict, phase (T,)) compatible with the
    skeleton renderer.
    """
    rng = np.random.default_rng(seed)
    h = 1.75  # fixed virtual-subject height for the sandbox
    t = np.arange(n_frames) * dt
    phase = 2 * np.pi * cadence_hz * t

    # X / Z components mirror gait_sim._kinematics (height-scaled) so that
    # the AI sees the same distribution it was trained on. The user-tunable
    # arm/leg swing knobs drive the Y components only.
    l_arm_x = -0.20 * h + 0.02 * np.sin(2 * phase)
    l_arm_y = arm_swing_L * np.sin(phase + np.pi)
    l_arm_z = 0.55 * h + 0.03 * np.sin(2 * phase)
    r_arm_x = +0.20 * h + 0.02 * np.sin(2 * phase)
    r_arm_y = arm_swing_R * np.sin(phase)
    r_arm_z = 0.55 * h + 0.03 * np.sin(2 * phase + np.pi)
    if tremor_amp > 0:
        tremor = tremor_amp * np.sin(2 * np.pi * 5.5 * t)
        l_arm_x = l_arm_x + tremor
        r_arm_x = r_arm_x + tremor
    if lateral_jitter > 0:
        l_arm_x = l_arm_x + rng.normal(0, lateral_jitter, n_frames)
        r_arm_x = r_arm_x - rng.normal(0, lateral_jitter, n_frames)

    l_leg_x = -0.10 * h + 0.01 * np.sin(phase)
    l_leg_y = leg_swing_L * np.sin(phase)
    l_leg_z = 0.05 + np.maximum(0, foot_clear_L * np.cos(phase))
    r_leg_x = +0.10 * h + 0.01 * np.sin(phase + np.pi)
    r_leg_y = leg_swing_R * np.sin(phase + np.pi)
    r_leg_z = 0.05 + np.maximum(0, foot_clear_R * np.cos(phase + np.pi))

    raw = np.stack([l_arm_x, l_arm_y, l_arm_z,
                    r_arm_x, r_arm_y, r_arm_z,
                    l_leg_x, l_leg_y, l_leg_z,
                    r_leg_x, r_leg_y, r_leg_z], axis=1)
    raw = raw + rng.normal(0, noise_sigma, raw.shape)

    subject = dict(height=1.75, cadence=cadence_hz, stride=stride_m,
                    arm_swing=max(arm_swing_L, arm_swing_R) / 1.75 + 0.2,
                    noise=noise_sigma, drift=0.0)
    return raw, subject, phase


def compute_clinical_metrics(raw, subject=None, dt: float = 0.075,
                              start_frame: int = 0):
    """Validated spatiotemporal + symmetry metrics from the 4-IMU stream.

    Returns a dict with cadence (steps/min), walking speed (m/s),
    range-of-motion per limb, Robinson Symmetry Index (%) per metric,
    and a composite asymmetry score.

    `start_frame` lets callers skip a pre-event healthy prefix so ROM /
    symmetry indices reflect the *current* gait state, not the whole clip.
    """
    from scipy.signal import find_peaks
    raw = raw[start_frame:]
    n = raw.shape[0]
    duration = n * dt
    l_ankle_z, r_ankle_z = raw[:, 8], raw[:, 11]
    l_wrist_y, r_wrist_y = raw[:, 1], raw[:, 4]
    l_ankle_y, r_ankle_y = raw[:, 7], raw[:, 10]

    # Step events: ankle-z peaks (foot lifted = swing phase apex).
    # `distance=8` ≈ 0.6 s — bars two-peaks-per-stride double-counting at the
    # typical 1 Hz cadence; `height=0.08` filters out IMU noise wobble.
    l_peaks, _ = find_peaks(l_ankle_z, height=0.08, distance=8)
    r_peaks, _ = find_peaks(r_ankle_z, height=0.08, distance=8)
    n_steps = len(l_peaks) + len(r_peaks)
    cadence = (n_steps / duration) * 60 if duration > 0 else 0.0

    stride_length = float(subject["stride"]) if subject is not None else 0.7
    walking_speed = (cadence / 60.0) * stride_length / 2.0

    l_arm_rom = float(np.ptp(l_wrist_y))
    r_arm_rom = float(np.ptp(r_wrist_y))
    l_leg_rom = float(np.ptp(l_ankle_y))
    r_leg_rom = float(np.ptp(r_ankle_y))
    l_clear = float(np.ptp(l_ankle_z))
    r_clear = float(np.ptp(r_ankle_z))

    def SI(a, b):  # Robinson 1987 Symmetry Index
        m = 0.5 * (a + b)
        return 100.0 * abs(a - b) / m if m > 1e-6 else 0.0

    arm_sym = SI(l_arm_rom, r_arm_rom)
    leg_sym = SI(l_leg_rom, r_leg_rom)
    clear_sym = SI(l_clear, r_clear)
    composite = float(np.mean([arm_sym, leg_sym, clear_sym]))

    return dict(
        duration_s=duration,
        cadence_spm=float(cadence),
        stride_length_m=stride_length,
        walking_speed_mps=float(walking_speed),
        arm_rom_L=l_arm_rom, arm_rom_R=r_arm_rom,
        leg_rom_L=l_leg_rom, leg_rom_R=r_leg_rom,
        foot_clearance_L=l_clear, foot_clearance_R=r_clear,
        symmetry_arms_pct=float(arm_sym),
        symmetry_legs_pct=float(leg_sym),
        symmetry_clearance_pct=float(clear_sym),
        composite_asymmetry_pct=composite,
        n_l_steps=int(len(l_peaks)), n_r_steps=int(len(r_peaks)),
    )


def alert_frame(probs, centers, threshold: float = 0.5, debounce: int = 3):
    """First window-center index where probability stays above threshold
    for `debounce` consecutive windows. Returns None if it never triggers."""
    streak = 0
    for i, p in enumerate(probs):
        if p > threshold:
            streak += 1
            if streak >= debounce:
                return int(centers[i])
        else:
            streak = 0
    return None
