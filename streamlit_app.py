"""Motion-AI — clinical-grade gait pathology dashboard.

Tabs:
  🩺 Diagnosis           — animated patient + healthy digital twin
  📈 Biomarkers          — 4-sensor traces + 24-channel input heatmap
  🚶 Clinical scenarios  — 300 pedestrians, real-time AI alert, clinical
                           report card, neurons-saved counter, cohort retrieval
  🧬 Architecture
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import hashlib
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from tensorflow.keras.models import load_model, Model

from gait_sim import build_features, PROFILES, WINDOW
from scenarios import (simulate_walk, build_skeleton_frames,
                       generate_scenarios, sliding_window_inference,
                       alert_frame, compute_saliency,
                       compute_clinical_metrics, simulate_custom)

# ---------------------------------------------------------------- constants
PROFILE_LABELS = {
    "healthy":    "Healthy symmetric gait",
    "hemi_left":  "Left hemiparesis (post-stroke)",
    "hemi_right": "Right hemiparesis (post-stroke)",
    "spastic":    "Spastic circumduction gait",
    "parkinson":  "Parkinsonian bradykinesia + tremor",
    "ataxic":     "Cerebellar ataxia",
}
PROFILE_EMOJI = {"healthy": "🟢", "hemi_left": "🟥", "hemi_right": "🟥",
                 "spastic": "🟧", "parkinson": "🟪", "ataxic": "🟨"}

# Saver JL, "Time is brain — quantified", Stroke 2006: ~1.9M neurons / min,
# i.e. ~32,000 neurons per second of untreated ischemic stroke.
NEURONS_PER_SECOND_LOST = 32_000
# Time it usually takes a bystander to notice + react to a sudden gait event
# (informed by witnessed-stroke survey data — conservative).
BYSTANDER_REACTION_SEC = 30.0

# ---------------------------------------------------------------- page
st.set_page_config(page_title="Motion-AI Clinical", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#05070d 0%,#0b1020 100%); }
      .clin-card { background:#0f1729; border:1px solid #1f2937;
                   border-radius:8px; padding:10px 14px; margin-bottom:8px; }
      .clin-card h4 { color:#f3f4f6; margin:0 0 6px 0;
                      font-size:12px; letter-spacing:.5px;
                      text-transform:uppercase; }
      .clin-row { display:flex; justify-content:space-between; align-items:center;
                  font-size:13px; color:#d1d5db; padding:3px 0;
                  border-bottom:1px dashed #1f2937; }
      .clin-row:last-child { border:none; }
      .clin-val { font-weight:600; font-variant-numeric:tabular-nums; }
      .flag-ok   { color:#22c55e; }
      .flag-warn { color:#fbbf24; }
      .flag-bad  { color:#ef4444; }
      .neurons-card { background:linear-gradient(135deg,#7c3aed 0%,#ec4899 100%);
                      border-radius:14px; padding:14px 18px; color:white;
                      box-shadow:0 0 32px rgba(124,58,237,0.35);
                      margin-bottom:8px; }
      .neurons-card .big { font-size:30px; font-weight:800;
                           letter-spacing:-.5px; font-variant-numeric:tabular-nums;}
      .neurons-card .sub { font-size:12px; opacity:.85; line-height:1.4; }
      .scen-pill { display:inline-block; padding:2px 9px; border-radius:10px;
                   background:#1f2937; color:#e5e7eb; font-size:12px;
                   margin-right:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🧠 Motion-AI — Clinical Gait Pathology Detector")
st.caption("Quad-IMU wearables · 1.5 s rolling CNN-BiLSTM · binary screener + "
           "6-class differential head · digital twin · gradient saliency · "
           "cohort retrieval · time-is-brain accounting")

# ---------------------------------------------------------------- assets
@st.cache_resource
def load_assets():
    bin_m = load_model("stroke_model_v2.h5", compile=False)
    mc_m  = load_model("motion_multiclass.h5", compile=False)
    scaler = joblib.load("scaler_v2.save")
    emb_m = Model(bin_m.input, bin_m.layers[-2].output)  # Dense(32) layer
    return bin_m, mc_m, scaler, emb_m

try:
    bin_model, mc_model, scaler, emb_model = load_assets()
except Exception as e:
    st.error(f"Model load failed: {e}"); st.stop()

SCENARIOS = generate_scenarios(n=300, seed=42, n_frames=80)


@st.cache_resource(show_spinner="Precomputing 300-pedestrian cohort embeddings...")
def precompute_cohort(_scenarios):
    """Embed every scenario's final 1.5 s window into the Dense(32) space."""
    X = np.zeros((len(_scenarios), WINDOW, 24), dtype=np.float32)
    for i, s in enumerate(_scenarios):
        rng = np.random.default_rng(s["subject_seed"])
        raw, _, _, _ = simulate_walk(s["profile"], rng, n_frames=80,
                                      onset_frame=s["onset_frame"],
                                      peak_severity=s["severity"])
        X[i] = scaler.transform(build_features(raw[-WINDOW:]))
    return emb_model.predict(X, verbose=0)

COHORT_EMB = precompute_cohort(SCENARIOS)


# ---------------------------------------------------------------- helpers
BONES = [("head","neck"),("neck","pelvis"),
         ("neck","l_shoulder"),("neck","r_shoulder"),
         ("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
         ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
         ("pelvis","l_hip"),("pelvis","r_hip"),
         ("l_hip","l_knee"),("l_knee","l_ankle"),
         ("r_hip","r_knee"),("r_knee","r_ankle")]
SENSOR_KEYS = ("l_wrist","r_wrist","l_ankle","r_ankle")
SENSOR_LABELS = ("L-wrist","R-wrist","L-ankle","R-ankle")


def _bone_xyz(joints, xoff=0.0):
    xs, ys, zs = [], [], []
    for a, b in BONES:
        xs += [joints[a][0]+xoff, joints[b][0]+xoff, None]
        ys += [joints[a][1],      joints[b][1],      None]
        zs += [joints[a][2],      joints[b][2],      None]
    return xs, ys, zs


def _color_anom(l):
    return "#22c55e" if l < 0.25 else "#f59e0b" if l < 0.55 else "#ef4444"


def _quad(x0, x1, y0, y1, z, color, opacity=1.0):
    return go.Mesh3d(x=[x0,x1,x1,x0], y=[y0,y0,y1,y1], z=[z]*4,
        i=[0,0], j=[1,2], k=[2,3], color=color, opacity=opacity,
        hoverinfo="skip", showlegend=False, flatshading=True,
        lighting=dict(ambient=0.85, diffuse=0.4))


def per_sensor_anomaly(raw):
    n = raw.shape[0]; b = min(20, n)
    base = raw[:b].mean(0); std = raw[:b].std(0) + 0.02
    z = np.abs(raw - base) / std
    return 1 - np.exp(-np.stack([z[:, 0:3].mean(1), z[:, 3:6].mean(1),
                                  z[:, 6:9].mean(1), z[:, 9:12].mean(1)],
                                 axis=1) / 4.0)


def _flag(v, low, high, prefer="middle"):
    """Color flag from normative band. Returns (emoji, css_class, label)."""
    if prefer == "low":
        if v < low:  return ("🟢", "flag-ok",   "normal")
        if v < high: return ("🟡", "flag-warn", "borderline")
        return ("🔴", "flag-bad", "abnormal")
    if low <= v <= high: return ("🟢", "flag-ok",   "normal")
    span = max(high - low, 0.1)
    if v < low - span or v > high + span: return ("🔴", "flag-bad", "abnormal")
    return ("🟡", "flag-warn", "borderline")


# ===================================================================
#               CINEMATIC 3D SCENE  (patient + digital twin)
# ===================================================================
PATIENT_X = 0.0
TWIN_X    = -1.8  # twin walks a parallel lane to the patient's left


def _humanoid_traces(j, xoff, body_color, halo_color, alpha=1.0):
    """Return a list of Plotly traces that draw a volumetric humanoid body.

    Components: 3D torso box (Mesh3d), spherical head (big marker),
    thick limb halo+core (Scatter3d lines), joint balls (markers).
    Pelvis is the foot of the torso; head is its head.
    """
    # ----- TORSO box -----
    DEPTH = 0.13
    sl, sr = j["l_shoulder"].copy(), j["r_shoulder"].copy()
    hl, hr = j["l_hip"].copy(),      j["r_hip"].copy()
    for v in (sl, sr, hl, hr):
        v[0] += xoff
    # 8 vertices: 0-3 front face (sl, sr, hr, hl), 4-7 back face
    back_off = np.array([0.0, -DEPTH, 0.0])
    verts = [sl, sr, hr, hl,
             sl + back_off, sr + back_off, hr + back_off, hl + back_off]
    vx = [v[0] for v in verts]
    vy = [v[1] for v in verts]
    vz = [v[2] for v in verts]
    # Triangulation: 6 faces × 2 triangles
    i_idx = [0, 0,  4, 4,  0, 0,  1, 1,  2, 2,  3, 3]
    j_idx = [1, 2,  5, 6,  1, 5,  2, 6,  3, 7,  0, 4]
    k_idx = [2, 3,  6, 7,  5, 4,  6, 5,  7, 6,  4, 7]
    torso = go.Mesh3d(
        x=vx, y=vy, z=vz, i=i_idx, j=j_idx, k=k_idx,
        color=body_color, opacity=alpha * 0.92,
        flatshading=True, lighting=dict(ambient=0.7, diffuse=0.5),
        hoverinfo="skip", showlegend=False)

    # ----- HEAD sphere (big filled marker) -----
    hp = j["head"].copy(); hp[0] += xoff
    head = go.Scatter3d(
        x=[hp[0]], y=[hp[1]], z=[hp[2]],
        mode="markers",
        marker=dict(size=46, color=body_color, opacity=alpha,
                    line=dict(color=halo_color, width=3)),
        hoverinfo="skip", showlegend=False)

    # ----- LIMBS as thick halo + crisp core lines -----
    LIMBS = [("head", "neck"),
             ("neck", "l_shoulder"), ("neck", "r_shoulder"),
             ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
             ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
             ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
             ("r_hip", "r_knee"), ("r_knee", "r_ankle")]
    xs, ys, zs = [], [], []
    for a, b in LIMBS:
        xs += [j[a][0] + xoff, j[b][0] + xoff, None]
        ys += [j[a][1],         j[b][1],         None]
        zs += [j[a][2],         j[b][2],         None]
    halo = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=halo_color, width=22), opacity=alpha * 0.40,
        hoverinfo="skip", showlegend=False)
    core = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=body_color, width=10), opacity=alpha,
        hoverinfo="skip", showlegend=False)

    # ----- JOINT balls -----
    JK = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
          "l_hip", "r_hip", "l_knee", "r_knee"]
    jp = np.array([j[k] for k in JK])
    jp[:, 0] += xoff
    joints = go.Scatter3d(
        x=jp[:, 0], y=jp[:, 1], z=jp[:, 2], mode="markers",
        marker=dict(size=13, color=body_color, opacity=alpha,
                    line=dict(color=halo_color, width=2)),
        hoverinfo="skip", showlegend=False)

    return [torso, halo, core, head, joints]


def _in_place(skel_list):
    """Subtract per-frame pelvis_y from every joint so the subject 'walks
    in place' (treadmill view) — the standard clinical gait observation
    setup."""
    out = []
    for j in skel_list:
        py = j["pelvis_y"]
        nj = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in j.items()}
        for k, v in nj.items():
            if hasattr(v, "copy") and v.shape == (3,):
                v[1] -= py
        nj["pelvis_y"] = 0.0
        out.append(nj)
    return out


# ===================================================================
#         CINEMATIC 2D SAGITTAL RENDERER  (matplotlib → GIF)
# ===================================================================
# Plotly's 3D scene framing was unreliable; instead we render the gait
# as a proper 2D side-projection animation through matplotlib and serve
# it as a looping GIF. This gives professional-grade composition control:
# layered fills, soft shadows, glow halos, motion blur trails, particle
# pulses on alert. The 2D sagittal projection is also the standard
# clinical gait-analysis camera angle.

_BONE_PAIRS = [
    ("head", "neck"), ("neck", "pelvis"),
    ("neck", "l_shoulder"), ("neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("pelvis", "l_hip"), ("pelvis", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]


def _project(joint, x_off):
    """Project a 3D joint (x, y, z) into 2D sagittal screen coords.

    Slight 3/4 tilt (x weighted by 0.25) gives depth parallax so left
    and right limbs separate visually instead of overlapping perfectly.
    """
    return (joint[1] + x_off + 0.25 * joint[0], joint[2])


def _hash_inputs(*arrays):
    """Stable hash for caching animation bytes."""
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@st.cache_data(show_spinner="🎬 Rendering cinematic gait video…")
def render_cinematic_gif(patient_raw_bytes, twin_raw_bytes,
                          anom_bytes, alert_after, sensor_saliency_bytes,
                          n_frames, hash_key, fps=15):
    """Render the looping cinematic GIF. Cached by hash_key.

    Performance-tuned for Streamlit Cloud's 1 GB CPU-only VM: input frames
    are subsampled to a target of ~30, figure DPI is dropped, and the floor
    reflection is removed — keeps the cinematic look but cuts render time
    from ~20s locally to ~3-5s locally, ~10-15s on Cloud."""
    AMP_Y = 1.6  # forward-back swing exaggeration
    AMP_Z = 1.5  # vertical (foot-lift / wrist-bob) exaggeration
    # decode bytes → arrays
    patient_raw = np.frombuffer(patient_raw_bytes, dtype=np.float64).reshape(n_frames, 12).copy()
    twin_raw    = np.frombuffer(twin_raw_bytes,    dtype=np.float64).reshape(n_frames, 12).copy()
    anom        = np.frombuffer(anom_bytes,        dtype=np.float64).reshape(n_frames, 4)
    sensor_sal  = (np.frombuffer(sensor_saliency_bytes, dtype=np.float64)
                   if sensor_saliency_bytes else None)
    # SUBSAMPLE: cap at 32 rendered frames to keep render time bounded
    TARGET_FRAMES = 32
    if n_frames > TARGET_FRAMES:
        stride = max(1, n_frames // TARGET_FRAMES)
        idx = np.arange(0, n_frames, stride)[:TARGET_FRAMES]
        patient_raw = patient_raw[idx]
        twin_raw = twin_raw[idx]
        anom = anom[idx]
        if alert_after is not None:
            # remap alert frame to subsampled index space
            alert_after = int(np.searchsorted(idx, alert_after))
        n_frames = len(idx)
    # visual amplification: subtract mean, scale, add back. Affects only
    # the rendered position — model already saw the original signal.
    for raw_arr in (patient_raw, twin_raw):
        for ch_y in (1, 4, 7, 10):
            mean = raw_arr[:, ch_y].mean()
            raw_arr[:, ch_y] = mean + AMP_Y * (raw_arr[:, ch_y] - mean)
        for ch_z in (2, 5, 8, 11):
            mean = raw_arr[:, ch_z].mean()
            raw_arr[:, ch_z] = mean + AMP_Z * (raw_arr[:, ch_z] - mean)

    h = 1.75
    # subtle synthetic bounce + body sway driven by the gait cycle for
    # micromovement realism (real walking has ~2 cm head bob and slight
    # lateral hip sway with each step)
    dt_s = 0.075
    cadence_hz = 1.0
    def micro(i):
        t = i * dt_s
        bounce = 0.022 * np.sin(2 * 2 * np.pi * cadence_hz * t)   # double-cadence bob
        sway   = 0.015 * np.sin(2 * np.pi * cadence_hz * t)        # single-cadence sway
        head_nod = 0.012 * np.cos(2 * 2 * np.pi * cadence_hz * t + 0.4)
        return bounce, sway, head_nod
    def skeleton_at(raw_t, i):
        b, s, n = micro(i)
        return dict(
            head=        np.array([0.0 + 0.3*s, 0.02 * h + n,  0.97 * h + b]),
            neck=        np.array([0.0 + 0.5*s, 0.0,           0.87 * h + b]),
            pelvis=      np.array([0.0 + s,     0.0,           0.55 * h + 0.4*b]),
            l_shoulder=  np.array([-0.18 * h + 0.5*s, 0.0,     0.82 * h + b]),
            r_shoulder=  np.array([+0.18 * h + 0.5*s, 0.0,     0.82 * h + b]),
            l_hip=       np.array([-0.10 * h + s, 0.0,         0.55 * h + 0.4*b]),
            r_hip=       np.array([+0.10 * h + s, 0.0,         0.55 * h + 0.4*b]),
            l_wrist=     np.array([raw_t[0],     raw_t[1],     raw_t[2]]),
            r_wrist=     np.array([raw_t[3],     raw_t[4],     raw_t[5]]),
            l_ankle=     np.array([raw_t[6],     raw_t[7],     raw_t[8]]),
            r_ankle=     np.array([raw_t[9],     raw_t[10],    raw_t[11]]),
        )
    # derive elbows + knees as midpoints with slight bend
    def with_joints(s):
        for side, sign in (("l", -1), ("r", +1)):
            sh, wr = s[f"{side}_shoulder"], s[f"{side}_wrist"]
            mid = 0.5 * (sh + wr)
            s[f"{side}_elbow"] = mid + np.array([sign * -0.04 * h, 0.03 * h, -0.04 * h])
            hp, an = s[f"{side}_hip"], s[f"{side}_ankle"]
            midk = 0.5 * (hp + an)
            s[f"{side}_knee"] = midk + np.array([0.0, 0.03 * h, 0.0])
        return s

    patient_skel = [with_joints(skeleton_at(patient_raw[i], i)) for i in range(n_frames)]
    twin_skel    = [with_joints(skeleton_at(twin_raw[i],    i)) for i in range(n_frames)]

    # --- figure setup (compact for cloud perf) ---
    Y_FLOOR = 0.0
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor="#03060e", dpi=85)
    ax.set_facecolor("#03060e")
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-0.15, 2.25)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ---- cinematic backdrop: deep-blue vertical gradient ----
    bg_grad = np.linspace(0.08, 0.0, 300)[:, None] * np.ones((1, 600))
    ax.imshow(bg_grad, extent=[-3.6, 3.6, 0.0, 2.25],
              cmap="Blues", aspect="auto", zorder=-20, alpha=0.7)

    # ---- twin & patient spotlights (radial light pools on floor) ----
    for cx, color in ((-1.7, "#22c55e"), (1.7, "#f43f5e")):
        for r, alpha in [(1.5, 0.05), (1.0, 0.07), (0.6, 0.10)]:
            spot = Circle((cx, Y_FLOOR + 0.02), r,
                           facecolor=color, alpha=alpha,
                           zorder=-12, edgecolor="none")
            ax.add_patch(spot)

    # ---- floor line ----
    ax.plot([-3.6, 3.6], [Y_FLOOR, Y_FLOOR],
             color="#60a5fa", linewidth=1.5, alpha=0.5, zorder=-5)
    # floor centre dashed lane divider
    for x0 in np.arange(-3.5, 3.6, 0.4):
        ax.plot([x0, x0 + 0.22], [Y_FLOOR - 0.005, Y_FLOOR - 0.005],
                 color="#facc15", linewidth=2.5, alpha=0.30, zorder=-4)
    # meter ticks
    for tx in np.arange(-3, 3.1, 1.0):
        ax.plot([tx, tx], [Y_FLOOR - 0.04, Y_FLOOR + 0.02],
                 color="#60a5fa", linewidth=1, alpha=0.4, zorder=-3)

    # static labels (top of frame)
    ax.text(-1.7, 2.15, "●  HEALTHY TWIN", ha="center", color="#22c55e",
            family="monospace", fontsize=12, weight="bold")
    ax.text( 1.7, 2.15, "●  PATIENT",      ha="center", color="#f43f5e",
            family="monospace", fontsize=12, weight="bold")

    # mutable artist list cleared on each frame
    state = {"artists": []}

    SENSOR_KEYS_LOCAL = ("l_wrist", "r_wrist", "l_ankle", "r_ankle")
    SENSOR_LABELS_LOCAL = ("L-wrist", "R-wrist", "L-ankle", "R-ankle")

    def color_anom(l):
        return "#22c55e" if l < 0.25 else "#f59e0b" if l < 0.55 else "#ef4444"

    def render_body(j, x_off, body_color, halo_color, alpha=1.0,
                    body_zorder=5, reflect=False):
        """Render one humanoid body. If reflect=True, draws the body
        mirrored below the floor at low opacity (water-style reflection)."""
        arts = []
        # vertical flip multiplier for reflection mode
        if reflect:
            def Y(yv): return -yv * 0.7         # reflect below floor, compressed
            ref_alpha = alpha * 0.30
            zo = body_zorder - 10
        else:
            def Y(yv): return yv
            ref_alpha = alpha
            zo = body_zorder

        # soft elliptical shadow on floor (only for upright pass)
        if not reflect:
            px, _ = _project(j["pelvis"], x_off)
            for r, a in [(0.45, 0.25), (0.33, 0.35), (0.22, 0.45)]:
                shd = Polygon(
                    [[px - r, -0.012], [px + r, -0.012],
                     [px + r*0.85, 0.008], [px - r*0.85, 0.008]],
                    closed=True, facecolor="black",
                    alpha=alpha * a, zorder=zo + 0.1)
                ax.add_patch(shd); arts.append(shd)

        # torso polygon (filled — clearly visible body)
        torso = Polygon(
            [(p[0], Y(p[1])) for p in (_project(j["l_shoulder"], x_off),
                                         _project(j["r_shoulder"], x_off),
                                         _project(j["r_hip"],      x_off),
                                         _project(j["l_hip"],      x_off))],
            closed=True, facecolor=body_color, edgecolor=halo_color,
            linewidth=3.0, alpha=ref_alpha * 0.85, zorder=zo + 1,
            joinstyle="round")
        ax.add_patch(torso); arts.append(torso)

        # limb halo + crisp core — fatter, more visible
        for a_key, b_key in _BONE_PAIRS:
            pa = _project(j[a_key], x_off); pb = _project(j[b_key], x_off)
            ln_halo, = ax.plot([pa[0], pb[0]], [Y(pa[1]), Y(pb[1])],
                color=halo_color, linewidth=22, alpha=ref_alpha * 0.32,
                solid_capstyle="round", zorder=zo + 2)
            ln_core, = ax.plot([pa[0], pb[0]], [Y(pa[1]), Y(pb[1])],
                color=body_color, linewidth=9, alpha=ref_alpha,
                solid_capstyle="round", zorder=zo + 3)
            arts.extend([ln_halo, ln_core])

        # joint balls — larger and more contrasty
        for key in ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
                    "l_hip", "r_hip", "l_knee", "r_knee"):
            p = _project(j[key], x_off)
            c_halo = Circle((p[0], Y(p[1])), 0.085,
                       facecolor=halo_color, alpha=ref_alpha * 0.4,
                       zorder=zo + 3.5)
            c = Circle((p[0], Y(p[1])), 0.070,
                       facecolor=body_color, edgecolor=halo_color,
                       linewidth=2.5, alpha=ref_alpha, zorder=zo + 4)
            ax.add_patch(c_halo); ax.add_patch(c); arts.extend([c_halo, c])

        # hands — bigger filled discs at wrists
        for side in ("l", "r"):
            wp = _project(j[f"{side}_wrist"], x_off)
            hand_halo = Circle((wp[0], Y(wp[1])), 0.085,
                               facecolor=halo_color, alpha=ref_alpha * 0.4,
                               zorder=zo + 3.5)
            hand = Circle((wp[0], Y(wp[1])), 0.07,
                          facecolor=body_color, edgecolor=halo_color,
                          linewidth=2, alpha=ref_alpha, zorder=zo + 4)
            ax.add_patch(hand_halo); ax.add_patch(hand)
            arts.extend([hand_halo, hand])

        # feet — proper foot-shaped polygons, longer
        for side in ("l", "r"):
            ap = _project(j[f"{side}_ankle"], x_off)
            foot = Polygon(
                [(ap[0] - 0.08, Y(ap[1])),       (ap[0] + 0.20, Y(ap[1])),
                 (ap[0] + 0.20, Y(ap[1] + 0.06)),(ap[0] + 0.08, Y(ap[1] + 0.09)),
                 (ap[0] - 0.08, Y(ap[1] + 0.06))],
                closed=True, facecolor=body_color, edgecolor=halo_color,
                linewidth=2, alpha=ref_alpha, zorder=zo + 4)
            ax.add_patch(foot); arts.append(foot)

        # head — bigger, double halo, eye spots for character
        hp = _project(j["head"], x_off)
        head_halo2 = Circle((hp[0], Y(hp[1])), 0.27,
                             facecolor=halo_color,
                             alpha=ref_alpha * 0.18, zorder=zo + 4)
        head_halo = Circle((hp[0], Y(hp[1])), 0.22,
                            facecolor=halo_color,
                            alpha=ref_alpha * 0.30, zorder=zo + 4.5)
        head = Circle((hp[0], Y(hp[1])), 0.17,
                      facecolor=body_color, edgecolor=halo_color,
                      linewidth=3, alpha=ref_alpha, zorder=zo + 5)
        spec = Circle((hp[0] - 0.055, Y(hp[1] + (0.055 if not reflect else -0.055))),
                       0.035, facecolor="white",
                       alpha=ref_alpha * 0.75, zorder=zo + 6)
        ax.add_patch(head_halo2); ax.add_patch(head_halo)
        ax.add_patch(head); ax.add_patch(spec)
        arts.extend([head_halo2, head_halo, head, spec])
        return arts

    def render_sensors_and_trails(i, x_off, anom_frame):
        arts = []
        for k, key in enumerate(SENSOR_KEYS_LOCAL):
            # trail (last 6 frames) — alpha gradient
            n_trail = min(6, i + 1)
            for tf in range(n_trail):
                idx = i - tf
                p = _project(patient_skel[idx][key], x_off)
                t_alpha = (1 - tf / max(n_trail, 1)) * 0.55
                size = max(2, 100 - 14 * tf)
                ln, = ax.plot([p[0]], [p[1]], "o", markersize=size**0.5 * 3,
                              color=color_anom(anom_frame[k]), alpha=t_alpha,
                              markeredgewidth=0, zorder=20)
                arts.append(ln)
            # current sensor (bright diamond + glow halo)
            p = _project(patient_skel[i][key], x_off)
            glow = Circle(p, 0.10, facecolor=color_anom(anom_frame[k]),
                           alpha=0.30, zorder=21)
            ax.add_patch(glow); arts.append(glow)
            sc = ax.scatter([p[0]], [p[1]], s=140,
                             c=color_anom(anom_frame[k]),
                             marker="D", edgecolors="white", linewidths=1.5,
                             zorder=22)
            arts.append(sc)
        return arts

    def render_saliency_pulse(i, x_off):
        """Soft pulsing ring on impaired limbs once the AI has fired."""
        if sensor_sal is None or alert_after is None or i < alert_after:
            return []
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (i - alert_after) / 12)
        arts = []
        sal_map = {"l_wrist": 0, "r_wrist": 1, "l_ankle": 2, "r_ankle": 3}
        for key, idx in sal_map.items():
            sv = float(sensor_sal[idx])
            if sv > 0.45:
                p = _project(patient_skel[i][key], x_off)
                ring = Circle(p, 0.10 + 0.08 * sv + 0.04 * pulse,
                    facecolor="none", edgecolor="#facc15",
                    linewidth=2.5 + 1.5 * pulse, alpha=0.55 + 0.25 * pulse,
                    zorder=23)
                ax.add_patch(ring); arts.append(ring)
        return arts

    def update(i):
        for a in state["artists"]:
            try: a.remove()
            except Exception: pass
        state["artists"] = []
        alerted = alert_after is not None and i >= alert_after
        body_p = "#fecaca" if alerted else "#bae6fd"
        halo_p = "#ef4444" if alerted else "#22d3ee"
        state["artists"].extend(render_body(patient_skel[i], 1.7,
                                              body_p, halo_p, alpha=1.0,
                                              body_zorder=10))
        state["artists"].extend(render_body(twin_skel[i], -1.7,
                                              "#a7f3d0", "#22c55e", alpha=0.55,
                                              body_zorder=5))
        # sensors + trails on patient
        state["artists"].extend(render_sensors_and_trails(i, 1.7, anom[i]))
        # saliency pulse on impaired limbs (patient side)
        state["artists"].extend(render_saliency_pulse(i, 1.7))
        # alert flash banner across top
        if alerted:
            flash_alpha = 0.6 + 0.3 * np.sin(2 * np.pi * (i - alert_after) / 8)
            txt = ax.text(0, 2.05, "■  AI ANOMALY ALERT  ■",
                ha="center", color="#fecaca", fontsize=14, weight="bold",
                family="monospace", alpha=flash_alpha,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#7f1d1d",
                          edgecolor="#ef4444", linewidth=2,
                          alpha=flash_alpha * 0.85))
            state["artists"].append(txt)
        # frame counter / time
        ts = i * 0.075
        tt = ax.text(-3.45, 2.05, f"t = {ts:5.2f} s   frame {i:02d}/{n_frames-1}",
            color="#9ca3af", fontsize=9, family="monospace", ha="left")
        state["artists"].append(tt)
        return state["artists"]

    update(0)
    anim = FuncAnimation(fig, update, frames=n_frames,
                          interval=1000 / fps, blit=False)
    # PillowWriter needs a path on disk, not a BytesIO buffer
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    try:
        anim.save(tmp.name, writer=PillowWriter(fps=fps))
        with open(tmp.name, "rb") as f:
            data = f.read()
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass
        plt.close(fig)
    return data


def render_scene_gif_for(patient_raw, twin_raw, anom, alert_after,
                          sensor_saliency=None, fps=24):
    """Wrapper that hashes the inputs and calls the cached GIF renderer."""
    patient_raw = np.ascontiguousarray(patient_raw, dtype=np.float64)
    twin_raw    = np.ascontiguousarray(twin_raw,    dtype=np.float64)
    anom        = np.ascontiguousarray(anom,        dtype=np.float64)
    sal_bytes = b""
    if sensor_saliency is not None:
        sal_bytes = np.ascontiguousarray(sensor_saliency,
                                          dtype=np.float64).tobytes()
    key = _hash_inputs(patient_raw, twin_raw, anom,
                        np.asarray([alert_after if alert_after is not None else -1,
                                     fps], dtype=np.int64),
                        np.frombuffer(sal_bytes, dtype=np.float64)
                        if sal_bytes else np.array([0.0]))
    return render_cinematic_gif(patient_raw.tobytes(),
                                  twin_raw.tobytes(),
                                  anom.tobytes(),
                                  alert_after, sal_bytes,
                                  patient_raw.shape[0], key, fps=fps)


def build_clinical_scene(patient_skel, twin_skel, raw, anom,
                          brain_features=None, prob_curve=None, prob_centers=None,
                          onset_frame=None, alert_after=None,
                          sensor_saliency=None, height=560):
    """Treadmill-style clinical gait scene.

    The subject walks in place (pelvis_y normalized to 0 each frame), the
    camera stays rock-still on the body, and a healthy digital twin walks
    alongside at lateral offset for direct visual comparison. Returns a
    single 3D Plotly figure — no subplots, no per-frame camera jitter.
    """
    patient_skel = _in_place(patient_skel)
    if twin_skel is not None:
        twin_skel = _in_place(twin_skel)
    n = len(patient_skel)

    PX = 0.5    # patient lane x
    TX = -1.3   # twin lane x
    Y_HALF = 0.9
    Z_TOP = 2.0

    fig = go.Figure()

    # --- floor (single quad) ---
    fig.add_trace(_quad(TX - 0.9, PX + 0.9, -Y_HALF, Y_HALF, 0.0, "#0b1426"))
    # --- subtle grid ---
    for gx in np.arange(TX - 0.9, PX + 0.91, 0.4):
        fig.add_trace(go.Scatter3d(
            x=[gx, gx], y=[-Y_HALF, Y_HALF], z=[0.005, 0.005],
            mode="lines", line=dict(color="#1e3a5f", width=1),
            opacity=0.4, hoverinfo="skip", showlegend=False))
    for gy in np.arange(-Y_HALF, Y_HALF + 0.01, 0.3):
        fig.add_trace(go.Scatter3d(
            x=[TX - 0.9, PX + 0.9], y=[gy, gy], z=[0.005, 0.005],
            mode="lines", line=dict(color="#1e3a5f", width=1),
            opacity=0.4, hoverinfo="skip", showlegend=False))
    # --- lane labels ---
    fig.add_trace(go.Scatter3d(
        x=[PX], y=[Y_HALF - 0.05], z=[0.04], mode="text",
        text=["PATIENT"],
        textfont=dict(color="#f43f5e", size=12, family="monospace"),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=[TX], y=[Y_HALF - 0.05], z=[0.04], mode="text",
        text=["HEALTHY TWIN"],
        textfont=dict(color="#22c55e", size=12, family="monospace"),
        hoverinfo="skip", showlegend=False))

    n_static = len(fig.data)

    def dyn(i):
        pj = patient_skel[i]
        tj = twin_skel[i] if twin_skel is not None else None
        alerted = alert_after is not None and i >= alert_after

        # patient shadow (slight oval under feet)
        p_shadow = go.Scatter3d(
            x=[PX], y=[0.0], z=[0.012], mode="markers",
            marker=dict(size=36, color="black", opacity=0.45),
            hoverinfo="skip", showlegend=False)
        # patient volumetric humanoid (5 traces)
        body = "#fecaca" if alerted else "#cffafe"
        halo = "#ef4444" if alerted else "#06b6d4"
        p_body = _humanoid_traces(pj, PX, body, halo, alpha=1.0)

        # twin shadow + ghostly humanoid
        if tj is not None:
            t_shadow = go.Scatter3d(
                x=[TX], y=[0.0], z=[0.012], mode="markers",
                marker=dict(size=28, color="black", opacity=0.30),
                hoverinfo="skip", showlegend=False)
            t_body = _humanoid_traces(tj, TX, "#a7f3d0", "#22c55e", alpha=0.6)
        else:
            t_shadow = go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False)
            t_body = [go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False)
                for _ in range(5)]

        # 4 IMU diamonds (with anomaly colour) + glowing trail per sensor
        sx, sy, sz, sc, sh = [], [], [], [], []
        for k, key in enumerate(SENSOR_KEYS):
            sx.append(pj[key][0] + PX)
            sy.append(pj[key][1]); sz.append(pj[key][2])
            sc.append(_color_anom(anom[i, k]))
            sh.append(f"{SENSOR_LABELS[k]}<br>anomaly={anom[i,k]:.2f}")
        sensors = go.Scatter3d(
            x=sx, y=sy, z=sz, mode="markers",
            marker=dict(size=14, color=sc, symbol="diamond",
                        line=dict(color="white", width=2), opacity=1.0),
            hovertext=sh, hoverinfo="text", showlegend=False)

        trail_traces = []
        for k, key in enumerate(SENSOR_KEYS):
            s0 = max(0, i - 6)
            tx = [patient_skel[fi][key][0] + PX for fi in range(s0, i + 1)]
            ty = [patient_skel[fi][key][1]      for fi in range(s0, i + 1)]
            tz = [patient_skel[fi][key][2]      for fi in range(s0, i + 1)]
            trail_traces.append(go.Scatter3d(
                x=tx, y=ty, z=tz, mode="lines",
                line=dict(color=sc[k], width=5), opacity=0.55,
                hoverinfo="skip", showlegend=False))

        # saliency glow rings (always 4 traces for stable indexing)
        sal_traces = []
        sal_map = {"l_wrist": 0, "r_wrist": 1, "l_ankle": 2, "r_ankle": 3}
        for key, idx in sal_map.items():
            if sensor_saliency is not None and alerted and sensor_saliency[idx] > 0.4:
                pt = pj[key].copy(); pt[0] += PX
                sv = float(sensor_saliency[idx])
                sal_traces.append(go.Scatter3d(
                    x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers",
                    marker=dict(size=22 + 30 * sv, color="#facc15",
                                opacity=0.40,
                                line=dict(color="#fde047", width=2)),
                    hoverinfo="skip", showlegend=False))
            else:
                sal_traces.append(go.Scatter3d(
                    x=[0], y=[0], z=[0], mode="markers",
                    marker=dict(size=1, opacity=0),
                    hoverinfo="skip", showlegend=False))

        return [p_shadow, *p_body, t_shadow, *t_body,
                sensors, *trail_traces, *sal_traces]

    # ---- initial frame ----
    for tr in dyn(0):
        fig.add_trace(tr)
    dyn_indices = list(range(n_static, len(fig.data)))

    # ---- frames ----
    fig.frames = [go.Frame(data=dyn(i), traces=dyn_indices, name=str(i))
                  for i in range(n)]

    # ---- TIGHT close-up camera, never moves (subject walks in place) ----
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[TX - 1.0, PX + 1.0], showbackground=False,
                       visible=False),
            yaxis=dict(range=[-Y_HALF, Y_HALF], showbackground=False,
                       visible=False),
            zaxis=dict(range=[0, Z_TOP], showbackground=False, visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=2.2, y=-1.8, z=1.1),
                        center=dict(x=(TX + PX) / 2 - 0.1, y=0.0, z=0.95),
                        up=dict(x=0, y=0, z=1)),
            bgcolor="#05070d"),
        paper_bgcolor="#05070d",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=20, b=20),
        height=height, showlegend=False,
        updatemenus=[dict(type="buttons", x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            bgcolor="#1f2937", font=dict(color="white"),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=60,
                                                       easing="cubic-in-out"))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.12, y=0.02, len=0.82,
            currentvalue=dict(prefix="frame ", font=dict(color="#e5e7eb")),
            steps=[dict(method="animate", label=str(i),
                args=[[str(i)], dict(mode="immediate",
                                      frame=dict(duration=0, redraw=True),
                                      transition=dict(duration=0))])
                for i in range(n)])])
    return fig


def build_brain_panel(brain_features, prob_curve=None, prob_centers=None,
                       onset_frame=None, alert_after=None, height=240):
    """Companion AI-brain heatmap (24 biomarker channels × time) with
    onset / alert markers — shown as a separate static chart under the
    3D scene so the scene stays simple and fast."""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=brain_features.T,
        x=np.arange(brain_features.shape[0]), y=np.arange(24),
        colorscale="inferno", showscale=False,
        zmin=float(np.percentile(brain_features, 2)),
        zmax=float(np.percentile(brain_features, 98)),
        hovertemplate="t=%{x}  ch=%{y}  z=%{z:.2f}<extra></extra>"))
    if onset_frame is not None:
        fig.add_vline(x=onset_frame, line=dict(color="#22c55e", dash="dash"),
                       annotation_text="onset", annotation_position="top")
    if alert_after is not None:
        fig.add_vline(x=alert_after, line=dict(color="#ef4444"),
                       annotation_text="ALERT", annotation_position="top")
    fig.update_layout(
        title=dict(text="🧠 AI brain — 24 biomarker channels × time",
                    font=dict(color="white", size=13), x=0.01),
        paper_bgcolor="#05070d", plot_bgcolor="#05070d",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=40, b=10), height=height,
        xaxis=dict(title="frame", color="#9ca3af"),
        yaxis=dict(title="channel", color="#9ca3af"))
    return fig


# ===================================================================
#                          CLINICAL REPORT CARD
# ===================================================================
def render_clinical_report(metrics, patient_metrics_twin=None):
    """Render the clinical card stack with normative flags."""
    def row(label, val, fmt, flag):
        emoji, css, _ = flag
        st.markdown(
            f"<div class='clin-row'><span>{label}</span>"
            f"<span class='clin-val {css}'>{val:{fmt}} {emoji}</span></div>",
            unsafe_allow_html=True)

    m = metrics
    # spatiotemporal card
    st.markdown("<div class='clin-card'><h4>Spatiotemporal</h4>", unsafe_allow_html=True)
    row("Cadence (steps/min)", m["cadence_spm"], ".0f",
        _flag(m["cadence_spm"], 100, 130))
    row("Walking speed (m/s)", m["walking_speed_mps"], ".2f",
        _flag(m["walking_speed_mps"], 1.10, 1.45))
    row("Stride length (m)", m["stride_length_m"], ".2f",
        _flag(m["stride_length_m"], 1.20, 1.60))
    st.markdown("</div>", unsafe_allow_html=True)

    # Symmetry (Robinson SI 1987 — <5% normal, 5-15% borderline, >15% abnormal)
    st.markdown("<div class='clin-card'><h4>Symmetry Indices (Robinson SI)</h4>", unsafe_allow_html=True)
    row("Upper-extremity (arms)", m["symmetry_arms_pct"], ".1f",
        _flag(m["symmetry_arms_pct"], 5, 15, prefer="low"))
    row("Lower-extremity (legs)", m["symmetry_legs_pct"], ".1f",
        _flag(m["symmetry_legs_pct"], 5, 15, prefer="low"))
    row("Foot clearance", m["symmetry_clearance_pct"], ".1f",
        _flag(m["symmetry_clearance_pct"], 5, 15, prefer="low"))
    row("Composite asymmetry", m["composite_asymmetry_pct"], ".1f",
        _flag(m["composite_asymmetry_pct"], 5, 15, prefer="low"))
    st.markdown("</div>", unsafe_allow_html=True)

    # ROM (L vs R)
    st.markdown("<div class='clin-card'><h4>Range of motion (L vs R)</h4>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='clin-row'><span>Arm swing (m)</span>"
        f"<span class='clin-val'>{m['arm_rom_L']:.2f} / {m['arm_rom_R']:.2f}</span></div>"
        f"<div class='clin-row'><span>Leg swing (m)</span>"
        f"<span class='clin-val'>{m['leg_rom_L']:.2f} / {m['leg_rom_R']:.2f}</span></div>"
        f"<div class='clin-row'><span>Foot clearance (cm)</span>"
        f"<span class='clin-val'>{m['foot_clearance_L']*100:.1f} / {m['foot_clearance_R']*100:.1f}</span></div>"
        "</div>", unsafe_allow_html=True)


def render_neurons_saved(alert_frame_idx, onset_frame_idx, dt=0.075):
    """Time-is-brain banner: how many neurons the AI's early alert avoided
    versus a 30-second bystander baseline. Source: Saver, Stroke 2006."""
    if alert_frame_idx is None or onset_frame_idx is None:
        st.markdown(
            "<div class='neurons-card'>"
            "<div class='big'>—</div>"
            "<div class='sub'>No alert was raised, or no pathology onset to "
            "compare against. (Neurons-saved counter is only meaningful for "
            "true positive events.)</div></div>", unsafe_allow_html=True)
        return
    detection_lag = max(0.0, (alert_frame_idx - onset_frame_idx) * dt)
    saved_seconds = max(0.0, BYSTANDER_REACTION_SEC - detection_lag)
    saved_neurons = int(saved_seconds * NEURONS_PER_SECOND_LOST)
    st.markdown(
        f"<div class='neurons-card'>"
        f"<div class='sub'>⏱ Detection lag: <b>{detection_lag:.2f} s</b> · "
        f"bystander baseline: <b>{BYSTANDER_REACTION_SEC:.0f} s</b></div>"
        f"<div class='big'>{saved_neurons:,} neurons saved</div>"
        f"<div class='sub'>Computed against Saver JL, "
        "<i>Time Is Brain — Quantified</i>, Stroke 2006 "
        f"(≈{NEURONS_PER_SECOND_LOST/1000:.0f}K neurons lost per second of "
        "untreated ischemic stroke). The earlier the AI fires, the more brain "
        "tissue tPA / thrombectomy can rescue.</div></div>",
        unsafe_allow_html=True)


# ===================================================================
#                                LAYOUT
# ===================================================================
tabs = st.tabs(["🩺 Diagnosis", "📈 Biomarkers",
                "🚶 Clinical scenarios (300)",
                "🧪 Test the AI", "🧬 Architecture"])

with st.sidebar:
    st.header("⚙️ Telemetry")
    profile = st.radio("Patient profile", PROFILES,
                       format_func=lambda p: PROFILE_LABELS[p])
    severity = st.slider("Severity", 0.0, 1.0, 0.8, 0.05,
                         disabled=(profile == "healthy"))
    seed = st.number_input("Random seed", value=7, step=1)
    n_frames_demo = st.slider("Animation length (frames)", 40, 120, 80, 10)
    run = st.button("▶ Run analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption("Live simulator and training simulator share the same code "
               "(`gait_sim` + `scenarios`) — what you see is exactly what the "
               "AI scored.")


# ----- TAB 1: DIAGNOSIS -------------------------------------------
with tabs[0]:
    if not run:
        st.info("👈 Choose a profile and click **Run analysis**.")
    else:
        rng = np.random.default_rng(int(seed))
        raw, sev_curve, subject, phase = simulate_walk(
            profile, rng, n_frames=n_frames_demo, onset_frame=0,
            peak_severity=float(severity) if profile != "healthy" else 0.0,
            ramp_frames=1)
        # twin = same subject_seed walking healthy
        rng_twin = np.random.default_rng(int(seed))
        twin_raw, _, twin_subj, twin_phase = simulate_walk(
            "healthy", rng_twin, n_frames=n_frames_demo,
            onset_frame=None, peak_severity=0.0)
        skel = build_skeleton_frames(raw, subject, phase, walk_forward=True)
        twin_skel = build_skeleton_frames(twin_raw, twin_subj, twin_phase,
                                          walk_forward=True)
        anom = per_sensor_anomaly(raw)

        all_feats = build_features(raw)
        brain_diag = scaler.transform(all_feats)
        mid = max(0, n_frames_demo // 2 - WINDOW // 2)
        scaled_win = brain_diag[mid:mid + WINDOW].reshape(1, WINDOW, 24)
        p_abn = float(bin_model.predict(scaled_win, verbose=0)[0, 0])
        mc_probs = mc_model.predict(scaled_win, verbose=0)[0]
        mc_top = int(mc_probs.argmax())
        metrics = compute_clinical_metrics(raw, subject)

        c1, c2 = st.columns([1.6, 1])
        with c1:
            st.subheader(f"🎬 Cinematic gait — {PROFILE_LABELS[profile]}")
            gif_bytes = render_scene_gif_for(raw, twin_raw, anom,
                                              alert_after=None, fps=24)
            st.image(gif_bytes, use_container_width=True)
            st.caption("Cinematic 2-D sagittal projection — the standard "
                       "clinical gait-lab view. **Right** = patient · "
                       "**Left** = healthy digital twin (same subject, "
                       "counterfactual healthy kinematics). Every kinematic "
                       "asymmetry — reduced arm swing, foot drop, tremor, "
                       "bradykinesia — is immediately visible against the "
                       "ghosted twin.")
            st.plotly_chart(build_brain_panel(brain_diag, height=220),
                             use_container_width=True)
        with c2:
            st.subheader("AI diagnosis")
            if p_abn > 0.5:
                st.error(f"🚨 ABNORMAL\n\n**P(abnormal) = {p_abn*100:.1f}%**")
            else:
                st.success(f"✅ NORMAL\n\n**P(abnormal) = {p_abn*100:.1f}%**")
            st.markdown(f"**Differential:** `{PROFILES[mc_top]}` — "
                        f"{PROFILE_LABELS[PROFILES[mc_top]]}")
            bar = go.Figure(go.Bar(
                x=mc_probs*100, y=list(PROFILES), orientation="h",
                marker_color=["#22c55e" if i == mc_top else "#3b82f6"
                              for i in range(len(PROFILES))],
                text=[f"{p*100:.1f}%" for p in mc_probs], textposition="outside"))
            bar.update_layout(height=240, margin=dict(l=10, r=30, t=10, b=10),
                              xaxis=dict(range=[0, 110], color="#9ca3af"),
                              yaxis=dict(color="#9ca3af"),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"))
            st.plotly_chart(bar, use_container_width=True)
            render_clinical_report(metrics)


# ----- TAB 2: BIOMARKERS ------------------------------------------
with tabs[1]:
    if not run:
        st.info("Run an analysis from the sidebar first.")
    else:
        st.subheader("4-sensor centered magnitudes (1.5 s window)")
        slice_ = raw[mid:mid + WINDOW]
        fig = go.Figure()
        for i, name in enumerate(SENSOR_LABELS):
            mag = np.linalg.norm(
                slice_[:, 3*i:3*i+3] - slice_[:, 3*i:3*i+3].mean(0), axis=1)
            fig.add_trace(go.Scatter(y=mag, mode="lines+markers", name=name))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"),
                          yaxis_title="|XYZ − mean| (m)", xaxis_title="frame")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("24-channel scaled biomarker heatmap (model input)")
        hm = px.imshow(scaled_win[0].T, color_continuous_scale="inferno",
                       aspect="auto",
                       labels=dict(x="timestep", y="feature ch.", color="z"))
        hm.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                         paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(hm, use_container_width=True)
        st.caption("0–11 raw XYZ · 12–17 L−R drift · 18–21 limb magnitudes · "
                   "22–23 upper/lower asymmetry index.")


# ----- TAB 3: CLINICAL SCENARIOS (300) ----------------------------
with tabs[2]:
    st.subheader("🚶 Clinical scenarios — 300 procedurally generated pedestrians")
    st.caption("A subject walks down a virtual MoCap lane. At a sampled moment, "
               "a neurological event may strike. The 4-IMU stream feeds the "
               "Motion-AI, which raises an alert when its 1.5 s rolling-window "
               "confidence holds above 50 % for 3 consecutive windows "
               "(~0.23 s of sustained evidence).")

    filter_col, nav_col, _ = st.columns([1.2, 1, 1])
    with filter_col:
        prof_filter = st.multiselect("Filter scenarios",
                                     options=list(PROFILES) + ["abnormal-only"],
                                     default=[])
    pool = SCENARIOS
    if prof_filter:
        if "abnormal-only" in prof_filter:
            pool = [s for s in pool if s["profile"] != "healthy"]
        else:
            pool = [s for s in pool if s["profile"] in prof_filter]
    pool_ids = [s["id"] for s in pool]
    if not pool_ids:
        st.warning("No scenarios match that filter."); st.stop()
    if ("scen_idx" not in st.session_state
            or st.session_state.scen_idx not in pool_ids):
        st.session_state.scen_idx = pool_ids[0]
    with nav_col:
        a, b, c = st.columns(3)
        if a.button("⏮ Prev"):
            i = pool_ids.index(st.session_state.scen_idx)
            st.session_state.scen_idx = pool_ids[(i - 1) % len(pool_ids)]
        if b.button("🎲 Random"):
            st.session_state.scen_idx = int(np.random.choice(pool_ids))
        if c.button("Next ⏭"):
            i = pool_ids.index(st.session_state.scen_idx)
            st.session_state.scen_idx = pool_ids[(i + 1) % len(pool_ids)]

    scen = SCENARIOS[st.session_state.scen_idx]
    onset_text = "—" if scen["onset_seconds"] is None else f"{scen['onset_seconds']} s"
    st.markdown(
        f"**Scenario #{scen['id']+1} / 300** · "
        f"<span class='scen-pill'>{PROFILE_EMOJI[scen['profile']]} "
        f"{scen['profile']}</span> "
        f"<span class='scen-pill'>severity {scen['severity']:.2f}</span> "
        f"<span class='scen-pill'>onset {onset_text}</span> "
        f"<span class='scen-pill'>subject seed {scen['subject_seed']}</span>",
        unsafe_allow_html=True)

    # ----- simulate patient + digital twin -----
    rng = np.random.default_rng(scen["subject_seed"])
    raw, sev_curve, subject, phase = simulate_walk(
        scen["profile"], rng, n_frames=80,
        onset_frame=scen["onset_frame"], peak_severity=scen["severity"])
    rng_twin = np.random.default_rng(scen["subject_seed"])
    twin_raw, _, twin_subj, twin_phase = simulate_walk(
        "healthy", rng_twin, n_frames=80,
        onset_frame=None, peak_severity=0.0)
    skel = build_skeleton_frames(raw, subject, phase, walk_forward=True)
    twin_skel = build_skeleton_frames(twin_raw, twin_subj, twin_phase,
                                      walk_forward=True)
    anom = per_sensor_anomaly(raw)
    centers, probs = sliding_window_inference(raw, scaler, bin_model, build_features)
    alert = alert_frame(probs, centers)
    all_feats = build_features(raw)
    brain_full = scaler.transform(all_feats)
    # Compute clinical metrics on the trailing ~3 s so they reflect the
    # current (post-ramp) gait state, not a healthy-prefix average.
    metrics_start = max(0, len(raw) - 40)
    metrics = compute_clinical_metrics(raw, subject, start_frame=metrics_start)

    # multiclass on the final window
    last_scaled = brain_full[-WINDOW:].reshape(1, WINDOW, 24)
    mc_final = mc_model.predict(last_scaled, verbose=0)[0]
    mc_top = int(mc_final.argmax())

    # gradient saliency on the alert-triggering window
    sal_start = max(0, min(len(raw) - WINDOW,
                            (alert if alert is not None else len(raw) - 1) - WINDOW // 2))
    sal_in = brain_full[sal_start:sal_start + WINDOW].reshape(1, WINDOW, 24)
    _, per_sensor_sal = compute_saliency(sal_in, bin_model)

    left, right = st.columns([1.55, 1])
    with left:
        gif_bytes = render_scene_gif_for(
            raw, twin_raw, anom,
            alert_after=alert, sensor_saliency=per_sensor_sal, fps=24)
        st.image(gif_bytes, use_container_width=True)
        st.plotly_chart(build_brain_panel(
            brain_full, prob_curve=probs, prob_centers=centers,
            onset_frame=scen["onset_frame"], alert_after=alert,
            height=240), use_container_width=True)
        st.caption("▶ Press **Play**. The patient (red lane) and healthy "
                   "**digital twin** (green lane) share the same anthropometrics "
                   "and the same gait phase — every deviation you see between "
                   "them is the deficit. IMU diamonds turn amber→red as their "
                   "signal diverges from the pre-onset baseline. The skeleton "
                   "glows red the moment the AI fires; affected joints swell "
                   "proportionally to the gradient saliency `∂P(abn)/∂x` — "
                   "the AI's own answer to *“which sensor told you?”* The brain "
                   "panel below shows the 24 biomarker channels lit up over "
                   "time, with the yellow line marking playback frame, dashed "
                   "green = true onset, solid red = AI alert.")

    with right:
        # 1. AI verdict
        st.subheader("🤖 AI verdict")
        if alert is None and scen["profile"] == "healthy":
            st.success("✅ No anomaly across 6 s of walking. Classified **healthy**.")
        elif alert is None:
            st.warning(f"⚠️ No alert; ground truth = `{scen['profile']}`. "
                       f"Sub-threshold for this subject (severity "
                       f"{scen['severity']:.2f}).")
        else:
            onset_s = (f"{scen['onset_frame']*0.075:.2f} s"
                       if scen["onset_frame"] is not None else "—")
            lag = ("n/a" if scen["onset_frame"] is None
                   else f"{(alert - scen['onset_frame'])*0.075:+.2f} s")
            st.error(f"🚨 **ANOMALY** at t = {alert*0.075:.2f} s · "
                     f"onset {onset_s} (Δ {lag})\n\n"
                     f"Differential: **{PROFILES[mc_top]}** "
                     f"({mc_final[mc_top]*100:.1f} %)")

        # 2. Neurons-saved (time-is-brain)
        st.subheader("⏱ Time-is-brain")
        render_neurons_saved(alert, scen["onset_frame"])

        # 3. Clinical report card (validated metrics)
        st.subheader("📋 Clinical report card")
        render_clinical_report(metrics)

    # Rolling P(abnormal) curve under the main scene
    st.divider()
    st.subheader("📈 Rolling P(abnormal) over the walk")
    pfig = go.Figure()
    pfig.add_hrect(y0=0.5, y1=1.0, fillcolor="#ef4444", opacity=0.08, line_width=0)
    pfig.add_trace(go.Scatter(x=centers * 0.075, y=probs, mode="lines+markers",
        line=dict(color="#f87171", width=3),
        marker=dict(size=6, color="#fecaca"), name="P(abnormal)"))
    pfig.add_hline(y=0.5, line=dict(color="#9ca3af", dash="dash"),
                   annotation_text="threshold")
    if scen["onset_frame"] is not None:
        pfig.add_vline(x=scen["onset_frame"]*0.075,
                       line=dict(color="#22c55e", dash="dot"),
                       annotation_text="onset")
    if alert is not None:
        pfig.add_vline(x=alert*0.075, line=dict(color="#ef4444"),
                       annotation_text="ALERT")
    pfig.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="time (s)", color="#9ca3af"),
        yaxis=dict(title="P(abnormal)", range=[-0.05, 1.05], color="#9ca3af"))
    st.plotly_chart(pfig, use_container_width=True)

    # ================== k-NN cohort retrieval (innovative feature) ============
    st.divider()
    st.subheader("🔎 Cohort retrieval — most similar prior cases")
    st.caption("**Innovative clinical-decision-support layer.** We embed the "
               "current scenario's final 1.5 s window into the model's own "
               "**Dense(32) latent space** and retrieve its 5 nearest "
               "neighbours from the 300-pedestrian cohort. This is the same "
               "principle as case-based reasoning in radiology / pathology "
               "(*“patients whose signature looked like this were diagnosed "
               "as X with outcome Y”*). It also lets you audit *why* the AI "
               "decided what it did — by looking at which cases it considers "
               "this scenario's nearest precedents.")

    current_emb = emb_model.predict(last_scaled, verbose=0)[0]
    dists = np.linalg.norm(COHORT_EMB - current_emb, axis=1)
    dists[scen["id"]] = np.inf      # exclude self
    nearest_ids = np.argsort(dists)[:5]
    near_cols = st.columns(5)
    for col, nid in zip(near_cols, nearest_ids):
        ns = SCENARIOS[int(nid)]
        d = float(dists[nid])
        match_strength = max(0.0, 1.0 - d / (np.median(dists) + 1e-6))
        n_onset = "—" if ns["onset_seconds"] is None else f"{ns['onset_seconds']:.1f}s"
        col.markdown(
            f"<div class='clin-card'>"
            f"<h4>#{ns['id']+1}</h4>"
            f"<div class='clin-row'><span>profile</span>"
            f"<span class='clin-val'>{PROFILE_EMOJI[ns['profile']]} "
            f"{ns['profile']}</span></div>"
            f"<div class='clin-row'><span>severity</span>"
            f"<span class='clin-val'>{ns['severity']:.2f}</span></div>"
            f"<div class='clin-row'><span>onset</span>"
            f"<span class='clin-val'>{n_onset}</span></div>"
            f"<div class='clin-row'><span>distance</span>"
            f"<span class='clin-val'>{d:.2f}</span></div>"
            f"<div class='clin-row'><span>similarity</span>"
            f"<span class='clin-val flag-ok'>{match_strength*100:.0f}%</span></div>"
            f"</div>", unsafe_allow_html=True)

    # AI brain breakdown (4-column step-by-step retained but compact)
    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("**1. Sense.** 4 IMU bands → 12 numbers / frame @ 13 Hz.")
    s2.markdown("**2. Engineer.** Rolling 1.5 s window → 24 biomarkers "
                "(raw + drift + magnitudes + asymmetry).")
    s3.markdown("**3. Classify.** Conv1D×2 → BiLSTM(48) → Dense(32) → sigmoid; "
                "shared backbone with 6-class differential head.")
    s4.markdown("**4. Debounce + alert.** P(abn) > 0.5 for 3 consecutive "
                "windows ≈ 0.23 s of sustained evidence.")

    with st.expander("📚 See the whole 300-scenario index"):
        import pandas as pd
        df = pd.DataFrame([{
            "id": s["id"] + 1, "profile": s["profile"],
            "severity": round(s["severity"], 2),
            "onset (s)": s["onset_seconds"],
            "subject seed": s["subject_seed"],
        } for s in SCENARIOS])
        st.dataframe(df, use_container_width=True, height=380)


# ----- TAB 4: TEST THE AI -----------------------------------------
with tabs[3]:
    st.subheader("🧪 Test the Motion-AI")
    st.caption("Three ways to probe the model directly: a parameter "
               "**sandbox** for hand-crafted deficits, a **pathology sweep** "
               "across severities, and a **full-cohort benchmark** with "
               "confusion matrix and detection-lag statistics.")
    sub_a, sub_b, sub_c = st.tabs(["🎛 Sandbox",
                                   "🔥 Pathology sweep",
                                   "📊 Cohort benchmark"])

    # ============== SANDBOX ==================================
    with sub_a:
        st.markdown("**Hand-craft a deficit.** Each slider drives one "
                    "independent biomechanical parameter. The AI's verdict "
                    "updates live — useful for probing decision boundaries "
                    "(*“how low does the left arm swing have to go before "
                    "the model fires?”*).")
        col_p, col_v = st.columns([1, 1.4])
        with col_p:
            st.markdown("##### Upper extremity")
            sb_arm_L = st.slider("Left arm swing (m)",  0.00, 0.60, 0.40, 0.02, key="sb_arm_L")
            sb_arm_R = st.slider("Right arm swing (m)", 0.00, 0.60, 0.40, 0.02, key="sb_arm_R")
            sb_tremor = st.slider("Tremor amplitude (m)", 0.00, 0.05, 0.0, 0.005, key="sb_tremor")
            sb_jitter = st.slider("Lateral jitter σ (m)", 0.00, 0.15, 0.0, 0.01, key="sb_jitter")
            st.markdown("##### Lower extremity")
            sb_leg_L = st.slider("Left leg swing (m)",  0.00, 0.80, 0.55, 0.02, key="sb_leg_L")
            sb_leg_R = st.slider("Right leg swing (m)", 0.00, 0.80, 0.55, 0.02, key="sb_leg_R")
            sb_clear_L = st.slider("Left foot clearance (m)",  0.00, 0.20, 0.12, 0.01, key="sb_clr_L")
            sb_clear_R = st.slider("Right foot clearance (m)", 0.00, 0.20, 0.12, 0.01, key="sb_clr_R")
            st.markdown("##### Global")
            sb_cad = st.slider("Cadence (Hz)", 0.6, 1.4, 1.0, 0.05, key="sb_cad")
            sb_noise = st.slider("IMU noise σ (m)", 0.000, 0.05, 0.012, 0.002, key="sb_noise")

        with col_v:
            raw_sb, sub_sb, ph_sb = simulate_custom(
                arm_swing_L=sb_arm_L, arm_swing_R=sb_arm_R,
                leg_swing_L=sb_leg_L, leg_swing_R=sb_leg_R,
                foot_clear_L=sb_clear_L, foot_clear_R=sb_clear_R,
                tremor_amp=sb_tremor, lateral_jitter=sb_jitter,
                cadence_hz=sb_cad, noise_sigma=sb_noise)
            feats_sb = build_features(raw_sb[-WINDOW:])
            scaled_sb = scaler.transform(feats_sb).reshape(1, WINDOW, 24)
            p_sb = float(bin_model.predict(scaled_sb, verbose=0)[0, 0])
            mc_sb = mc_model.predict(scaled_sb, verbose=0)[0]
            mc_top_sb = int(mc_sb.argmax())

            # verdict card
            if p_sb > 0.5:
                st.error(f"🚨 **ABNORMAL** · P(abnormal) = {p_sb*100:.1f}%")
            else:
                st.success(f"✅ **NORMAL** · P(abnormal) = {p_sb*100:.1f}%")
            st.markdown(f"**Differential:** `{PROFILES[mc_top_sb]}` — "
                        f"{PROFILE_LABELS[PROFILES[mc_top_sb]]}")

            # multiclass bar
            barf = go.Figure(go.Bar(
                x=mc_sb*100, y=list(PROFILES), orientation="h",
                marker_color=["#22c55e" if i == mc_top_sb else "#3b82f6"
                              for i in range(len(PROFILES))],
                text=[f"{p*100:.1f}%" for p in mc_sb], textposition="outside"))
            barf.update_layout(height=210, margin=dict(l=10, r=30, t=10, b=10),
                xaxis=dict(range=[0, 110], color="#9ca3af"),
                yaxis=dict(color="#9ca3af"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"))
            st.plotly_chart(barf, use_container_width=True)

            # mini-skeleton preview — render as an animated GIF using the same
            # cinematic renderer (twin = the sandbox subject walking healthy)
            anom_sb = per_sensor_anomaly(raw_sb)
            # generate a "healthy reference" twin with the same cadence/noise
            twin_sb_raw, _, _ = simulate_custom(
                arm_swing_L=0.40, arm_swing_R=0.40,
                leg_swing_L=0.55, leg_swing_R=0.55,
                foot_clear_L=0.12, foot_clear_R=0.12,
                tremor_amp=0.0, lateral_jitter=0.0,
                cadence_hz=sb_cad, noise_sigma=sb_noise)
            st.markdown("**Cinematic sandbox playback** "
                        "(your custom subject vs healthy reference):")
            sb_gif = render_scene_gif_for(raw_sb, twin_sb_raw, anom_sb,
                alert_after=(0 if p_sb > 0.5 else None),
                sensor_saliency=None, fps=20)
            st.image(sb_gif, use_container_width=True)

    # ============== SWEEP ====================================
    with sub_b:
        st.markdown("**Pathology × severity sweep.** For every profile we "
                    "run the AI across 5 severity levels and chart the "
                    "binary P(abnormal). The cleaner the diagonal, the "
                    "better-calibrated the model's decision boundary.")

        @st.cache_data(show_spinner="Running 6×5 sweep…")
        def run_sweep():
            severities = [0.0, 0.25, 0.5, 0.75, 1.0]
            res = np.zeros((len(PROFILES), len(severities)))
            for i, prof in enumerate(PROFILES):
                for j, sev in enumerate(severities):
                    # 5-seed average to smooth out noise
                    probs_seeds = []
                    for seed in range(5):
                        rng = np.random.default_rng(seed * 17 + j)
                        raw_s, _, _, _ = simulate_walk(
                            prof, rng, n_frames=80,
                            onset_frame=0 if prof != "healthy" else None,
                            peak_severity=sev, ramp_frames=1)
                        feats = build_features(raw_s[-WINDOW:])
                        sc = scaler.transform(feats).reshape(1, WINDOW, 24)
                        probs_seeds.append(
                            float(bin_model.predict(sc, verbose=0)[0, 0]))
                    res[i, j] = float(np.mean(probs_seeds))
            return severities, res

        severities, sweep = run_sweep()
        hm = go.Figure(go.Heatmap(
            z=sweep, x=[f"{s:.2f}" for s in severities],
            y=list(PROFILES), colorscale="RdYlGn_r", zmin=0, zmax=1,
            text=[[f"{v*100:.0f}%" for v in row] for row in sweep],
            texttemplate="%{text}",
            hovertemplate="profile=%{y} severity=%{x} P(abn)=%{z:.2f}<extra></extra>"))
        hm.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"),
                          xaxis=dict(title="severity", color="#9ca3af"),
                          yaxis=dict(title="profile", color="#9ca3af"))
        st.plotly_chart(hm, use_container_width=True)
        st.caption("Each cell is the **mean** P(abnormal) over 5 seeds. "
                   "A well-calibrated model should be near 0 for healthy / "
                   "severity 0, rise smoothly with severity, and saturate at "
                   "1 for high-severity pathology.")

    # ============== BENCHMARK ================================
    with sub_c:
        st.markdown("**Full-cohort benchmark.** Runs the AI on all 300 "
                    "pre-generated pedestrians and reports the metrics "
                    "you'd care about in a clinical screening context.")

        @st.cache_data(show_spinner="Benchmarking all 300 scenarios…")
        def run_benchmark():
            confusion = np.zeros((len(PROFILES), len(PROFILES)), dtype=int)
            sens_correct, sens_total = 0, 0
            spec_correct, spec_total = 0, 0
            lags = []
            for s in SCENARIOS:
                rng = np.random.default_rng(s["subject_seed"])
                raw, _, _, _ = simulate_walk(
                    s["profile"], rng, n_frames=80,
                    onset_frame=s["onset_frame"],
                    peak_severity=s["severity"])
                centers, probs = sliding_window_inference(
                    raw, scaler, bin_model, build_features)
                a = alert_frame(probs, centers)
                true_abn = s["profile"] != "healthy"
                pred_abn = a is not None
                if true_abn:
                    sens_total += 1
                    if pred_abn: sens_correct += 1
                    if pred_abn and s["onset_frame"] is not None:
                        lags.append((a - s["onset_frame"]) * 0.075)
                else:
                    spec_total += 1
                    if not pred_abn: spec_correct += 1
                # multiclass (final window)
                feats_last = build_features(raw[-WINDOW:])
                sc = scaler.transform(feats_last).reshape(1, WINDOW, 24)
                pred_idx = int(mc_model.predict(sc, verbose=0)[0].argmax())
                true_idx = PROFILES.index(s["profile"])
                confusion[true_idx, pred_idx] += 1
            return dict(
                sensitivity=sens_correct / max(1, sens_total),
                specificity=spec_correct / max(1, spec_total),
                mc_accuracy=float(np.trace(confusion) / confusion.sum()),
                confusion=confusion, lags=lags,
                sens_n=sens_total, spec_n=spec_total)

        bench = run_benchmark()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sensitivity (TPR)",
                  f"{bench['sensitivity']*100:.1f}%",
                  f"n={bench['sens_n']}")
        m2.metric("Specificity (TNR)",
                  f"{bench['specificity']*100:.1f}%",
                  f"n={bench['spec_n']}")
        m3.metric("Multiclass accuracy",
                  f"{bench['mc_accuracy']*100:.1f}%",
                  f"n={int(bench['confusion'].sum())}")
        m4.metric("Median detection lag",
                  f"{np.median(bench['lags'])*1000:.0f} ms"
                  if bench["lags"] else "—",
                  f"min {np.min(bench['lags'])*1000:.0f} ms"
                  if bench["lags"] else "")

        st.subheader("Confusion matrix (multiclass)")
        cm = go.Figure(go.Heatmap(
            z=bench["confusion"], x=list(PROFILES), y=list(PROFILES),
            colorscale="Blues",
            text=bench["confusion"], texttemplate="%{text}",
            hovertemplate="true=%{y} pred=%{x} n=%{z}<extra></extra>"))
        cm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
            xaxis=dict(title="predicted", color="#9ca3af"),
            yaxis=dict(title="true", color="#9ca3af"))
        st.plotly_chart(cm, use_container_width=True)

        if bench["lags"]:
            st.subheader("Detection-lag distribution")
            lags_ms = np.array(bench["lags"]) * 1000
            hist = go.Figure(go.Histogram(
                x=lags_ms, nbinsx=30, marker_color="#22d3ee",
                marker_line=dict(color="#0e7490", width=1)))
            hist.add_vline(x=float(np.median(lags_ms)),
                line=dict(color="#fbbf24", dash="dash"),
                annotation_text=f"median {np.median(lags_ms):.0f} ms")
            hist.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(title="lag from true onset → AI alert (ms)",
                            color="#9ca3af"),
                yaxis=dict(title="scenarios", color="#9ca3af"))
            st.plotly_chart(hist, use_container_width=True)
            st.caption("Negative values mean the AI fired *before* the "
                       "ground-truth onset frame — that can happen because "
                       "the alert is the centre of a 20-frame window, so "
                       "early frames of the window may pre-date the onset.")


# ----- TAB 5: ARCHITECTURE ----------------------------------------
with tabs[4]:
    st.markdown("""
### End-to-end pipeline
```
 4 IMU bands  (2 wrists + 2 ankles)
   │   3-D positions @ ~13 Hz  ──►  (T, 12) raw stream
   ▼
 build_features                ──►  (T, 24) clinical biomarkers
   │     raw XYZ · L-R drift · limb magnitudes · upper/lower asymmetry
   ▼
 StandardScaler  (fit on training set)
   ▼
 ┌──────────────────────────── shared backbone ──────────────────────────┐
 │  Conv1D(64) → BN → Conv1D(64) → MaxPool → Dropout                     │
 │  Bidirectional LSTM(48) → Dropout → Dense(32, ReLU)                   │
 └─────────────┬──────────────────────────────────────┬──────────────────┘
        Dense(1, sigmoid)                      Dense(6, softmax)
        binary screener                        6-class differential
               │                                         │
        P(abnormal)                            P(profile | window)
                              │
                              ▼
        Dense(32) embedding → k-NN cohort retrieval (300 prior cases)
```

### Clinical layers built on top of the model

| Layer | Purpose | Source |
|---|---|---|
| **Digital twin** | Same subject anthropometrics rendered as a counterfactual healthy walker, so the deficit is visible | Patient-specific simulation |
| **Robinson Symmetry Index** | Validated bilateral asymmetry quantifier per metric | Robinson et al., *J Manipulative Physiol Ther* 1987 |
| **Spatiotemporal flags** | Cadence / speed / stride length scored against normative bands | Bohannon, *Age Ageing* 1997 |
| **Gradient saliency** | Per-limb importance from `∂P(abn)/∂x`, drives joint-size swell | Simonyan et al., 2014 |
| **k-NN cohort retrieval** | Latent-space nearest neighbours over the 300-case cohort | Case-based reasoning, classical CDSS |
| **Neurons-saved counter** | Detection-lag → avoided neuronal loss vs 30 s bystander baseline | Saver, *Stroke* 2006 — "Time is brain — quantified" |

### Held-out performance (1440-window stratified test)
- **Binary screener:** accuracy 99.5 %, ROC-AUC 0.9999
- **Multiclass differential:** accuracy 95 %, macro-F1 0.95

### Realistic gait simulator
- Subject variation: height 1.55–1.90 m · cadence 0.85–1.25 Hz · stride
  0.55–0.80 × height · arm swing 0.25–0.45 × height · IMU noise σ 0.5–2.5 cm.
- Literature-informed pathology signatures:
  - **Hemiparesis** — reduced arm swing + flexor synergy + foot drop on the
    affected side
  - **Spastic** — circumduction + reduced range on one side
  - **Parkinsonian** — global bradykinesia + shuffling + 5.5 Hz tremor
  - **Cerebellar ataxia** — high lateral variability + irregular timing

### Honest caveats
Trained on physics-informed synthetic data. Suitable as a research /
education baseline and as a transfer-learning starting point for real IMU
corpora (PhysioNet *Gait in Neurodegenerative Disease*, Daphnet FoG,
MotionSense). **Not** a diagnostic device.
""")
