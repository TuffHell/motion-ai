import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Neurological Stroke Detection", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered Hemiparesis & Stroke Detector")
st.markdown("Live 3D Biomechanical Tracking, Kinematics, & Explainable AI Diagnostics.")

# ---------------------------------------------------------
# 2. LOAD THE AI MODEL AND SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_ai_assets():
    try:
        model = load_model('stroke_model_v2.h5')
        scaler = joblib.load('scaler_v2.save')
        return model, scaler
    except (FileNotFoundError, OSError) as e:
        st.sidebar.caption(f"Model load error: {e}")
        return None, None

model, scaler = load_ai_assets()
if model:
    st.sidebar.success("✅ Neural Network Online")
else:
    st.sidebar.warning("⚠️ Demo Mode (AI Model files not detected)")

# ---------------------------------------------------------
# 3. CLINICAL PHYSICS ENGINE 
# ---------------------------------------------------------
def calculate_clinical_features(raw_12d_data):
    """Builds the 24-channel feature vector the v2 model was trained on.
    Input: (T, 12) = [L_wrist xyz, R_wrist xyz, L_ankle xyz, R_ankle xyz]."""
    l_arm = raw_12d_data[:, 0:3]
    r_arm = raw_12d_data[:, 3:6]
    l_leg = raw_12d_data[:, 6:9]
    r_leg = raw_12d_data[:, 9:12]

    arm_drift = l_arm - r_arm
    leg_drift = l_leg - r_leg

    l_arm_mag = np.linalg.norm(l_arm, axis=1, keepdims=True)
    r_arm_mag = np.linalg.norm(r_arm, axis=1, keepdims=True)
    l_leg_mag = np.linalg.norm(l_leg, axis=1, keepdims=True)
    r_leg_mag = np.linalg.norm(r_leg, axis=1, keepdims=True)

    arm_asym = (l_arm_mag - r_arm_mag) / (l_arm_mag + r_arm_mag + 1e-6)
    leg_asym = (l_leg_mag - r_leg_mag) / (l_leg_mag + r_leg_mag + 1e-6)

    return np.hstack((l_arm, r_arm, l_leg, r_leg, arm_drift, leg_drift,
                      l_arm_mag, r_arm_mag, l_leg_mag, r_leg_mag, arm_asym, leg_asym))

# ---------------------------------------------------------
# 4. PHOTOREALISTIC HUMAN GAIT (VOLUMETRIC)
# ---------------------------------------------------------
def generate_stickman_frame(t, profile, noise_level):
    cycle = t * np.pi * 2 
    bounce = 0.04 * np.sin(cycle * 2)
    
    pelvis = [0, 0, 0.9 + bounce]
    neck = [0, 0.05, 1.5 + bounce]
    head = [0, 0.1, 1.7 + bounce]
    
    s_rot = 0.05 * np.sin(cycle)
    h_rot = -0.05 * np.sin(cycle)
    
    r_shoulder = [0.25, s_rot, 1.5 + bounce]
    l_shoulder = [-0.25, -s_rot, 1.5 + bounce]
    r_hip = [0.15, h_rot, 0.9 + bounce]
    l_hip = [-0.15, -h_rot, 0.9 + bounce]

    # --- HEALTHY BASELINE KINEMATICS ---
    r_ankle_y = 0.4 * np.sin(cycle)
    r_ankle_z = 0.05 + max(0, 0.15 * np.cos(cycle)) 
    r_knee_y = 0.2 * np.sin(cycle) + 0.05
    r_knee_z = 0.5 + (r_ankle_z - 0.05) * 1.5 
    r_knee = [0.15, r_knee_y, r_knee_z]
    r_ankle = [0.15, r_ankle_y, r_ankle_z]
    
    l_ankle_y = 0.4 * np.sin(cycle + np.pi)
    l_ankle_z = 0.05 + max(0, 0.15 * np.cos(cycle + np.pi))
    l_knee_y = 0.2 * np.sin(cycle + np.pi) + 0.05
    l_knee_z = 0.5 + (l_ankle_z - 0.05) * 1.5
    l_knee = [-0.15, l_knee_y, l_knee_z]
    l_ankle = [-0.15, l_ankle_y, l_ankle_z]
    
    r_wrist_y = 0.4 * np.sin(cycle + np.pi)
    r_wrist_z = 0.9 + 0.1 * np.cos(cycle + np.pi) 
    r_elbow_y = 0.2 * np.sin(cycle + np.pi)
    r_elbow_z = 1.2
    r_elbow = [0.3, r_elbow_y, r_elbow_z]
    r_wrist = [0.3, r_wrist_y, r_wrist_z]

    l_wrist_y = 0.4 * np.sin(cycle)
    l_wrist_z = 0.9 + 0.1 * np.cos(cycle)
    l_elbow_y = 0.2 * np.sin(cycle)
    l_elbow_z = 1.2
    l_elbow = [-0.3, l_elbow_y, l_elbow_z]
    l_wrist = [-0.3, l_wrist_y, l_wrist_z]

    # --- APPLY CLINICAL PATHOLOGY PROFILES ---
    if profile == "Severe Left Hemiparesis":
        noise = np.random.normal(0, noise_level, 3)
        l_elbow = [-0.2, 0.1 + noise[0], 1.25] 
        l_wrist = [-0.1, 0.15 + noise[1], 1.35]  
        l_knee = [-0.15, 0.1 * np.sin(cycle + np.pi), 0.45] 
        l_ankle_x = -0.15 - max(0, 0.2 * np.cos(cycle + np.pi)) 
        l_ankle = [l_ankle_x, 0.2 * np.sin(cycle + np.pi), 0.05] 

    elif profile == "Mild Right Spasticity":
        noise = np.random.normal(0, noise_level * 2, 3)
        r_elbow = [0.3, 0.1 * np.sin(cycle + np.pi) + noise[0], 1.2]
        r_wrist = [0.3, 0.15 * np.sin(cycle + np.pi) + noise[1], 1.0]
        r_knee = [0.15, 0.15 * np.sin(cycle) + noise[0], 0.48]
        r_ankle = [0.15, 0.3 * np.sin(cycle), 0.06]

    elif profile == "Bilateral Bradykinesia (Parkinsonian)":
        noise = np.random.normal(0, noise_level * 1.5, 3)
        neck = [0, 0.2, 1.4]
        head = [0, 0.3, 1.55]
        r_ankle = [0.15, 0.1 * np.sin(cycle), 0.05]
        l_ankle = [-0.15, 0.1 * np.sin(cycle + np.pi), 0.05]
        r_knee = [0.15, 0.05 * np.sin(cycle) + 0.1, 0.45]
        l_knee = [-0.15, 0.05 * np.sin(cycle + np.pi) + 0.1, 0.45]
        r_elbow = [0.3, 0.05 * np.sin(cycle + np.pi) + noise[0], 1.1]
        r_wrist = [0.3, 0.05 * np.sin(cycle + np.pi) + noise[1], 0.95]
        l_elbow = [-0.3, 0.05 * np.sin(cycle) + noise[0], 1.1]
        l_wrist = [-0.3, 0.05 * np.sin(cycle) + noise[1], 0.95]

    # --- NEW VOLUMETRIC PATHING ---
    # By drawing a complete "box" for the torso, it stops glitching and looks solid.
    x = [
        head[0], neck[0], # Head to Neck
        l_shoulder[0], r_shoulder[0], r_hip[0], l_hip[0], l_shoulder[0], # TORSO BOX
        None, neck[0], pelvis[0], # Inner Spine
        None, r_shoulder[0], r_elbow[0], r_wrist[0], # Right Arm
        None, l_shoulder[0], l_elbow[0], l_wrist[0], # Left Arm
        None, r_hip[0], r_knee[0], r_ankle[0], # Right Leg
        None, l_hip[0], l_knee[0], l_ankle[0]  # Left Leg
    ]
    
    y = [
        head[1], neck[1],
        l_shoulder[1], r_shoulder[1], r_hip[1], l_hip[1], l_shoulder[1],
        None, neck[1], pelvis[1],
        None, r_shoulder[1], r_elbow[1], r_wrist[1],
        None, l_shoulder[1], l_elbow[1], l_wrist[1],
        None, r_hip[1], r_knee[1], r_ankle[1],
        None, l_hip[1], l_knee[1], l_ankle[1]
    ]
    
    z = [
        head[2], neck[2],
        l_shoulder[2], r_shoulder[2], r_hip[2], l_hip[2], l_shoulder[2],
        None, neck[2], pelvis[2],
        None, r_shoulder[2], r_elbow[2], r_wrist[2],
        None, l_shoulder[2], l_elbow[2], l_wrist[2],
        None, r_hip[2], r_knee[2], r_ankle[2],
        None, l_hip[2], l_knee[2], l_ankle[2]
    ]
    
    return x, y, z, l_wrist, r_wrist, l_ankle, r_ankle

def build_animated_stickman(frames_data, color_scheme):
    fig = go.Figure(
        # Added thick lines (width=8) and spherical joints (markers size=6) for a "meaty" look
        data=[go.Scatter3d(
            x=frames_data[0][0], y=frames_data[0][1], z=frames_data[0][2], 
            mode='lines+markers', 
            marker=dict(size=6, color=color_scheme, opacity=0.9),
            line=dict(color=color_scheme, width=8)
        )],
        layout=go.Layout(
            scene=dict(xaxis=dict(range=[-0.8, 0.8]), yaxis=dict(range=[-0.8, 0.8]), zaxis=dict(range=[0, 2]), aspectmode='cube'),
            # Force transition duration to 0 to prevent Plotly from glitching the None-gaps
            updatemenus=[dict(type="buttons", buttons=[dict(label="▶ Play Kinematics", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, transition=dict(duration=0))])])],
            margin=dict(l=0, r=0, b=0, t=0), height=450
        ),
        frames=[go.Frame(data=[go.Scatter3d(x=f[0], y=f[1], z=f[2])]) for f in frames_data]
    )
    return fig

# ---------------------------------------------------------
# 5. DASHBOARD CONTROLS (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("📡 Live Telemetry Control")
patient_profiles = ("Healthy Control", "Severe Left Hemiparesis", "Mild Right Spasticity", "Bilateral Bradykinesia (Parkinsonian)")
patient_type = st.sidebar.radio("Select Patient Profile:", patient_profiles)

st.sidebar.divider()
st.sidebar.subheader("Simulation Parameters")
num_frames = st.sidebar.slider("Sampling Resolution (Frames)", min_value=40, max_value=120, value=80, step=10)
noise_level = st.sidebar.slider("Neurological Tremor Intensity", min_value=0.0, max_value=0.15, value=0.03, step=0.01)

if st.sidebar.button("▶️ Initialize AI Biomechanical Scan"):
    with st.spinner(f"Rendering {num_frames} High-Fidelity Kinematic Frames..."):
        
        time_steps = np.linspace(0, 2, num_frames)
        healthy_frames = [generate_stickman_frame(t, "Healthy Control", 0) for t in time_steps]
        patient_frames = [generate_stickman_frame(t, patient_type, noise_level) for t in time_steps]
        
        raw_data = np.array([np.concatenate([f[3], f[4], f[5], f[6]]) for f in patient_frames])
        features_24d = calculate_clinical_features(raw_data)

        dt = time_steps[1] - time_steps[0]
        left_wrist_y_vel = np.gradient(raw_data[:, 1], dt)
        left_wrist_y_acc = np.gradient(left_wrist_y_vel, dt)
        right_wrist_y_vel = np.gradient(raw_data[:, 4], dt)
        right_wrist_y_acc = np.gradient(right_wrist_y_vel, dt)

        indices = np.linspace(0, num_frames - 1, 20, dtype=int)
        scan_window = features_24d[indices]

        prediction_prob = 0.0
        is_stroke = False

        if model is not None and scaler is not None:
            scaled_features = scaler.transform(scan_window).reshape(1, 20, 24)
            prediction_prob = float(model.predict(scaled_features, verbose=0)[0][0])
            is_stroke = prediction_prob > 0.5
        else:
            if patient_type == "Healthy Control":
                prediction_prob = 0.04
            elif patient_type == "Severe Left Hemiparesis":
                prediction_prob = 0.98
                is_stroke = True
            elif patient_type == "Mild Right Spasticity":
                prediction_prob = 0.76
                is_stroke = True
            else:
                prediction_prob = 0.62
                is_stroke = True

        # ---------------------------------------------------------
        # 6. UI TABS: CLINICAL, XAI, AND ARCHITECTURE
        # ---------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["🩺 Clinical View (3D)", "⚙️ Live AI Diagnostics", "📚 The Science (Architecture)"])
        
        # --- TAB 1: CLINICAL VIEW ---
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Neural Network Confidence", f"{prediction_prob * 100:.2f}%")
            if is_stroke:
                col2.error(f"🚨 ALERT: Abnormal Gait Detected ({patient_type})")
            else:
                col2.success("✅ STATUS: Healthy Symmetric Gait")
            col3.metric("Peak Lateral Balance Shift", f"{np.max(np.abs(features_24d[:, 22])):.3f}")

            st.subheader("Live 3D Spatial Tracking (60 FPS)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Baseline (Healthy Reference)**")
                st.plotly_chart(build_animated_stickman(healthy_frames, '#00CC96'), use_container_width=True) 
            with c2:
                st.markdown(f"**Live Patient ({patient_type})**")
                patient_color = '#EF553B' if is_stroke else '#00CC96' 
                st.plotly_chart(build_animated_stickman(patient_frames, patient_color), use_container_width=True)

        # --- TAB 2: EXPLAINABLE AI ---
        with tab2:
            st.markdown("### Deep Learning & Biomechanical Breakdown")
            
            st.subheader("1. CNN Tremor Detection: High-Frequency Acceleration (m/s²)")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=left_wrist_y_acc, mode='lines', name='Left Arm Acceleration', line=dict(color='red' if is_stroke and "Left" in patient_type else 'cyan', width=2)))
            fig_acc.add_trace(go.Scatter(y=right_wrist_y_acc, mode='lines', name='Right Arm Acceleration', line=dict(color='orange' if is_stroke and "Right" in patient_type else 'lightgreen', width=2)))
            fig_acc.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_acc, use_container_width=True)

            col_x1, col_x2 = st.columns(2)
            with col_x1:
                st.subheader("2. CNN Spasticity Filter (Amplitude)")
                if model:
                    fig_heat = px.imshow(scaled_features[0].T, color_continuous_scale="inferno", aspect="auto")
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    fake_heat = np.random.rand(24, 20) if not is_stroke else np.random.rand(24, 20) * 3
                    fig_heat = px.imshow(fake_heat, color_continuous_scale="inferno", aspect="auto")
                    st.plotly_chart(fig_heat, use_container_width=True)
                
            with col_x2:
                st.subheader("3. LSTM Bradykinesia Filter (Rhythm)")
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(y=features_24d[:, 18], mode='lines', name='Left Arm Energy', line=dict(color='red' if is_stroke and "Left" in patient_type else 'cyan', width=3)))
                fig_lstm.add_trace(go.Scatter(y=features_24d[:, 19], mode='lines', name='Right Arm Energy', line=dict(color='orange' if is_stroke and "Right" in patient_type else 'lightgreen', width=3, dash='dot')))
                fig_lstm.update_layout(height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig_lstm, use_container_width=True)

        # --- TAB 3: THE SCIENCE ---
        with tab3:
            st.header("🧠 AI Architecture & Clinical Methodology")
            st.markdown("The AI analyzes **480 data points** per prediction (24 features × 20 timesteps).")
            with st.expander("🔬 The 24 Clinical Biomarkers", expanded=True):
                st.markdown("""
                `calculate_clinical_features` turns the 4 raw sensor streams (left/right wrist + left/right ankle) into 24 channels:
                * **Raw Kinematics (0-11):** XYZ position of both wrists and both ankles.
                * **Positional Drift (12-17):** Arm drift (`l_arm - r_arm`) and leg drift (`l_leg - r_leg`).
                * **Kinetic Energy (18-21):** Magnitudes of left/right arm and left/right leg.
                * **Lateral Balance Shift (22-23):** Upper- and lower-extremity asymmetry indices.
                """)
            with st.expander("🧬 AI Architecture: Hybrid CNN-LSTM", expanded=True):
                st.markdown("""
                The Hybrid CNN-LSTM network scans across 1.5 seconds of continuous motion, hunting for three clinical stroke signatures:
                * **Bradykinesia:** Evaluated by the LSTM layer.
                * **Spasticity:** Evaluated by the CNN layer.
                * **Acute Tremors:** High-frequency anomaly detection via CNN.
                """)
else:
    st.info("👈 Select a Patient Profile and click **Initialize AI Biomechanical Scan**.")