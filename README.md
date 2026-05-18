# Motion-AI — Clinical Gait Pathology Detector

A continuous gait-screening system that takes 3-D positions from four IMU bands
(two wrists + two ankles) and runs a hybrid Conv1D + Bidirectional-LSTM neural
network in real time. Two heads share the backbone: a **binary screener**
(P(abnormal gait)) and a **six-class differential** (healthy, left/right
hemiparesis, spastic, parkinsonian, cerebellar ataxic).

## Features

- **Cinematic 2-D sagittal animation** of patient + healthy digital twin walking
  side-by-side, rendered via matplotlib as a looping GIF.
- **Clinical report card** — Robinson Symmetry Index, cadence, walking speed,
  range of motion, foot clearance, with normative bands and color flags.
- **Gradient saliency** — yellow glow rings highlight which limbs drove the
  alert, computed via TensorFlow GradientTape.
- **k-NN cohort retrieval** — retrieves the 5 most similar cases from a
  300-pedestrian cohort using the model's own Dense(32) latent space.
- **Time-is-brain accounting** — converts detection latency into "neurons
  saved" against a 30-second bystander baseline (Saver 2006).
- **Test the AI sandbox** — 10 live sliders, pathology sweep heatmap, full
  300-scenario benchmark with confusion matrix and detection-lag histogram.

## Deployment

This repo is ready for [Streamlit Community Cloud](https://share.streamlit.io).
Main entry point: **`app2.py`**.

### Local run

```bash
pip install -r requirements.txt
streamlit run app2.py
```

### Retrain the model

```bash
python train_model.py
```

Outputs `stroke_model_v2.h5`, `motion_multiclass.h5`, `scaler_v2.save`,
`training_report.txt`.

## File layout

| File | Purpose |
| --- | --- |
| `app2.py` | Streamlit dashboard (entry point) |
| `gait_sim.py` | Core gait simulator and 24-channel feature engineering |
| `scenarios.py` | Long-form scenarios, skeleton renderer, clinical metrics, saliency |
| `train_model.py` | End-to-end training script |
| `stroke_model_v2.h5` | Trained binary screener |
| `motion_multiclass.h5` | Trained 6-class differential head |
| `scaler_v2.save` | Fitted StandardScaler |
| `training_report.txt` | Held-out classification metrics |
| `Motion-AI_Technical_Specification.docx` | Full technical spec |

## Honest caveats

Trained on physics-informed **synthetic data**. Suitable as a research /
education baseline and as a transfer-learning starting point for real IMU
corpora (PhysioNet Gait-in-Neurodegenerative-Disease, Daphnet Freezing of
Gait). **Not** a diagnostic device.
