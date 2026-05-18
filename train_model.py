"""Train the CNN-LSTM motion-AI on realistic synthetic gait data.

Outputs:
  stroke_model_v2.h5  - binary (healthy vs abnormal) CNN-LSTM
  scaler_v2.save      - StandardScaler fit on the 24 biomarker channels
  motion_multiclass.h5- 6-class profile classifier (auxiliary head)
  training_report.txt - held-out metrics
"""
import os, json, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from gait_sim import make_dataset, WINDOW, N_FEAT, PROFILES

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------- data ----------
print("Generating dataset...")
X, _Xraw, y_bin, y_multi = make_dataset(n_per_class=1200, seed=SEED)
print(f"  X={X.shape} y_bin balance={np.bincount(y_bin)} y_multi={np.bincount(y_multi)}")

# scaler fit on flattened biomarker space (T*N samples × 24 features)
scaler = StandardScaler().fit(X.reshape(-1, N_FEAT))
Xs = scaler.transform(X.reshape(-1, N_FEAT)).reshape(X.shape)

# split (stratified on the multi-class label to keep all profiles in each split)
Xtr, Xte, ytr_b, yte_b, ytr_m, yte_m = train_test_split(
    Xs, y_bin, y_multi, test_size=0.2, stratify=y_multi, random_state=SEED)
Xtr, Xva, ytr_b, yva_b, ytr_m, yva_m = train_test_split(
    Xtr, ytr_b, ytr_m, test_size=0.15, stratify=ytr_m, random_state=SEED)
print(f"  train={Xtr.shape} val={Xva.shape} test={Xte.shape}")


# ---------- model ----------
def build_binary():
    inp = layers.Input(shape=(WINDOW, N_FEAT))
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return m


def build_multiclass(n_classes):
    inp = layers.Input(shape=(WINDOW, N_FEAT))
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


cb = [callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
      callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss")]

# ---------- binary ----------
print("\nTraining binary (healthy vs abnormal)...")
bin_model = build_binary()
bin_model.fit(Xtr, ytr_b, validation_data=(Xva, yva_b),
              epochs=40, batch_size=64, callbacks=cb, verbose=2)
prob = bin_model.predict(Xte, verbose=0).ravel()
pred = (prob > 0.5).astype(int)
auc = roc_auc_score(yte_b, prob)
print("\n=== BINARY TEST ===")
print(classification_report(yte_b, pred, target_names=["healthy", "abnormal"]))
print(f"ROC-AUC: {auc:.4f}")
print("Confusion:\n", confusion_matrix(yte_b, pred))
bin_model.save("stroke_model_v2.h5")
joblib.dump(scaler, "scaler_v2.save")

# ---------- multiclass ----------
print("\nTraining multiclass (6 gait profiles)...")
mc_model = build_multiclass(len(PROFILES))
mc_model.fit(Xtr, ytr_m, validation_data=(Xva, yva_m),
             epochs=40, batch_size=64, callbacks=cb, verbose=2)
mc_prob = mc_model.predict(Xte, verbose=0)
mc_pred = mc_prob.argmax(1)
print("\n=== MULTICLASS TEST ===")
print(classification_report(yte_m, mc_pred, target_names=PROFILES))
print("Confusion:\n", confusion_matrix(yte_m, mc_pred))
mc_model.save("motion_multiclass.h5")

with open("training_report.txt", "w") as f:
    f.write("Binary classification report:\n")
    f.write(classification_report(yte_b, pred, target_names=["healthy", "abnormal"]))
    f.write(f"\nROC-AUC: {auc:.4f}\n")
    f.write(f"Confusion:\n{confusion_matrix(yte_b, pred)}\n\n")
    f.write("Multiclass classification report:\n")
    f.write(classification_report(yte_m, mc_pred, target_names=list(PROFILES)))
    f.write(f"\nConfusion:\n{confusion_matrix(yte_m, mc_pred)}\n")
print("\n✓ saved stroke_model_v2.h5, scaler_v2.save, motion_multiclass.h5, training_report.txt")
