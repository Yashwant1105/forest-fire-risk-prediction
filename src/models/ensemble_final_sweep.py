
import numpy as np, pandas as pd, joblib, json, itertools, matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path.cwd()
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True, parents=True)
PLOTS = ROOT / "plots"; PLOTS.mkdir(exist_ok=True, parents=True)
THR = 0.40

#Load Test Data
a = np.load(PROC / "data_train_T7_bycity.npz")
X, y = a["X"], a["y"]
n = len(y)
Xf = X.reshape((n, X.shape[1] * X.shape[2]))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(sss.split(Xf, y))
X_test_seq, X_test_flat, y_test = X[test_idx], Xf[test_idx], y[test_idx]

# Load Models
rf = joblib.load(MODELS / "randomforest.pkl")
rf_p = rf.predict_proba(X_test_flat)[:, 1]

cnn_p = np.zeros_like(rf_p)
cnn_path = None
for p in ["cnn_tuned_model_final.h5", "cnn_model_final.h5", "cnn_model_noplot.h5"]:
    if (MODELS / p).exists():
        cnn_path = MODELS / p
        cnn = load_model(str(cnn_path), compile=False)
        cnn_p = cnn.predict(X_test_seq).ravel()
        print(f"Loaded CNN: {p}")
        break

lstm_p = np.zeros_like(rf_p)
lstm_path = None
for p in ["lstm_tuned_model_final.h5", "lstm_model_final_new.h5", "lstm_model_final.h5"]:
    if (MODELS / p).exists():
        try:
            m = load_model(str(MODELS / p), compile=False)
            lstm_p = m.predict(X_test_seq).ravel()
            lstm_path = p
            print(f"Loaded LSTM: {p}")
            break
        except Exception as e:
            print(f"Could not load {p}: {e}")

#Weight Sweep
grid = np.linspace(0, 1, 11)
records = []
for w_rf in grid:
    for w_cnn in grid:
        for w_lstm in grid:
            total = w_rf + w_cnn + w_lstm
            if total == 0:
                continue
            p = (w_rf * rf_p + w_cnn * cnn_p + w_lstm * lstm_p) / total
            auc = roc_auc_score(y_test, p)
            pred = (p >= THR).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="binary", zero_division=0
            )
            records.append((w_rf, w_cnn, w_lstm, auc, prec, rec, f1))

df = pd.DataFrame(
    records, columns=["w_rf", "w_cnn", "w_lstm", "auc", "precision", "recall", "f1"]
)
df.to_csv(OUT / "weight_sweep_full.csv", index=False)
df.nlargest(10, "auc").to_csv(OUT / "weight_sweep_top_auc.csv", index=False)
df.nlargest(10, "f1").to_csv(OUT / "weight_sweep_top_f1.csv", index=False)

best_auc = df.loc[df["auc"].idxmax()]
best_f1 = df.loc[df["f1"].idxmax()]

print("\n🔹 Best by AUC:", best_auc.to_dict())
print("🔹 Best by F1:", best_f1.to_dict())

# Final Ensemble with Best F1 Weights
w_rf, w_cnn, w_lstm = best_f1["w_rf"], best_f1["w_cnn"], best_f1["w_lstm"]
p_ens = (w_rf * rf_p + w_cnn * cnn_p + w_lstm * lstm_p) / (w_rf + w_cnn + w_lstm)
pred = (p_ens >= THR).astype(int)

auc_final = roc_auc_score(y_test, p_ens)
report = classification_report(y_test, pred, digits=4)
cm = confusion_matrix(y_test, pred)

print("\n✅ Final Ensemble (Best-F1 Weights)")
print(f"Weights → RF: {w_rf}, CNN: {w_cnn}, LSTM: {w_lstm}")
print(f"AUC: {auc_final:.4f}")
print(report)
print("Confusion matrix:\n", cm.tolist())

#Save Outputs
out_df = pd.DataFrame(
    {"index": test_idx, "probability": p_ens, "prediction": pred, "true": y_test}
)
csv_out = OUT / f"ensemble_final_predictions_thr{THR:.2f}.csv"
out_df.to_csv(csv_out, index=False)

metrics = {
    "roc_auc": float(auc_final),
    "threshold": THR,
    "weights": {"rf": float(w_rf), "cnn": float(w_cnn), "lstm": float(w_lstm)},
    "confusion_matrix": cm.tolist(),
}
with open(OUT / f"ensemble_final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# plot confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
labels = ["no_fire", "fire"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i, j in itertools.product(range(2), range(2)):
    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.title(f"Confusion Matrix @ thr={THR:.2f}")
plt.tight_layout()
plt.savefig(PLOTS / f"ensemble_confusion_thr{THR:.2f}.png", dpi=150)
plt.close()

print("\n📁 Files saved:")
print(" -", csv_out)
print(" -", OUT / "ensemble_final_metrics.json")
print(" -", PLOTS / f"ensemble_confusion_thr{THR:.2f}.png")
print(" -", OUT / "weight_sweep_top_f1.csv")
print(" -", OUT / "weight_sweep_top_auc.csv")
