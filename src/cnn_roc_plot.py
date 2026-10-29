"""Plot ROC curves from the already-saved one-time CNN test predictions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]

data = np.load(ROOT / "results" / "cnn_test_predictions.npz", allow_pickle=True)
artists = np.load(ROOT / "data" / "features.npz", allow_pickle=True)["artists"]
y_true = data["y_true"]
probabilities = data["probabilities"]
y_onehot = label_binarize(y_true, classes=np.arange(len(artists)))

curves = []
for index, artist in enumerate(artists):
    fpr, tpr, _ = roc_curve(y_onehot[:, index], probabilities[:, index])
    curves.append((fpr, tpr, auc(fpr, tpr), artist))
all_fpr = np.unique(np.concatenate([curve[0] for curve in curves]))
mean_tpr = np.zeros_like(all_fpr)
for fpr, tpr, _, _ in curves:
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= len(curves)
macro_auc = auc(all_fpr, mean_tpr)
macro_ovr_auc = roc_auc_score(y_onehot, probabilities, average="macro", multi_class="ovr")

plt.figure(figsize=(10, 8))
for fpr, tpr, artist_auc, artist in curves:
    plt.plot(fpr, tpr, linewidth=0.8, alpha=0.45, label=f"{artist} ({artist_auc:.3f})")
plt.plot(all_fpr, mean_tpr, color="black", linewidth=2.5,
         label=f"Macro average curve area={macro_auc:.3f}; mean class AUC={macro_ovr_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("One-vs-rest ROC curves: fine-tuned ResNet-50 test set")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(ROOT / "results" / "cnn_roc_auc.png", dpi=160)
plt.close()
print(f"Saved ROC curves; mean class AUC={macro_ovr_auc:.6f}, macro curve area={macro_auc:.6f}")
