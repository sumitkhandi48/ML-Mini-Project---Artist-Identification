import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

models = {"SVM": "svm", "KNN": "knn", "Decision Tree": "decision_tree"}
rows = []
for label, key in models.items():
    t = open(f"results/{key}_results.txt", encoding="utf-8").read()
    grab = lambda p: float(re.search(p, t).group(1))
    macro = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t)
    rows.append({
        "Model": label,
        "CV acc (3-fold)": grab(r"3-fold CV accuracy: ([\d.]+)"),
        "Train acc": grab(r"Train accuracy: ([\d.]+)"),
        "Val acc": grab(r"Validation accuracy: ([\d.]+)"),
        "Test acc": grab(r"Test accuracy: ([\d.]+)"),
        "Precision (macro)": float(macro.group(1)),
        "Recall (macro)": float(macro.group(2)),
        "F1 (macro)": float(macro.group(3)),
        "Train time (s)": grab(r"Training time \(s\): ([\d.]+)"),
    })

df = pd.DataFrame(rows)
df["Train-Test gap"] = (df["Train acc"] - df["Test acc"]).round(3)
df.to_csv("results/model_comparison.csv", index=False)
print(df.round(3).to_string(index=False))

# Chart 1: test accuracy and macro F1
ax = df.set_index("Model")[["Test acc", "F1 (macro)"]].plot(kind="bar", figsize=(7, 5))
ax.set_ylabel("Score"); ax.set_title("Model comparison on test set")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("results/comparison_test_scores.png", dpi=150); plt.close()

# Chart 2: train vs test accuracy (overfitting)
ax = df.set_index("Model")[["Train acc", "Test acc"]].plot(kind="bar", figsize=(7, 5))
ax.set_ylabel("Accuracy"); ax.set_title("Train vs test accuracy (overfitting)")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("results/comparison_overfitting.png", dpi=150); plt.close()
print("\nSaved results/model_comparison.csv and two charts")