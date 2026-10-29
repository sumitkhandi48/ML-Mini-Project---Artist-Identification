"""Train and evaluate the retained feature configuration exactly once on test data."""

import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, top_k_accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import split_data


def main():
    base = np.load("data/features.npz", allow_pickle=True)
    spatial = np.load("data/features_spatial.npz", allow_pickle=True)
    if not np.array_equal(base["files"], spatial["files"]):
        raise ValueError("Base and spatial caches are not aligned")
    X = np.hstack([base["X"], spatial["X"]])
    y, artists = base["y"], base["artists"]
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)

    scaler = StandardScaler().fit(X_tr)
    A_tr, A_va, A_te = (scaler.transform(part) for part in (X_tr, X_va, X_te))
    pca = PCA(n_components=300, random_state=42).fit(A_tr)
    P_tr, P_va, P_te = (pca.transform(part) for part in (A_tr, A_va, A_te))

    models = {
        "final_svm": SVC(C=100, gamma="scale", kernel="rbf", probability=True),
        "final_knn": KNeighborsClassifier(n_neighbors=3, weights="distance"),
        "final_decision_tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=8, min_samples_leaf=5, random_state=42
        ),
    }
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    rows = []
    for name, model in models.items():
        start = time.time()
        model.fit(P_tr, y_tr)
        train_time = time.time() - start
        pred_tr, pred_va, pred_te = model.predict(P_tr), model.predict(P_va), model.predict(P_te)
        test_prob = model.predict_proba(P_te) if hasattr(model, "predict_proba") else None
        test_acc = accuracy_score(y_te, pred_te)
        top3 = top_k_accuracy_score(y_te, test_prob, k=3, labels=np.arange(len(artists)))
        report = (
            f"Model: {name}\nFeatures: base + spatial HSV/LBP\nPCA components: 300\n"
            f"Parameters: {model.get_params()}\nTrain accuracy: {accuracy_score(y_tr, pred_tr):.3f}\n"
            f"Validation accuracy: {accuracy_score(y_va, pred_va):.3f}\n"
            f"Test accuracy: {test_acc:.3f}\nTop-3 test accuracy: {top3:.3f}\n"
            f"Training time (s): {train_time:.2f}\n\n"
            + classification_report(y_te, pred_te, target_names=artists, digits=3)
        )
        print(report)
        with open(f"results/{name}_results.txt", "w", encoding="utf-8") as output:
            output.write(report)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_te, pred_te), annot=True, fmt="d", cmap="Blues",
                    xticklabels=artists, yticklabels=artists)
        plt.xlabel("Predicted artist")
        plt.ylabel("True artist")
        plt.title(f"{name} confusion matrix")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"results/{name}_confusion_matrix.png", dpi=150)
        plt.close()
        rows.append((name, accuracy_score(y_tr, pred_tr), accuracy_score(y_va, pred_va),
                     test_acc, top3, train_time))

        if name == "final_svm":
            joblib.dump({"scaler": scaler, "pca": pca, "model": model,
                         "artists": artists}, "models/svm.joblib")
            joblib.dump({"scaler": scaler, "pca": pca, "model": model,
                         "artists": artists}, "models/svm_final.joblib")

    with open("results/final_model_comparison.csv", "w", encoding="utf-8") as output:
        output.write("Model,Train acc,Val acc,Test acc,Top-3 test acc,Train time (s)\n")
        for row in rows:
            output.write(",".join([row[0], *(f"{value:.3f}" for value in row[1:])]) + "\n")


if __name__ == "__main__":
    main()
