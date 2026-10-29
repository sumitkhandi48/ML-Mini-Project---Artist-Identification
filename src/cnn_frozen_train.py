"""Tune classifiers on frozen CNN features using train/validation only."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import split_data

ROOT = Path(__file__).resolve().parents[1]


def append_experiment(row):
    detailed = ROOT / "results" / "cnn_experiments.csv"
    detailed.parent.mkdir(exist_ok=True)
    fields = ["name", "backbone", "trainable_layers", "params", "epochs",
              "train_acc", "val_acc", "time_s"]
    exists = detailed.exists()
    with detailed.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    legacy = ROOT / "results" / "experiments.csv"
    with legacy.open("a", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([row["name"], "", "", "StandardScaler", row["params"], "",
                         "", f'{row["val_acc"]:.6f}', f'{row["train_acc"]:.6f}',
                         f'{row["time_s"]:.2f}'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet50", "convnext_tiny", "efficientnet_b0"],
                        default="resnet50")
    args = parser.parse_args()
    path = ROOT / "data" / f"cnn_features_{args.backbone}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run cnn_features.py first: {path}")
    data = np.load(path, allow_pickle=True)
    X_tr, X_va, _, y_tr, y_va, _ = split_data(data["X"], data["y"])
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
    candidates = []
    for c in (0.1, 1.0, 10.0, 100.0):
        candidates.append((f"cnn_frozen_{args.backbone}_logreg_C{c:g}",
                          LogisticRegression(C=c, max_iter=1000, random_state=42,
                                             n_jobs=-1), {"C": c}))
    for c in (1.0, 10.0, 100.0):
        candidates.append((f"cnn_frozen_{args.backbone}_svm_C{c:g}",
                          SVC(C=c, gamma="scale", probability=True, random_state=42),
                          {"C": c, "gamma": "scale"}))

    best = None
    for name, model, params in candidates:
        start = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - start
        train_acc = accuracy_score(y_tr, model.predict(X_tr))
        val_acc = accuracy_score(y_va, model.predict(X_va))
        row = {"name": name, "backbone": args.backbone, "trainable_layers": "none",
               "params": json.dumps(params), "epochs": 0, "train_acc": train_acc,
               "val_acc": val_acc, "time_s": elapsed}
        append_experiment(row)
        print(name, f"train={train_acc:.4f}", f"val={val_acc:.4f}", f"time={elapsed:.1f}s")
        if best is None or val_acc > best[0]:
            best = (val_acc, name, model)
    assert best is not None
    joblib.dump({"scaler": scaler, "model": best[2], "artists": data["artists"],
                 "backbone": args.backbone}, ROOT / "models" / f"cnn_frozen_best_{args.backbone}.joblib")
    print(f"Best validation accuracy: {best[0]:.4f} ({best[1]}); test was not evaluated.")


if __name__ == "__main__":
    main()
