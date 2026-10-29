"""Validation-only Step 1 experiments for classical artist recognition."""

import argparse
import csv
import os
import time

import cv2
import joblib
import numpy as np
from skimage.filters import gabor
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import load_data, split_data

IMAGE_SIZE = (256, 256)
RESULTS = "results/experiments.csv"
LOG_FIELDS = [
    "name", "n_features", "PCA", "scaler", "params", "CV mean", "CV std",
    "val acc", "train acc", "time",
]
GIST_CACHE = "data/features_gist_v2.npz"
TEXTURE_CACHE = "data/features_texture_v2.npz"


def _histogram(cell, channel, bins, value_range):
    values = cv2.calcHist([cell], [channel], None, [bins], value_range).ravel()
    total = values.sum()
    return values / total if total else values


def _image_path(artists, labels, files, index):
    return os.path.join("data", "images", str(artists[labels[index]]), str(files[index]))


def _gist(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    image = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0
    values = []
    # A GIST-like descriptor: multiscale/orientation Gabor energies pooled on 4x4.
    for frequency in (0.1, 0.2, 0.3, 0.4):
        for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            real, imag = gabor(image, frequency=frequency, theta=theta)
            energy = np.hypot(real, imag)
            for row in range(4):
                for column in range(4):
                    values.append(energy[row * 64:(row + 1) * 64,
                                         column * 64:(column + 1) * 64].mean())
    return np.asarray(values, dtype=np.float32)


def _texture(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = []
    edges = np.linspace(0, 256, 5, dtype=int)
    # Multi-scale uniform LBP plus the same Gabor energy bank as GIST.
    for radius in (1, 2, 3):
        lbp = local_binary_pattern(gray, 8 * radius, radius, method="uniform")
        for row in range(4):
            for column in range(4):
                cell = lbp[edges[row]:edges[row + 1], edges[column]:edges[column + 1]]
                counts = np.bincount(cell.astype(int).ravel(),
                                     minlength=8 * radius + 2).astype(float)
                values.extend(counts / counts.sum())
    values.extend(_gist(path))
    return np.asarray(values, dtype=np.float32)


def _build_cache(path, extractor, files, artists, labels):
    rows = []
    for index in range(len(files)):
        rows.append(extractor(_image_path(artists, labels, files, index)))
        if (index + 1) % 500 == 0:
            print(f"{os.path.basename(path)}: {index + 1}/{len(files)}")
    np.savez(path, X=np.asarray(rows), files=files, artists=artists, y=labels)
    return np.asarray(rows)


def _load_or_build_cache(path, extractor, files, artists, labels):
    if os.path.exists(path):
        cached = np.load(path, allow_pickle=True)
        if (np.array_equal(cached["files"], files)
                and np.array_equal(cached["artists"], artists)
                and np.array_equal(cached["y"], labels)):
            return cached["X"]
    return _build_cache(path, extractor, files, artists, labels)


def _append_log(row):
    os.makedirs("results", exist_ok=True)
    old_rows = []
    if os.path.exists(RESULTS):
        with open(RESULTS, newline="", encoding="utf-8-sig") as source:
            for old in csv.DictReader(source):
                old_rows.append({
                    "name": old.get("name", old.get("experiment", "")),
                    "n_features": old.get("n_features", old.get("features", "")),
                    "PCA": old.get("PCA", old.get("PCA components", "")),
                    "scaler": old.get("scaler", ""),
                    "params": old.get("params", old.get("model params",
                                                        old.get("model parameters", ""))),
                    "CV mean": old.get("CV mean", old.get("CV accuracy", "")),
                    "CV std": old.get("CV std", ""),
                    "val acc": old.get("val acc", old.get("validation accuracy", "")),
                    "train acc": old.get("train acc", old.get("train accuracy", "")),
                    "time": old.get("time", ""),
                })
    with open(RESULTS, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def _pipeline(pca, estimator):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca, random_state=42)),
        ("model", estimator),
    ])


def run_svm(name, features, labels, pca=300, grid=None, augment=False):
    train, val, _, y_train, y_val, _ = split_data(features, labels)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    if grid is None:
        grid = {"model__C": [100], "model__gamma": ["scale"]}
    search = GridSearchCV(
        _pipeline(pca, SVC(kernel="rbf", probability=True, random_state=42)),
        grid, cv=cv, n_jobs=-1, return_train_score=False,
    )
    start = time.time()
    # Augmentation is applied only after the original train/validation split.
    fit_train, fit_labels = train, y_train
    if augment:
        rng = np.random.default_rng(42)
        augmented = train.copy()
        # Feature-space duplication is deliberately restricted to train rows;
        # image-level augmentation is performed in the dedicated cache script.
        noise = rng.normal(0, 0.01, size=augmented.shape).astype(augmented.dtype)
        fit_train = np.vstack([train, augmented + noise])
        fit_labels = np.concatenate([y_train, y_train])
    search.fit(fit_train, fit_labels)
    elapsed = time.time() - start
    best_index = search.best_index_
    fitted = search.best_estimator_
    row = {
        "name": name, "n_features": features.shape[1], "PCA": pca,
        "scaler": "StandardScaler", "params": search.best_params_,
        "CV mean": f"{search.cv_results_['mean_test_score'][best_index]:.4f}",
        "CV std": f"{search.cv_results_['std_test_score'][best_index]:.4f}",
        "val acc": f"{fitted.score(val, y_val):.4f}",
        "train acc": f"{fitted.score(fit_train, fit_labels):.4f}",
        "time": f"{elapsed:.2f}",
    }
    _append_log(row)
    print(row)
    return row, fitted


def run_ensemble(name, features, labels):
    train, val, _, y_train, y_val, _ = split_data(features, labels)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    estimators = [
        ("svm", SVC(C=100, gamma="scale", probability=True, random_state=42)),
        ("logreg", LogisticRegression(C=1, max_iter=1000,
                                      random_state=42, n_jobs=-1)),
        ("rf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                      random_state=42, n_jobs=-1)),
    ]
    model = _pipeline(300, VotingClassifier(estimators=estimators,
                                             voting="soft", weights=[2, 1, 1]))
    start = time.time()
    scores = []
    for fold_train, fold_val in cv.split(train, y_train):
        fold_model = _pipeline(300, VotingClassifier(
            estimators=[(label, estimator.__class__(**estimator.get_params()))
                        for label, estimator in estimators],
            voting="soft", weights=[2, 1, 1]))
        fold_model.fit(train[fold_train], y_train[fold_train])
        scores.append(fold_model.score(train[fold_val], y_train[fold_val]))
    model.fit(train, y_train)
    elapsed = time.time() - start
    row = {
        "name": name, "n_features": features.shape[1], "PCA": 300,
        "scaler": "StandardScaler",
        "params": {"weights": [2, 1, 1], "SVM_C": 100, "LR_C": 1},
        "CV mean": f"{np.mean(scores):.4f}", "CV std": f"{np.std(scores):.4f}",
        "val acc": f"{model.score(val, y_val):.4f}",
        "train acc": f"{model.score(train, y_train):.4f}",
        "time": f"{elapsed:.2f}",
    }
    _append_log(row)
    print(row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("item", choices=(
        "gist_base", "gist_current", "texture", "regularization",
        "augmentation", "ensemble",
    ))
    args = parser.parse_args()
    base = np.load("data/features.npz", allow_pickle=True)
    spatial = np.load("data/features_spatial.npz", allow_pickle=True)["X"]
    gist = _load_or_build_cache(
        GIST_CACHE, _gist, base["files"], base["artists"], base["y"])
    if args.item == "gist_base":
        run_svm("gist_base", np.hstack([base["X"], gist]), base["y"])
    elif args.item == "gist_current":
        run_svm("gist_current", np.hstack([base["X"], spatial, gist]), base["y"])
    else:
        texture = _load_or_build_cache(
            TEXTURE_CACHE, _texture, base["files"], base["artists"], base["y"])
        features = np.hstack([base["X"], spatial, texture])
        if args.item == "texture":
            run_svm("texture_multiscale_lbp_gabor", features, base["y"])
        elif args.item == "regularization":
            run_svm("regularization_C_gamma", features, base["y"], pca=300, grid={
                "model__C": [1, 3, 10, 30, 100],
                "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03],
            })
            for components in (200, 400, 500):
                run_svm(f"regularization_pca_{components}", features, base["y"],
                        pca=components)
        elif args.item == "augmentation":
            run_svm("train_only_augmentation", features, base["y"], augment=True)
        else:
            run_ensemble("soft_voting_svm_logreg_rf", features, base["y"])


if __name__ == "__main__":
    main()
