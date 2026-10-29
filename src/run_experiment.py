"""Run validation-only feature/model experiments and append experiments.csv."""

import argparse
import os
import time

import cv2
import mahotas
import numpy as np
from skimage.filters import gabor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from common import load_data, split_data

RESULTS = "results/experiments.csv"


def load_features(kind):
    d = np.load("data/features.npz", allow_pickle=True)
    X, y, artists, files = d["X"], d["y"], d["artists"], d["files"]
    if kind == "base":
        return X, y, artists, files
    spatial = np.load("data/features_spatial.npz", allow_pickle=True)
    if not np.array_equal(files, spatial["files"]):
        raise ValueError("Base and spatial caches are not row-aligned")
    if kind == "spatial":
        return spatial["X"], y, artists, files
    return np.hstack([X, spatial["X"]]), y, artists, files


def fit_validate(X, y, components, scaler, grid):
    X_tr, X_va, _, y_tr, y_va, _ = split_data(X, y)
    scaler = scaler.fit(X_tr)
    A_tr, A_va = scaler.transform(X_tr), scaler.transform(X_va)
    pca = PCA(n_components=min(components, A_tr.shape[1]), whiten=False,
              random_state=42).fit(A_tr)
    P_tr, P_va = pca.transform(A_tr), pca.transform(A_va)
    search = GridSearchCV(SVC(kernel="rbf"), grid, cv=3, n_jobs=-1)
    start = time.time()
    search.fit(P_tr, y_tr)
    return {
        "cv": search.best_score_,
        "val": search.score(P_va, y_va),
        "train": search.score(P_tr, y_tr),
        "params": search.best_params_,
        "seconds": time.time() - start,
    }


def extract_texture_and_gist(files, artists, y, mode):
    rows = []
    for index, filename in enumerate(files):
        path = os.path.join("data/images", artists[y[index]], filename)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
        if mode == "haralick":
            rows.append(mahotas.features.haralick((image * 255).astype(np.uint8)).mean(0))
        else:
            values = []
            for frequency in (0.1, 0.2, 0.3, 0.4):
                for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
                    real, imag = gabor(image, frequency=frequency, theta=theta)
                    response = np.hypot(real, imag)
                    for ys in range(0, 256, 64):
                        for xs in range(0, 256, 64):
                            values.append(response[ys:ys + 64, xs:xs + 64].mean())
            rows.append(values)
        if (index + 1) % 500 == 0:
            print(f"{mode}: {index + 1}/{len(files)}")
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("item", choices=("baseline", "4", "2", "3", "5", "1"))
    args = parser.parse_args()
    d = np.load("data/features.npz", allow_pickle=True)
    X, y, artists, files = d["X"], d["y"], d["artists"], d["files"]
    if args.item == "baseline":
        features, components, scaler, grid, name = X, 100, MinMaxScaler(), {
            "C": [1, 10, 100], "gamma": ["scale", 0.1, 0.01]
        }, "baseline_recheck"
    elif args.item == "4":
        features, _, _, _ = load_features("combined")
        components, scaler, grid, name = 100, MinMaxScaler(), {
            "C": [1, 10, 100], "gamma": ["scale", 0.1, 0.01]
        }, "spatial_hsv_lbp"
    elif args.item == "2":
        features, _, _, _ = load_features("combined")
        components, scaler, grid, name = 300, StandardScaler(), {
            "C": [1, 10, 100], "gamma": ["scale", 0.01, 0.03, 0.1]
        }, "scaling_pca"
    elif args.item == "3":
        features, _, _, _ = load_features("combined")
        components, scaler, grid, name = 300, StandardScaler(), {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
            "class_weight": [None, "balanced"],
        }, "wide_svm_grid"
    elif args.item == "5":
        texture = extract_texture_and_gist(files, artists, y, "haralick")
        spatial = np.load("data/features_spatial.npz", allow_pickle=True)["X"]
        features, components, scaler, grid, name = np.hstack([X, spatial, texture]), 300, StandardScaler(), {
            "C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]
        }, "haralick"
    else:
        gist = extract_texture_and_gist(files, artists, y, "gist")
        spatial = np.load("data/features_spatial.npz", allow_pickle=True)["X"]
        features, components, scaler, grid, name = np.hstack([X, spatial, gist]), 300, StandardScaler(), {
            "C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]
        }, "gist_approximation"

    result = fit_validate(features, y, components, scaler, grid)
    print(f"{name}: CV={result['cv']:.3f} Val={result['val']:.3f} "
          f"Train={result['train']:.3f} Params={result['params']}")
    os.makedirs("results", exist_ok=True)
    header = "experiment,features,PCA components,model parameters,CV accuracy,validation accuracy,train accuracy,test accuracy\n"
    if not os.path.exists(RESULTS):
        with open(RESULTS, "w", encoding="utf-8") as output:
            output.write(header)
    with open(RESULTS, "a", encoding="utf-8") as output:
        output.write(f"{name},{features.shape[1]},{components},"
                     f"\"{result['params']}\",{result['cv']:.3f},"
                     f"{result['val']:.3f},{result['train']:.3f},\n")


if __name__ == "__main__":
    main()
