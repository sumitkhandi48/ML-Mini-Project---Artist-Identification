import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    d = np.load("data/features.npz", allow_pickle=True)
    return d["X"], d["y"], d["artists"]

def split_data(X, y, seed=42):
    # 10% test, then 1/9 of the rest = 10% of total for validation -> 80/10/10
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=1/9, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

def preprocess(X_tr, X_va, X_te, n_components=100):
    scaler = MinMaxScaler().fit(X_tr)
    X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)
    pca = PCA(n_components=n_components, random_state=42).fit(X_tr)
    return pca.transform(X_tr), pca.transform(X_va), pca.transform(X_te), scaler, pca

def report(name, model, data, artists, train_time, extra=""):
    import seaborn as sns
    X_tr, X_va, X_te, y_tr, y_va, y_te = data
    os.makedirs("results", exist_ok=True)
    acc_tr = accuracy_score(y_tr, model.predict(X_tr))
    acc_va = accuracy_score(y_va, model.predict(X_va))
    pred = model.predict(X_te)
    acc_te = accuracy_score(y_te, pred)
    text = (f"Model: {name}\n{extra}\n"
            f"Train accuracy: {acc_tr:.3f}\nValidation accuracy: {acc_va:.3f}\n"
            f"Test accuracy: {acc_te:.3f}\nTraining time (s): {train_time:.2f}\n\n"
            + classification_report(y_te, pred, target_names=artists, digits=3))
    print(text)
    with open(f"results/{name}_results.txt", "w", encoding="utf-8") as f:
        f.write(text)
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=artists, yticklabels=artists)
    plt.xlabel("Predicted artist"); plt.ylabel("True artist")
    plt.title(f"{name} - confusion matrix (test set)")
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig(f"results/{name}_confusion_matrix.png", dpi=150)
    plt.close()
    return acc_tr, acc_va, acc_te