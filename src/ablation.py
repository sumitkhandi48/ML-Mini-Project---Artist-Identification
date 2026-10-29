import numpy as np, pandas as pd, time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from common import load_data, split_data

X, y, artists = load_data()
X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)

# column ranges in our feature vector
blocks = {"HOG": slice(0, 1764), "Color": slice(1764, 1788), "Hu": slice(1788, 1795)}
combos = {
    "HOG": ["HOG"], "Color": ["Color"], "Hu": ["Hu"],
    "Color+Hu": ["Color", "Hu"], "HOG+Color": ["HOG", "Color"],
    "HOG+Color+Hu": ["HOG", "Color", "Hu"],
}

def pick(M, names):
    return np.hstack([M[:, blocks[n]] for n in names])

rows = []
for cname, names in combos.items():
    A_tr, A_va = pick(X_tr, names), pick(X_va, names)
    sc = MinMaxScaler().fit(A_tr)
    A_tr, A_va = sc.transform(A_tr), sc.transform(A_va)
    for nc in [20, 50, 100]:
        nc_eff = min(nc, A_tr.shape[1])
        pca = PCA(n_components=nc_eff, random_state=42).fit(A_tr)
        P_tr, P_va = pca.transform(A_tr), pca.transform(A_va)
        s = GridSearchCV(SVC(kernel="rbf"), {"C": [1, 10, 100], "gamma": ["scale", 0.1, 0.01]},
                         cv=3, n_jobs=-1).fit(P_tr, y_tr)
        rows.append({"Features": cname, "PCA comps": nc_eff,
                     "CV acc": round(s.best_score_, 3),
                     "Val acc": round(s.score(P_va, y_va), 3),
                     "Train acc": round(s.score(P_tr, y_tr), 3)})
        print(rows[-1])

df = pd.DataFrame(rows).drop_duplicates(["Features", "PCA comps"])
df = df.sort_values("CV acc", ascending=False)
df.to_csv("results/ablation.csv", index=False)
print("\nTop 8 by CV accuracy:")
print(df.head(8).to_string(index=False))