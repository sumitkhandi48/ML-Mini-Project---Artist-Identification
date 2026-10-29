import time, os, joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from common import load_data, split_data, preprocess, report

X, y, artists = load_data()
X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
print("Train/Val/Test sizes:", len(y_tr), len(y_va), len(y_te))

X_tr, X_va, X_te, scaler, pca = preprocess(X_tr, X_va, X_te, n_components=100)
print("After PCA:", X_tr.shape)

# Small grid search with 3-fold cross-validation on the training set only
grid = {"C": [1, 10, 100], "gamma": ["scale", 0.1, 0.01]}
search = GridSearchCV(SVC(kernel="rbf"), grid, cv=3, n_jobs=-1)

start = time.time()
search.fit(X_tr, y_tr)
train_time = time.time() - start
print("Best params:", search.best_params_)
print("Best 3-fold CV accuracy:", round(search.best_score_, 3))

extra = f"Best params: {search.best_params_}\n3-fold CV accuracy: {search.best_score_:.3f}"
report("svm", search.best_estimator_, (X_tr, X_va, X_te, y_tr, y_va, y_te),
       artists, train_time, extra)

os.makedirs("models", exist_ok=True)
joblib.dump({"scaler": scaler, "pca": pca, "model": search.best_estimator_,
             "artists": artists}, "models/svm.joblib")
print("Saved models/svm.joblib")