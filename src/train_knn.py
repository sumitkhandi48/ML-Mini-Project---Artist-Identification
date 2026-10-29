import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from common import load_data, split_data, preprocess, report

X, y, artists = load_data()
X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
X_tr, X_va, X_te, scaler, pca = preprocess(X_tr, X_va, X_te, n_components=100)

grid = {"n_neighbors": [1, 3, 5, 7, 11, 15, 21], "weights": ["uniform", "distance"]}
search = GridSearchCV(KNeighborsClassifier(), grid, cv=3, n_jobs=-1)

start = time.time()
search.fit(X_tr, y_tr)
train_time = time.time() - start
print("Best params:", search.best_params_)

extra = f"Best params: {search.best_params_}\n3-fold CV accuracy: {search.best_score_:.3f}"
report("knn", search.best_estimator_, (X_tr, X_va, X_te, y_tr, y_va, y_te),
       artists, train_time, extra)