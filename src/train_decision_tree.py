import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from common import load_data, split_data, preprocess, report

X, y, artists = load_data()
X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
X_tr, X_va, X_te, scaler, pca = preprocess(X_tr, X_va, X_te, n_components=100)

grid = {"max_depth": [3, 5, 8, 12, 20, None], "min_samples_leaf": [1, 3, 5]}
search = GridSearchCV(DecisionTreeClassifier(criterion="entropy", random_state=42),
                      grid, cv=3, n_jobs=-1)

start = time.time()
search.fit(X_tr, y_tr)
train_time = time.time() - start
print("Best params:", search.best_params_)

extra = f"Best params: {search.best_params_}\n3-fold CV accuracy: {search.best_score_:.3f}"
report("decision_tree", search.best_estimator_, (X_tr, X_va, X_te, y_tr, y_va, y_te),
       artists, train_time, extra)