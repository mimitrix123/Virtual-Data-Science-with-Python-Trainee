import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

SEED = 42

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# Supervised learning: leakage-safe preprocessing + Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=SEED)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = cross_validate(
    model, X_train, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Test Accuracy:", accuracy_score(y_test, pred))
print("Test Precision:", precision_score(y_test, pred))
print("Test Recall:", recall_score(y_test, pred))
print("Test F1:", f1_score(y_test, pred))
print("Test ROC-AUC:", roc_auc_score(y_test, prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# Unsupervised learning: evaluate K=2..6
X_scaled = StandardScaler().fit_transform(X)
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = km.fit_predict(X_scaled)
    print(f"K={k}: inertia={km.inertia_:.2f}, silhouette={silhouette_score(X_scaled, labels):.4f}")

# Documented selection: strongest tested silhouette score
best_k = 2
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=SEED)
clusters = kmeans.fit_predict(X_scaled)

# PCA for visualization only
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
print("PCA explained variance:", pca.explained_variance_ratio_)
