"""Week 3 - Unsupervised Learning and Clustering Analysis
Virtual Data Science with Python Trainee
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load public Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame.copy()
X = df.drop(columns=["target"])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare candidate cluster counts
results = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X_scaled, labels),
    })

results_df = pd.DataFrame(results)
print(results_df)

# Select three clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
print("Cluster sizes:\n", df["cluster"].value_counts().sort_index())
print("Silhouette score:", silhouette_score(X_scaled, df["cluster"]))

# Cluster profiles in original feature units
print("\nCluster profiles:\n", df.groupby("cluster").mean(numeric_only=True))

# PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for cluster in sorted(df["cluster"].unique()):
    mask = df["cluster"] == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster}")

plt.title("K-Means Clusters in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.show()
