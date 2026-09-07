# Week 3 — Unsupervised Learning and Clustering

This folder contains the Week 3 unsupervised learning project using the public Iris flower dataset and Scikit-learn.

## Methodology
- Load the Iris dataset with four numerical measurements.
- Standardize features with `StandardScaler` because K-Means is distance-based.
- Compare candidate values of K using the Elbow Method and Silhouette Score.
- Select K=3 and fit reproducible K-Means (`random_state=42`, `n_init=10`).
- Visualize clusters using PCA.
- Profile clusters using average measurements and cluster sizes.
- Compare clusters with known species only after training for post-hoc validation.

## Key result
The three-cluster solution provides a useful segmentation of the Iris observations. Petal length and petal width are especially effective at separating groups, while two clusters show more overlap because their measurements are similar.

## Applications
The workflow can be adapted to customer segmentation, product analytics, biological research, quality control, and marketing analysis.

## Files
- `README.md` — project documentation
- `clustering_iris.py` — reproducible K-Means workflow
- `report/` — place the completed DOCX report and final screenshots here
- `figures/` — location for generated plots
