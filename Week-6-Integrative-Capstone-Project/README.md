# Week 6 — Integrative Capstone Project and Evaluation

## Project
End-to-End Breast Cancer Diagnostic Analysis with Python.

## Objective
Integrate the internship's major data-science skills into one reproducible pipeline covering data acquisition, preprocessing, exploratory data analysis, supervised learning, unsupervised learning, evaluation, interpretation, and recommendations.

## Dataset
Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn:
- 569 observations
- 30 numerical features
- Binary target: malignant / benign
- Publicly accessible through `sklearn.datasets.load_breast_cancer`

## Methodology
1. Programmatic data acquisition
2. Data-quality checks and preprocessing
3. Exploratory analysis
4. Logistic Regression with StandardScaler and median imputation
5. Five-fold stratified cross-validation
6. Held-out test evaluation
7. K-Means clustering with K evaluated from 2 to 6
8. Silhouette analysis and PCA visualization
9. Critical analysis and recommendations

## Supervised Results
The documented Logistic Regression run achieved:
- Accuracy: **98.25%**
- Precision: **98.61%**
- Recall: **98.61%**
- F1-score: **98.61%**
- ROC-AUC: **0.995**

## Unsupervised Results
K-Means was applied without using the diagnosis label. Among K=2 through K=6, **K=2** produced the strongest silhouette score in the documented analysis. PCA was used to visualize the resulting cluster structure in two dimensions.

## Key Insights
- The supervised model provides strong predictive performance on the held-out benchmark.
- Cross-validation supports the stability of the baseline model.
- Clustering provides a complementary descriptive perspective without target labels.
- Multiple evaluation metrics are more informative than accuracy alone.
- High benchmark performance does not establish clinical readiness; external validation and domain-specific assessment are required.

## Files
- `README.md` — project overview
- `capstone_breast_cancer.py` — reproducible Python implementation
- `requirements.txt` — dependencies
- `report/README.md` — report submission note

## Future Work
External validation, repeated experiments, calibrated probabilities, threshold analysis, feature selection, alternative supervised models, and additional clustering techniques are recommended before any real-world deployment.
