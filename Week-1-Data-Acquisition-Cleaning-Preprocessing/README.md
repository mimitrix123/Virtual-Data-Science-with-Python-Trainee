# Week 1 — Data Acquisition, Cleaning, and Preprocessing

## Virtual Data Science with Python Trainee

This folder contains the Week 1 data science assignment focused on acquiring a public dataset, assessing data quality, cleaning missing and inconsistent values, detecting outliers, and preparing an analysis-ready dataset.

### Dataset
- **Dataset:** Titanic Passenger Dataset
- **Source:** Kaggle — Titanic: Machine Learning from Disaster
- **Training records:** 891
- **Original columns:** 12

### Objectives
1. Acquire a reliable public dataset.
2. Explore structure, data types, and quality.
3. Identify missing values and duplicates.
4. Validate numeric ranges and categorical values.
5. Detect potential outliers using the IQR method.
6. Apply justified preprocessing techniques.
7. Document the effect of preprocessing on downstream analysis.

### Main Cleaning Decisions
- **Age:** Missing values are imputed using the median within `Pclass` + `Sex` groups.
- **Embarked:** The two missing values are filled with the mode.
- **Cabin:** The original field is excluded from the baseline workflow because approximately 77% of values are missing. Deck extraction is proposed as an optional feature-engineering alternative.
- **Fare:** Extreme values are flagged using IQR rather than automatically removed because high fares may be legitimate observations.
- **Categorical values:** `Sex` and `Embarked` are standardized and validated.
- **Identifiers/text:** `PassengerId`, `Name`, and `Ticket` are removed from the baseline modeling matrix.
- **Encoding:** Categorical features are one-hot encoded.
- **Scaling:** Standardization is included as an optional step for algorithms that require comparable feature scales.

### Files
- `week1_data_cleaning.py` — reproducible end-to-end preprocessing script.
- `data/README.md` — dataset placement and provenance instructions.
- `report/` — location for the completed DOCX report and screenshots.

> The raw dataset should be kept unchanged. Do not commit private credentials or API keys.