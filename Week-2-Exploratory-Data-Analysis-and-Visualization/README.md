# Week 2 — Exploratory Data Analysis and Visualization

This folder contains the Week 2 EDA project using the Titanic passenger training dataset. The analysis uses Pandas for data inspection and aggregation and Matplotlib/Seaborn for visualization.

## Analysis covered
- Dataset structure and descriptive statistics
- Missing-value analysis
- Overall survival distribution
- Survival rate by sex
- Survival rate by passenger class
- Survival rate by embarkation port
- Fare distribution and skewness
- GroupBy-based transformations and aggregations
- Critical interpretation of associations versus causation

## Key findings
- 342 of 891 passengers survived (38.38%).
- Female passengers had a much higher observed survival rate than male passengers.
- First-class passengers had the highest observed survival rate, while third-class passengers had the lowest.
- Survival rates differed across embarkation ports, but port should not be treated as a causal explanation.
- Cabin has substantial missingness and Fare is strongly right-skewed.

## Dataset
Use the Titanic `train.csv` dataset from Kaggle. Keep the raw dataset unchanged and do not commit credentials or API keys.

## Suggested structure
- `README.md` — project documentation
- `eda_titanic.py` — reproducible EDA script
- `report/` — place the completed DOCX report and final screenshots here
- `data/` — local dataset location; raw data is not required to be committed
