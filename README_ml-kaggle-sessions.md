# ML Kaggle Sessions

> Machine Learning coursework — three end-to-end Kaggle competition notebooks

This repository contains the three session notebooks developed as part of a university ML course, each tackling a stage of increasing complexity on the same real estate dataset (Melbourne housing prices). All sessions were submitted to a course-internal Kaggle competition.

**Team:** Pablo Prol Prieto & Pablo Sala Fernández *(Team Pablo(s))*

---

## Sessions

### Session 1 — Feature Engineering & Regression Baseline
- Exploratory data analysis on Melbourne housing data (price, suburb, property type, building area, etc.)
- Handling of missing values with suburb-grouped imputation strategies
- Encoding of high-cardinality categorical features (hash encoding for `Suburb` and `SellerG`, one-hot for `Type`, `Regionname`, `Method`)
- Date decomposition into `Year` and `Month`
- Models evaluated: **KNN Regressor** and **Linear Regression**

### Session 2 — Preprocessing Pipelines & Stratified Splitting
- Custom sklearn `Pipeline` with `FunctionTransformer`, `ColumnTransformer`, `OneHotEncoder`
- Stratified train/validation split on price quantiles to preserve distribution
- Log transformation of skewed features
- Feature engineering: combined attributes (e.g. rooms-per-area ratios)
- Outlier detection via IQR and log-scale visualisation

### Session 3 — Imbalanced Classification & Custom Boosting
- Dataset: cross-sell response prediction (vehicle insurance)
- Handling class imbalance (~12% positive rate) with weighted loss functions
- Custom **AdaBoost-style boosting classifier** implemented from scratch with `DecisionTreeClassifier` weak learners and adaptive sample weighting
- sklearn preprocessing pipeline with `OneHotEncoder` and `ColumnTransformer`
- Submission to Kaggle with XGBoost as comparison baseline

---

## Repository Structure

```
ml-kaggle-sessions/
├── session1/
│   └── EntregaSession1.ipynb
├── session2/
│   └── EntregaSession2.ipynb
├── session3/
│   └── EntregaSession3.ipynb
└── README.md
```

---

## Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
