# ML Kaggle Sessions

> Machine Learning coursework — four end-to-end Kaggle competition notebooks

This repository contains the four session notebooks developed as part of a university ML course. Sessions 1–3 tackle progressive stages of a real estate regression problem (Melbourne housing prices), while Session 4 moves into computer vision with a crowd counting task. All sessions were submitted to a course-internal Kaggle competition.

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

### Session 4 — CNN Regression for Crowd Counting
- Dataset: 15,000 grayscale park images labelled with the number of people (0–29), balanced across all classes
- Task framed as **regression** (continuous count prediction) rather than classification
- Data augmentation via `tf.data`: random horizontal/vertical flips, 90° rotations, brightness and contrast jitter; augmented set concatenated with original training data
- Multiple CNN architectures explored and compared (commented out in notebook):
  - Standard deep CNN (sequential conv blocks with increasing filter depth)
  - **SACNN** (Scale-Adaptive CNN): two parallel branches with 3×3 and 5×5 kernels, merged via concatenation
  - **MCNN** (Multi-Column CNN): three parallel columns with 9×9, 7×7 and 5×5 kernels, inspired by crowd counting literature
  - **Hybrid deep/shallow CNN**: VGG-style deep branch merged with a shallow average-pooling branch
- Final model: sequential CNN (32→64→128 filters) compiled with MSE loss
- Training with `Adam(lr=0.0001)`, `ReduceLROnPlateau` scheduler and `EarlyStopping`
- Submission generated on test set after retraining on train + half of validation data

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
├── session4/
│   └── crowd-counting-aa.ipynb
└── README.md
```

---

## Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
