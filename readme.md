# 🍷 Wine Quality Predictor

## 📜 Project Overview
This project predicts wine quality using **Machine Learning** techniques based on chemical properties. It explores two primary approaches:
- **Regression**: Predicts wine quality score (0-10).
- **Classification**: Categorizes wine quality as **low**, **medium**, or **high**.

### Dataset
The **Wine Quality Dataset** from Kaggle was used. [Access it here](https://www.kaggle.com/datasets/joebeachcapital/wine-quality?select=winequality-white.csv).

---

## 🛠️ Features
The dataset includes chemical properties like:
- **Alcohol percentage**
- **pH**
- **Acidity levels**
- **Sulfur dioxide content**

#### Data Preparation
- No missing values.
- Balanced dataset by adding samples for underrepresented classes.
- Normalized skewed distributions with **log transformations**.

#### Feature Engineering
- **Derived Metrics**:
  - `Alcohol-Sugar Interaction`: `alcohol × residual sugar`
  - `Total Acidity`: `fixed acidity + volatile acidity + citric acid`
  - `Total Sulfur`: `free sulfur dioxide + total sulfur dioxide`

---

## 📊 Methodology
### Model Building
#### **Random Forest (Classification)**
- Handles complex relationships and reduces overfitting.
- **Hyperparameters**: Tuned using `GridSearchCV`.
  - `max_depth=10`
  - `min_samples_split=2`
  - `min_samples_leaf=1`

#### **XGBoost (Regression)**
- Optimized for large datasets with gradient-boosted trees.
- **Hyperparameters**: Tuned using `Optuna`.
  - `subsample=0.7449`
  - `learning_rate=0.0309`
  - `max_depth=11`

---

## 🚀 Results
- **Classification (Random Forest)**:
  - **Accuracy**: 85.62%
  - **F1 Score**: 0.70807
- **Regression (XGBoost)**:
  - **R² Score**: 0.6028
  - **Mean Squared Error (MSE)**: 0.3350

Key predictors included **Alcohol** and **Total Acidity**, as identified through feature importance analysis.

---

## Installation
```plaintext
git clone https://github.com/ahmetbekir22/wine-quality-prediction
cd wine-quality-predictor
