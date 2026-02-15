# 🔥 WiDS 2026: The Comprehensive EDA & Wildfire Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=for-the-badge)
![Technique](https://img.shields.io/badge/Technique-Ensemble_Modeling-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📌 Project Overview
**"Understanding the Physics of Fire through Data."**

This repository contains a comprehensive Exploratory Data Analysis (EDA) and Machine Learning pipeline developed for the **Women in Data Science (WiDS) 2026 Datathon**. 

The mission objective is to predict **Wildfire Danger** across different regions. Before blindly applying algorithms, this notebook dives deep into the underlying environmental patterns, using Pandas core functions and advanced visualizations to understand the drivers of extreme fire events.

**Author:** Muhammad Atif

## 📂 Dataset
The dataset provides meteorological and environmental metrics crucial for forecasting fire risks.
- **Target Variable:** Wildfire Danger / Probability of Fire Occurrence (Multi-class or Binary classification based on time horizons).
- **Features Include:** Temperature, Humidity, Wind Speed, Vegetation Index, and historical fire occurrences.

## 🛠 Tech Stack
- **Language:** Python
- **Data Engineering:** `pandas`, `numpy`, `imblearn` (SMOTE for dealing with imbalanced classes)
- **Visualization:** `matplotlib`, `seaborn`, `shap` (for model interpretability)
- **Machine Learning:** - **Ensemble Models:** `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, `GradientBoostingClassifier`
  - **Meta-Estimators:** `StackingClassifier`, `VotingClassifier`, `MultiOutputClassifier`
  - **Preprocessing:** `StandardScaler`, `MinMaxScaler`, `RobustScaler`

## 📊 Project Workflow: The "Physics of Fire"

### 1. Advanced Data Intelligence 🌲
- **Extensive EDA:** Checked distributions, shapes, and statistical summaries of all weather variables.
- **Correlation Mapping:** Identified collinearity between metrics like "High Temps + Low Humidity = Maximum Danger."
- **Data Aggregation:** Renamed columns and aggregated temporal features to build stronger predictive signals.

### 2. Handling Imbalanced Nature ⚖️
Wildfires are severe but relatively rare events in the dataset.
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the models don't just default to predicting "No Fire."

### 3. Model Building & Ensembling 🧠
Instead of relying on a single algorithm, we built a robust pipeline to prevent overfitting and capture complex non-linear relationships:
- **Base Models:** Utilized the speed of `HistGradientBoosting` and the variance-reduction of `RandomForest` & `ExtraTrees`.
- **Ensemble Strategy:** Combined these base models using `StackingClassifier` and `VotingClassifier` (with `LogisticRegression` as a potential meta-learner) to maximize accuracy and F1-score.
- **Validation:** Implemented rigorous `StratifiedKFold` cross-validation to simulate real-world unseen data scenarios.

### 4. Interpretability (SHAP) 🔍
Integrated SHAP (SHapley Additive exPlanations) to explain *why* the model predicted a fire, allowing domain experts (like meteorologists) to verify the logic.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/wids-2026-wildfire.git](https://github.com/your-username/wids-2026-wildfire.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn shap
    ```
3.  **Run the Notebook:**
    Open `WIDS_WORLD.ipynb` to explore the visualizations and the stacked ensemble pipeline.

## 🤝 Contributing
If you have ideas for bringing in external satellite data (like MODIS) or implementing spatial-temporal cross-validation, feel free to fork and open a PR!

---
_Created by Muhammad Atif | WiDS 2026 Initiative_
