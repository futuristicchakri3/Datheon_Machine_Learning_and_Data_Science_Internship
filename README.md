# Datheon Machine Learning & Data Science Internship

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Internship](https://img.shields.io/badge/Datheon-Internship-blueviolet.svg)

This repository contains the complete portfolio of projects developed during the **Machine Learning & Data Science Internship** at **[Datheon — AI Systems & Automation Studio](https://www.datheon.in/)**.

---

## 👨‍💻 Intern Profile

- **Name:** G. Chakravarthi Shiva Sai
- **Internship ID:** DT-2026-UKTYN
- **Domain:** Machine Learning & Data Science
- **Organization:** Datheon AI Systems & Automation Studio

---

## 📂 Repository Architecture & Roadmap

```
Datheon_Machine_Learning_and_Data_Science_Internship/
│
├── Task_1_Classification/                     # [COMPLETED]
│   ├── dataset/
│   │   └── iris.csv                           # Official benchmark dataset
│   ├── notebooks/
│   │   └── classification.ipynb               # Fully executed 16-section notebook
│   ├── images/                                # 7 publication-grade visualizations
│   │   ├── feature_distributions.png
│   │   ├── pairplot.png
│   │   ├── correlation_heatmap.png
│   │   ├── lr_confusion_matrix.png
│   │   ├── dt_confusion_matrix.png
│   │   ├── model_comparison.png
│   │   └── decision_boundaries.png
│   ├── generate_pipeline.py                   # Automated pipeline script
│   └── README.md                              # Dedicated Task 1 documentation
│
├── Task_2_Housing_Price_Prediction/           # [COMPLETED]
│   ├── dataset/
│   │   └── california_housing.csv             # Official California Housing Census dataset
│   ├── notebooks/
│   │   └── housing_price_prediction.ipynb     # Fully executed 21-section regression notebook
│   ├── images/                                # 8 publication-grade visualizations
│   │   ├── target_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── vif_analysis.png
│   │   ├── model_comparison.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_distribution.png
│   │   ├── residuals_vs_predicted.png
│   │   └── regularization_coefficients.png
│   ├── src/
│   │   └── preprocessing.py                   # Reusable transformer & utility module
│   ├── generate_visualizations.py             # Benchmark generation script
│   └── README.md                              # Dedicated Task 2 documentation
│
├── Task_4_Sentiment_Analysis/                 # [PENDING / PLANNED]
│   └── .gitkeep                               # Reserved for Task 4 implementation
│
├── requirements.txt                           # Project-wide dependencies
├── README.md                                  # Root repository overview
└── .gitignore                                 # Git exclusion rules
```

---

## 📋 Internship Project Portfolio

| Task | Title | Key Algorithms & Methods | Status | Directory Link |
| :---: | :--- | :--- | :---: | :--- |
| **Task 1** | **Multi-Class Benchmark Classification Pipeline** | Scikit-Learn Pipelines, StandardScaler, Logistic Regression, Decision Tree Classifier, Confusion Matrices, PCA 2D Visualization | **✅ Completed** | [Task_1_Classification](./Task_1_Classification/) |
| **Task 2** | **Real Estate Housing Price Prediction** | Ridge & Lasso Regression, L1/L2 Regularization, Multicollinearity (VIF), Feature Engineering, Residual Diagnostics | **✅ Completed** | [Task_2_Housing_Price_Prediction](./Task_2_Housing_Price_Prediction/) |
| **Task 4** | **NLP Sentiment Classification on Customer Reviews** | TF-IDF Vectorization, Text Preprocessing, Sentiment Analysis Classifiers | **⏳ Pending** | [Task_4_Sentiment_Analysis](./Task_4_Sentiment_Analysis/) |

---

## 🌟 Task 1 Highlights: Classification Benchmark

- **Dataset:** Fisher's Iris Dataset (150 samples, 4 continuous features, 3 balanced classes: *setosa*, *versicolor*, *virginica*).
- **Leak-Free Preprocessing:** Atomic Scikit-Learn pipelines ensuring `StandardScaler` is fitted exclusively on the 80% training set.
- **Stratified Split:** 80% train (120 samples) / 20% test (30 samples) with fixed seed `random_state=42`.
- **Benchmark Results:**
  - **Decision Tree Classifier:** **96.67% Test Accuracy**, **0.9666 F1-Score** (Champion Model)
  - **Logistic Regression:** **93.33% Test Accuracy**, **0.9333 F1-Score**
- **Decision Boundaries:** 2D decision landscapes projected via PCA capturing **95.81%** of total dataset variance.

👉 **For complete methodology, figures, and classification reports, see [Task 1 README](./Task_1_Classification/README.md).**

---

## 🌟 Task 2 Highlights: Housing Price Prediction (Ridge & Lasso)

- **Dataset:** California Housing Dataset (1990 U.S. Census, 20,640 records, 10 original features, median house value capped at \$500,001).
- **Leak-Free Preprocessing:** Scikit-Learn `ColumnTransformer` & `Pipeline` combining median imputation, standard scaling, and one-hot encoding fitted exclusively on 80% train split.
- **Multicollinearity & VIF Analysis:** Diagnosed extreme collinearity between `total_bedrooms` (VIF 50.39) and `households` (VIF 48.38); explained mathematical superiority of Ridge ($L_2$) conditioning $(X^TX + \alpha I)^{-1}$ and Lasso ($L_1$) coordinate sparsity.
- **Feature Engineering:** Derived structural ratios (`bedrooms_per_room`, `rooms_per_household`, `population_per_household`) and interaction feature `income_per_room` using numerically safe division ($\epsilon = 10^{-6}$).
- **Benchmark Results (20% Holdout Test):**
  - **Ridge Regression ($\alpha=1.0$):** **Champion CV RMSE: \$67,840.09**, Test RMSE: **\$69,096.46**, Test MAE: **\$49,614.74**, $R^2$: **0.6357** (Shrank sparse category `ocean_proximity_ISLAND` coefficient by >20%).
  - **OLS Regression:** Test RMSE: **\$69,086.96**, Test MAE: **\$49,606.59**, $R^2$: **0.6358**.
  - **Lasso Regression ($\alpha=1.0$):** Test RMSE: **\$69,088.18**, Test MAE: **\$49,607.68**, $R^2$: **0.6357**.
- **Residual Diagnostics:** Mean residual bias of **+\$48.06** (practically zero), symmetric Gaussian error curve, homoscedastic error bounds, and documented the \$500,001 artificial ceiling cap effect.

👉 **For complete methodology, figures, and regression reports, see [Task 2 README](./Task_2_Housing_Price_Prediction/README.md).**

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher (compatible with Python 3.14)
- Git 2.x

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/futuristicchakri3/Datheon_Machine_Learning_and_Data_Science_Internship.git
cd Datheon_Machine_Learning_and_Data_Science_Internship

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Pipeline or Notebook
```bash
# Execute Task 1 automated pipeline script
python Task_1_Classification/generate_pipeline.py

# Execute Task 2 visualization and benchmark script
python Task_2_Housing_Price_Prediction/generate_visualizations.py

# Launch Jupyter Notebook
jupyter notebook Task_2_Housing_Price_Prediction/notebooks/housing_price_prediction.ipynb
```

---

## 📜 Academic & Professional Integrity

All code, analysis, and documentation in this repository have been developed independently for the Datheon Machine Learning & Data Science Internship adhering to professional software engineering standards.
