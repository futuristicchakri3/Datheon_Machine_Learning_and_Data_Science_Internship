# Datheon Machine Learning & Data Science Internship

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4%2B-red.svg)
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
├── Task_3_Customer_Churn_Classification/       # [COMPLETED]
│   ├── dataset/
│   │   └── Telco-Customer-Churn.csv           # Official IBM Telco Customer Churn dataset
│   ├── notebooks/
│   │   └── customer_churn_prediction.ipynb    # Fully executed 26-section classification notebook
│   ├── images/                                # 8 publication-grade visualizations
│   │   ├── churn_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── roc_curve_comparison.png
│   │   ├── random_forest_confusion_matrix.png
│   │   ├── xgboost_confusion_matrix.png
│   │   ├── random_forest_feature_importance.png
│   │   ├── xgboost_feature_importance.png
│   │   └── model_comparison.png
│   ├── src/
│   │   └── preprocessing.py                   # Reusable preprocessing & evaluation module
│   ├── generate_visualizations.py             # Benchmark generation script
│   ├── build_notebook.py                      # Automated notebook generator script
│   ├── requirements.txt                       # Task-specific dependencies
│   └── README.md                              # Dedicated Task 3 documentation
│
├── Task_4_Sentiment_Analysis/                 # [COMPLETED]
│   ├── dataset/
│   │   └── sentiment_reviews.csv              # Official 10-record supplied review dataset
│   ├── notebooks/
│   │   └── sentiment_classification.ipynb     # Fully executed 26-section NLP classification notebook
│   ├── images/                                # 6 publication-grade NLP visualizations
│   │   ├── sentiment_distribution.png
│   │   ├── review_length_distribution.png
│   │   ├── word_frequency.png
│   │   ├── confusion_matrix_naive_bayes.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   └── model_comparison.png
│   ├── src/
│   │   └── text_preprocessing.py              # Modular NLP pipeline, NLTK & scikit-learn transformer
│   ├── generate_visualizations.py             # Benchmark & visualization generator script
│   ├── build_notebook.py                      # Automated notebook builder & execution runner
│   ├── requirements.txt                       # Task-specific dependencies
│   └── README.md                              # Dedicated Task 4 documentation
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
| **Task 3** | **Predictive Customer Churn Classification** | Random Forest, XGBoost, Class Imbalance Handling (SMOTE & Class Weights), ROC-AUC Optimization, Feature Importance | **✅ Completed** | [Task_3_Customer_Churn_Classification](./Task_3_Customer_Churn_Classification/) |
| **Task 4** | **NLP Sentiment Classification on Customer Reviews** | TF-IDF Vectorization, Negation-Preserving NLP Pipeline, Multinomial Naive Bayes, Logistic Regression, Custom Predictor | **✅ Completed** | [Task_4_Sentiment_Analysis](./Task_4_Sentiment_Analysis/) |

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

## 🌟 Task 3 Highlights: Predictive Customer Churn Classification

- **Dataset:** Official IBM Telco Customer Churn Dataset (7,043 subscriber records, 21 attributes, 26.54% empirical churn rate).
- **Class Imbalance Resolution:** Solved 2.77:1 majority-to-minority imbalance using cost-sensitive learning (`class_weight='balanced'` in Random Forest and `scale_pos_weight` in XGBoost) within atomic `ColumnTransformer` pipelines.
- **Data Cleaning & Pipeline:** Imputed 11 new-account whitespace strings in `TotalCharges` (`tenure = 0` to 0.0), excluded `customerID`, standard-scaled numerical attributes, and one-hot encoded 16 categorical predictors without test leakage.
- **Benchmark Results (20% Holdout Test - 1,409 Subscribers):**
  - **XGBoost Classifier (Tuned):** **Champion ROC-AUC = 0.8460**, **Accuracy = 80.55%**, **Precision = 67.48%**, F1-Score = **0.5848** (Max Discrimination & Precision).
  - **Random Forest Classifier (Tuned):** **Test ROC-AUC = 0.8415**, **Recall = 79.14%**, Accuracy = **73.95%**, Precision = **50.60%**, F1-Score = **0.6173** (Champion Retention: captures ~80% of churning subscribers).
- **Core Churn Drivers Identified:** `Contract_Month-to-month` (42.7% segment churn rate), low `tenure` (first 12 months critical), `InternetService_Fiber optic` (41.9% churn rate), `PaymentMethod_Electronic check` (45.3% churn rate), and absence of `OnlineSecurity` / `TechSupport`.

👉 **For complete methodology, figures, and classification reports, see [Task 3 README](./Task_3_Customer_Churn_Classification/README.md).**

---

## 🌟 Task 4 Highlights: NLP Sentiment Classification on Customer Reviews (TF-IDF)

- **Dataset:** Authentic supplied customer review dataset (`sentiment_reviews.csv` - 10 records: 5 Positive, 4 Negative, 1 Neutral).
- **Neutral Record Isolation:** Isolated `Review ID 3` (*"Average performance..."*) into a documented auditing subset to adhere strictly to binary positive/negative classification.
- **Negation-Preserving NLP Pipeline:** Integrated BeautifulSoup HTML stripping, case normalization, non-alphanumeric punctuation removal, NLTK tokenization, context-aware Penn Treebank POS-tagged WordNet lemmatization, and stopword filtering while strictly safeguarding polarity-altering negation tokens (*not*, *no*, *never*, *cannot*).
- **TF-IDF Feature Representation:** Extracted 113 unique n-gram features using `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)` with 89.28% sparsity.
- **Model Development & Benchmark:**
  - **Multinomial Naive Bayes ($\alpha=1.0$):** **1.000 In-Sample Accuracy**, **1.000 F1-Score**; demonstrated out-of-vocabulary handling in LOOCV.
  - **Logistic Regression ($L_2$ Regularization):** **1.000 In-Sample Accuracy**, **1.000 F1-Score**.
- **Interactive Sentiment Function:** Deployed production-grade `predict_sentiment(text)` function returning predicted polarity, class confidence scores, and input validation handling.
- **Critical Dataset Size Notice:** Explicitly documented that the 10-record corpus is an instructional NLP demonstration rather than a real-world generalization benchmark.

👉 **For complete methodology, figures, and NLP diagnostic reports, see [Task 4 README](./Task_4_Sentiment_Analysis/README.md).**

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

# Execute Task 3 visualization and benchmark script
python Task_3_Customer_Churn_Classification/generate_visualizations.py

# Execute Task 4 visualization and benchmark script
python Task_4_Sentiment_Analysis/generate_visualizations.py

# Launch Jupyter Notebooks
jupyter notebook Task_4_Sentiment_Analysis/notebooks/sentiment_classification.ipynb
```

---

## 📜 Academic & Professional Integrity

All code, analysis, and documentation in this repository have been developed independently for the Datheon Machine Learning & Data Science Internship adhering to professional software engineering standards.
