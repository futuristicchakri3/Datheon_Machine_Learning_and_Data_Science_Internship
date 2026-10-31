"""
Task 3: Predictive Customer Churn Classification
Module: Data Preprocessing and Feature Pipeline Utilities
Internship: Datheon Machine Learning & Data Science Internship
Author: G. Chakravarthi Shiva Sai (Intern ID: DT-2026-UKTYN)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_clean_data(filepath: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Load the Telco Customer Churn dataset and perform initial data cleaning:
    - Convert TotalCharges to numeric, replacing whitespace with NaN and imputing with median/zero.
    - Remove customerID identifier column.
    - Map target variable Churn: 'No' -> 0, 'Yes' -> 1.
    - Identify numerical and categorical predictor columns.
    """
    df = pd.read_csv(filepath)

    # 1. Handle TotalCharges blank strings and convert to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip() if df["TotalCharges"].dtype == object else df["TotalCharges"], errors="coerce")
    # For tenure == 0 (new customers), TotalCharges is appropriately 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 2. Extract and encode target
    y = (df["Churn"] == "Yes").astype(int)

    # 3. Drop uninformative identifier and target
    X = df.drop(columns=["customerID", "Churn"])

    # 4. Feature typing
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """
    Construct an atomic, leak-free Scikit-Learn ColumnTransformer:
    - Numerical: Median imputation followed by StandardScaler
    - Categorical: Most frequent imputation followed by OneHotEncoder(handle_unknown='ignore')
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer, num_cols: list[str], cat_cols: list[str]) -> list[str]:
    """
    Extract transformed feature names after ColumnTransformer preprocessing
    for exact mapping in feature importance plots.
    """
    cat_transformer = fitted_preprocessor.named_transformers_["cat"]
    onehot_step = cat_transformer.named_steps["onehot"]
    cat_feature_names = list(onehot_step.get_feature_names_out(cat_cols))
    return num_cols + cat_feature_names


def evaluate_classifier(y_true, y_pred, y_prob, model_name: str = "Classifier") -> dict:
    """
    Compute comprehensive classification metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC (using predicted probabilities)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "Model": model_name,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(auc)
    }
