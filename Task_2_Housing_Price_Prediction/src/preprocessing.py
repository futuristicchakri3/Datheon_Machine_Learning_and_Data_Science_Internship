"""
Task 2: Real Estate Housing Price Prediction
Module: Data Preprocessing and Feature Engineering Utilities
Internship: Datheon Machine Learning & Data Science Internship
Author: G. Chakravarthi Shiva Sai (Intern ID: DT-2026-UKTYN)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to engineer domain-specific housing features:
    - rooms_per_household = total_rooms / households
    - bedrooms_per_room = total_bedrooms / total_rooms
    - population_per_household = population / households
    - income_per_household = median_income / rooms_per_household (economic purchasing power proxy)

    Applies numerical stability using safe division to avoid division-by-zero errors.
    """

    def __init__(self, add_interaction_features: bool = True, eps: float = 1e-6):
        self.add_interaction_features = add_interaction_features
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Support both pandas DataFrame and numpy ndarray
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            df["rooms_per_household"] = df["total_rooms"] / (df["households"] + self.eps)
            df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + self.eps)
            df["population_per_household"] = df["population"] / (df["households"] + self.eps)

            if self.add_interaction_features:
                # Interaction: income intensity per room
                df["income_per_room"] = df["median_income"] / (df["rooms_per_household"] + self.eps)

            return df
        else:
            # If numpy array, assume original column ordering:
            # 0: longitude, 1: latitude, 2: housing_median_age, 3: total_rooms,
            # 4: total_bedrooms, 5: population, 6: households, 7: median_income
            rooms_per_household = X[:, 3] / (X[:, 6] + self.eps)
            bedrooms_per_room = X[:, 4] / (X[:, 3] + self.eps)
            population_per_household = X[:, 5] / (X[:, 6] + self.eps)

            extra_features = [rooms_per_household, bedrooms_per_room, population_per_household]

            if self.add_interaction_features:
                income_per_room = X[:, 7] / (rooms_per_household + self.eps)
                extra_features.append(income_per_room)

            return np.c_[X, *extra_features]


def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for specified numerical predictors
    using iterative Ordinary Least Squares regression: VIF = 1 / (1 - R^2).
    A VIF > 5 indicates moderate multicollinearity; VIF > 10 indicates severe multicollinearity.
    """
    clean_df = df[feature_cols].dropna()
    vif_records = []

    for i, col in enumerate(feature_cols):
        y = clean_df[col]
        X = clean_df.drop(columns=[col])

        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)

        if r_squared >= 0.999999:
            vif = float("inf")
        else:
            vif = 1.0 / (1.0 - r_squared)

        vif_records.append({
            "Feature": col,
            "R_Squared": round(r_squared, 4),
            "VIF": round(vif, 2)
        })

    vif_df = pd.DataFrame(vif_records).sort_values(by="VIF", ascending=False).reset_index(drop=True)
    return vif_df


def create_preprocessor(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Construct an atomic, leak-free Scikit-Learn ColumnTransformer:
    - Numerical: Median imputation followed by StandardScaler
    - Categorical: Most frequent imputation followed by OneHotEncoder(drop='first')
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def evaluate_regression_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Compute standard regression evaluation metrics:
    - RMSE: Root Mean Squared Error (interpretable in original currency scale)
    - MAE: Mean Absolute Error (robust to extreme outliers)
    - R² Score: Coefficient of Determination (explained variance ratio)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2 Score": float(r2)
    }
