"""
Task 2: Real Estate Housing Price Prediction
Script: Visualizations & Regression Benchmark Generation
Internship: Datheon Machine Learning & Data Science Internship
Author: G. Chakravarthi Shiva Sai (Intern ID: DT-2026-UKTYN)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.preprocessing import (
    HousingFeatureEngineer,
    create_preprocessor,
    compute_vif,
    evaluate_regression_model,
)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

img_dir = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(img_dir, exist_ok=True)

# 1. Load dataset
csv_path = os.path.join(os.path.dirname(__file__), 'dataset', 'california_housing.csv')
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Plot 1: Target Variable Distribution ---
plt.figure(figsize=(9, 5), dpi=300)
sns.histplot(df['median_house_value'], kde=True, bins=50, color='#1f77b4', edgecolor='black', alpha=0.6)
plt.axvline(df['median_house_value'].median(), color='red', linestyle='--', linewidth=1.5, label=f"Median: ${df['median_house_value'].median():,.0f}")
plt.axvline(df['median_house_value'].mean(), color='orange', linestyle='-', linewidth=1.5, label=f"Mean: ${df['median_house_value'].mean():,.0f}")
plt.axvline(500000, color='darkred', linestyle=':', linewidth=2, label="Capped Ceiling ($500,001)")
plt.title('Distribution of Median House Value (Target Variable)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Median House Value ($)', fontsize=12)
plt.ylabel('Frequency / Count', fontsize=12)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'target_distribution.png'))
plt.close()
print("Saved target_distribution.png")

# 2. Feature Engineering
fe = HousingFeatureEngineer(add_interaction_features=True)
df_fe = fe.transform(df)

# --- Plot 2: Correlation Heatmap ---
num_cols_for_corr = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
    'income_per_room', 'median_house_value'
]
corr_matrix = df_fe[num_cols_for_corr].corr()

plt.figure(figsize=(11, 9), dpi=300)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)
plt.title('Feature Correlation Matrix (Including Engineered Features)', fontsize=14, fontweight='bold', pad=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'correlation_heatmap.png'))
plt.close()
print("Saved correlation_heatmap.png")

# --- Plot 3: VIF Analysis ---
num_features = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'income_per_room'
]
vif_df = compute_vif(df_fe, num_features)

plt.figure(figsize=(10, 6), dpi=300)
colors = ['#d9534f' if v > 10 else '#f0ad4e' if v > 5 else '#5cb85c' for v in vif_df['VIF']]
bars = plt.barh(vif_df['Feature'][::-1], vif_df['VIF'][::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
plt.axvline(5, color='#f0ad4e', linestyle='--', linewidth=1.5, label='Moderate Collinearity (VIF=5)')
plt.axvline(10, color='#d9534f', linestyle='--', linewidth=1.5, label='Severe Collinearity (VIF=10)')
plt.title('Variance Inflation Factor (VIF) Multicollinearity Analysis', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('VIF Value', fontsize=12)
plt.ylabel('Predictor Variable', fontsize=12)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{w:.1f}', va='center', fontsize=9, fontweight='bold')
plt.xlim(0, max(vif_df['VIF']) + 6)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'vif_analysis.png'))
plt.close()
print("Saved vif_analysis.png")

# 3. Model Training & Comparison
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_fe = fe.transform(X_train)
X_test_fe = fe.transform(X_test)

cat_features = ['ocean_proximity']
preprocessor = create_preprocessor(num_features, cat_features)

# OLS
ols_pipe = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
ols_pipe.fit(X_train_fe, y_train)
y_pred_ols = ols_pipe.predict(X_test_fe)

# Ridge
ridge_pipe = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge(random_state=42))])
ridge_grid = GridSearchCV(ridge_pipe, {'regressor__alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]}, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
ridge_grid.fit(X_train_fe, y_train)
best_ridge = ridge_grid.best_estimator_
y_pred_ridge = best_ridge.predict(X_test_fe)

# Lasso
lasso_pipe = Pipeline([('preprocessor', preprocessor), ('regressor', Lasso(random_state=42, max_iter=20000))])
lasso_grid = GridSearchCV(lasso_pipe, {'regressor__alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]}, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
lasso_grid.fit(X_train_fe, y_train)
best_lasso = lasso_grid.best_estimator_
y_pred_lasso = best_lasso.predict(X_test_fe)

best_ridge_alpha = ridge_grid.best_params_['regressor__alpha']
best_lasso_alpha = lasso_grid.best_params_['regressor__alpha']

metrics = [
    evaluate_regression_model(y_test, y_pred_ols, 'OLS'),
    evaluate_regression_model(y_test, y_pred_ridge, f'Ridge (alpha={best_ridge_alpha})'),
    evaluate_regression_model(y_test, y_pred_lasso, f'Lasso (alpha={best_lasso_alpha})')
]
comp_df = pd.DataFrame(metrics)
print("\nBenchmark Results:\n", comp_df)

# --- Plot 4: Model Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
models = ['OLS', f'Ridge (α={best_ridge_alpha})', f'Lasso (α={best_lasso_alpha})']
palette = ['#2b5c8f', '#e26d5c', '#38b000']

# RMSE
axes[0].bar(models, comp_df['RMSE'], color=palette, edgecolor='black', alpha=0.85)
axes[0].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE ($)', fontsize=11)
axes[0].set_ylim(65000, 71500)
for i, v in enumerate(comp_df['RMSE']):
    axes[0].text(i, v + 80, f"${v:,.2f}", ha='center', fontsize=9, fontweight='bold')

# MAE
axes[1].bar(models, comp_df['MAE'], color=palette, edgecolor='black', alpha=0.85)
axes[1].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MAE ($)', fontsize=11)
axes[1].set_ylim(47000, 51500)
for i, v in enumerate(comp_df['MAE']):
    axes[1].text(i, v + 60, f"${v:,.2f}", ha='center', fontsize=9, fontweight='bold')

# R2 Score
axes[2].bar(models, comp_df['R2 Score'], color=palette, edgecolor='black', alpha=0.85)
axes[2].set_title('R² Score (Variance Explained)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('R² Score', fontsize=11)
axes[2].set_ylim(0.60, 0.66)
for i, v in enumerate(comp_df['R2 Score']):
    axes[2].text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Performance Comparison: OLS vs Ridge vs Lasso Regression', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'model_comparison.png'))
plt.close()
print("Saved model_comparison.png")

# Champion model: Ridge
champ_name = "Ridge"
champ_pred = y_pred_ridge
residuals = y_test - champ_pred

# --- Plot 5: Actual vs Predicted ---
plt.figure(figsize=(8, 7), dpi=300)
plt.scatter(y_test, champ_pred, alpha=0.25, color='#1f77b4', edgecolors='none', s=20, label='Test Observations')
lims = [0, 550000]
plt.plot(lims, lims, color='red', linestyle='--', linewidth=2, label='Identity Line (y = x)')
plt.title(f'Actual vs Predicted House Values ({champ_name} Regression)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Actual Median House Value ($)', fontsize=11)
plt.ylabel('Predicted Median House Value ($)', fontsize=11)
plt.xlim(lims)
plt.ylim(lims)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'actual_vs_predicted.png'))
plt.close()
print("Saved actual_vs_predicted.png")

# --- Plot 6: Residual Distribution ---
plt.figure(figsize=(9, 5), dpi=300)
sns.histplot(residuals, kde=True, bins=60, color='#2ca02c', edgecolor='black', alpha=0.6)
plt.axvline(0, color='red', linestyle='--', linewidth=1.8, label='Zero Error Reference')
plt.axvline(residuals.mean(), color='orange', linestyle='-', linewidth=1.5, label=f"Mean Residual: ${residuals.mean():,.2f}")
plt.title(f'Residual Distribution Curve ({champ_name} Regression)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Residual Error (Actual - Predicted) ($)', fontsize=11)
plt.ylabel('Density / Count', fontsize=11)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'residual_distribution.png'))
plt.close()
print("Saved residual_distribution.png")

# --- Plot 7: Residuals vs Predicted ---
plt.figure(figsize=(9, 6), dpi=300)
plt.scatter(champ_pred, residuals, alpha=0.25, color='#9467bd', edgecolors='none', s=20)
plt.axhline(0, color='red', linestyle='--', linewidth=1.8, label='Zero Residual Line')
plt.axhline(2 * residuals.std(), color='grey', linestyle=':', label='±2σ Confidence Boundary')
plt.axhline(-2 * residuals.std(), color='grey', linestyle=':')
plt.title(f'Residuals vs Predicted Values ({champ_name} Regression)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Predicted Median House Value ($)', fontsize=11)
plt.ylabel('Residuals (Actual - Predicted) ($)', fontsize=11)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'residuals_vs_predicted.png'))
plt.close()
print("Saved residuals_vs_predicted.png")

# --- Plot 8: Regularization Coefficients Comparison ---
fitted_prep = best_ridge.named_steps['preprocessor']
cat_cols = list(fitted_prep.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features))
all_features = num_features + cat_cols

coef_df = pd.DataFrame({
    'Feature': all_features,
    'OLS': ols_pipe.named_steps['regressor'].coef_,
    'Ridge': best_ridge.named_steps['regressor'].coef_,
    'Lasso': best_lasso.named_steps['regressor'].coef_
})

plt.figure(figsize=(12, 7), dpi=300)
x = np.arange(len(all_features))
width = 0.28
plt.bar(x - width, coef_df['OLS'], width, label='OLS', color='#2b5c8f', alpha=0.85)
plt.bar(x, coef_df['Ridge'], width, label=f'Ridge (α={best_ridge_alpha})', color='#e26d5c', alpha=0.85)
plt.bar(x + width, coef_df['Lasso'], width, label=f'Lasso (α={best_lasso_alpha})', color='#38b000', alpha=0.85)
plt.xticks(x, all_features, rotation=45, ha='right', fontsize=9)
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel('Standardized Coefficient Weight', fontsize=11)
plt.title('Feature Coefficient Comparison Across Regression Models', fontsize=13, fontweight='bold', pad=12)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'regularization_coefficients.png'))
plt.close()
print("Saved regularization_coefficients.png")

print("\nAll visualizations successfully generated and saved to images/ directory.")
