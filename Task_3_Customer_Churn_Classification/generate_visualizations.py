"""
Task 3: Predictive Customer Churn Classification
Script: Benchmark Training & Publication Visualizations
Internship: Datheon Machine Learning & Data Science Internship
Author: G. Chakravarthi Shiva Sai (Intern ID: DT-2026-UKTYN)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.preprocessing import (
    load_and_clean_data,
    build_preprocessor,
    get_feature_names,
    evaluate_classifier,
)

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

img_dir = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(img_dir, exist_ok=True)

# 1. Load data
csv_path = os.path.join(os.path.dirname(__file__), 'dataset', 'Telco-Customer-Churn.csv')
raw_df = pd.read_csv(csv_path)

# --- Plot 1: Churn Class Distribution ---
plt.figure(figsize=(8, 5), dpi=300)
counts = raw_df['Churn'].value_counts()
colors = ['#2b5c8f', '#e63946']
bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.85, width=0.5)
plt.title('Telco Customer Churn Class Distribution (Target Imbalance)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Customer Churn Status', fontsize=11)
plt.ylabel('Number of Customers', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    pct = yval / len(raw_df) * 100
    plt.text(bar.get_x() + bar.get_width()/2, yval + 80, f"{yval:,} ({pct:.1f}%)", ha='center', fontsize=10, fontweight='bold')
plt.ylim(0, max(counts.values) + 600)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'churn_distribution.png'))
plt.close()
print("Saved churn_distribution.png")

# --- Plot 2: Correlation Heatmap ---
clean_df = raw_df.copy()
clean_df['TotalCharges'] = pd.to_numeric(clean_df['TotalCharges'].str.strip(), errors='coerce').fillna(0.0)
clean_df['Churn_Numeric'] = (clean_df['Churn'] == 'Yes').astype(int)
num_cols_corr = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_Numeric']
corr_matrix = clean_df[num_cols_corr].corr()

plt.figure(figsize=(8, 6.5), dpi=300)
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
    cbar_kws={'label': 'Pearson Correlation'}
)
plt.title('Correlation Heatmap: Numerical Features & Churn', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'correlation_heatmap.png'))
plt.close()
print("Saved correlation_heatmap.png")

# 2. Data Cleaning & Train-Test Split
X, y, num_cols, cat_cols = load_and_clean_data(csv_path)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = build_preprocessor(num_cols, cat_cols)

# 3. Model 1: Random Forest Classifier
rf_pipe = ImbPipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'clf__n_estimators': [100, 150],
    'clf__max_depth': [6, 10],
    'clf__min_samples_split': [2, 5],
    'clf__class_weight': ['balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Tuning Random Forest...")
rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
y_pred_rf = best_rf.predict(X_test)
metrics_rf = evaluate_classifier(y_test, y_pred_rf, y_prob_rf, "Random Forest (Tuned)")

# 4. Model 2: XGBoost Classifier
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_pipe = ImbPipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
])

xgb_param_grid = {
    'clf__n_estimators': [100, 150],
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.03, 0.1],
    'clf__scale_pos_weight': [1.0, scale_pos]
}

print("Tuning XGBoost...")
xgb_grid = GridSearchCV(xgb_pipe, xgb_param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = best_xgb.predict(X_test)
metrics_xgb = evaluate_classifier(y_test, y_pred_xgb, y_prob_xgb, "XGBoost (Tuned)")

# --- Plot 3: ROC-AUC Curve Comparison ---
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = auc(fpr_rf, tpr_rf)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(8.5, 6.5), dpi=300)
plt.plot(fpr_rf, tpr_rf, color='#2b5c8f', linewidth=2.2, label=f"Random Forest (AUC = {auc_rf:.4f})")
plt.plot(fpr_xgb, tpr_xgb, color='#e63946', linewidth=2.2, label=f"XGBoost (AUC = {auc_xgb:.4f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Chance Baseline (AUC = 0.5000)')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'roc_curve_comparison.png'))
plt.close()
print("Saved roc_curve_comparison.png")

# --- Plot 4: Random Forest Confusion Matrix ---
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6.5, 5.5), dpi=300)
sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['Retained (No)', 'Churned (Yes)'],
    yticklabels=['Retained (No)', 'Churned (Yes)'],
    annot_kws={'fontsize': 13, 'fontweight': 'bold'}
)
plt.title(f'Random Forest Confusion Matrix\n(Accuracy: {metrics_rf["Accuracy"]*100:.2f}% | Recall: {metrics_rf["Recall"]*100:.2f}%)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('True Label', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'random_forest_confusion_matrix.png'))
plt.close()
print("Saved random_forest_confusion_matrix.png")

# --- Plot 5: XGBoost Confusion Matrix ---
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6.5, 5.5), dpi=300)
sns.heatmap(
    cm_xgb,
    annot=True,
    fmt='d',
    cmap='Reds',
    cbar=False,
    xticklabels=['Retained (No)', 'Churned (Yes)'],
    yticklabels=['Retained (No)', 'Churned (Yes)'],
    annot_kws={'fontsize': 13, 'fontweight': 'bold'}
)
plt.title(f'XGBoost Confusion Matrix\n(Accuracy: {metrics_xgb["Accuracy"]*100:.2f}% | Precision: {metrics_xgb["Precision"]*100:.2f}%)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('True Label', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'xgboost_confusion_matrix.png'))
plt.close()
print("Saved xgboost_confusion_matrix.png")

# Feature Name Extraction
fitted_prep = best_rf.named_steps['prep']
all_feature_names = get_feature_names(fitted_prep, num_cols, cat_cols)

# --- Plot 6: Random Forest Feature Importance ---
rf_importances = best_rf.named_steps['clf'].feature_importances_
rf_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': rf_importances}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(9, 5.5), dpi=300)
bars = plt.barh(rf_imp_df['Feature'][::-1], rf_imp_df['Importance'][::-1], color='#2b5c8f', edgecolor='black', alpha=0.85)
plt.title('Top 10 Influential Features: Random Forest Classifier', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Gini Feature Importance', fontsize=11)
plt.ylabel('Feature', fontsize=11)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=9, fontweight='bold')
plt.xlim(0, max(rf_imp_df['Importance']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'random_forest_feature_importance.png'))
plt.close()
print("Saved random_forest_feature_importance.png")

# --- Plot 7: XGBoost Feature Importance ---
xgb_importances = best_xgb.named_steps['clf'].feature_importances_
xgb_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': xgb_importances}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(9, 5.5), dpi=300)
bars = plt.barh(xgb_imp_df['Feature'][::-1], xgb_imp_df['Importance'][::-1], color='#e63946', edgecolor='black', alpha=0.85)
plt.title('Top 10 Influential Features: XGBoost Classifier', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Gain / Weight Feature Importance', fontsize=11)
plt.ylabel('Feature', fontsize=11)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=9, fontweight='bold')
plt.xlim(0, max(xgb_imp_df['Importance']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'xgboost_feature_importance.png'))
plt.close()
print("Saved xgboost_feature_importance.png")

# --- Plot 8: Model Comparison ---
comp_df = pd.DataFrame([metrics_rf, metrics_xgb])
print("\nFinal Test Metrics Comparison:\n", comp_df)

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(10, 6), dpi=300)
bars1 = plt.bar(x - width/2, [metrics_rf[m] for m in metric_names], width, label='Random Forest (Balanced)', color='#2b5c8f', edgecolor='black', alpha=0.85)
bars2 = plt.bar(x + width/2, [metrics_xgb[m] for m in metric_names], width, label='XGBoost', color='#e63946', edgecolor='black', alpha=0.85)

for bar in bars1:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.3f}", ha='center', fontsize=9, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.3f}", ha='center', fontsize=9, fontweight='bold')

plt.xticks(x, metric_names, fontsize=11)
plt.ylabel('Metric Score (0.0 - 1.0)', fontsize=11)
plt.ylim(0, 1.05)
plt.title('Performance Comparison: Random Forest vs XGBoost Classifier', fontsize=13, fontweight='bold', pad=12)
plt.axhline(0.80, color='green', linestyle='--', linewidth=1.2, label='Assignment Benchmark Target (>80% ROC-AUC)')
plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'model_comparison.png'))
plt.close()
print("Saved model_comparison.png")

print("\nAll visualizations successfully created and saved in images/ directory.")
