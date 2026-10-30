"""
Datheon Machine Learning & Data Science Internship - Task 1
Multi-Class Benchmark Classification Pipeline (Iris Dataset)
Script to execute the complete end-to-end ML pipeline and export all publication-grade figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 15

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'iris.csv')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)

print("="*60)
print("1. LOADING DATASET")
print("="*60)
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types & Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nClass Distribution:")
print(df['species'].value_counts())

feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_col = 'species'
classes = df[target_col].unique()

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS PLOTS
# ==========================================
print("\n" + "="*60)
print("2. GENERATING EDA VISUALIZATIONS")
print("="*60)

# 2.1 Feature Distribution Plots (Histograms with KDE & Boxplots)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
palette = {'setosa': '#4C72B0', 'versicolor': '#55A868', 'virginica': '#C44E52'}

for i, col in enumerate(feature_cols):
    # Top row: KDE & Histograms per species
    for sp in classes:
        sns.histplot(
            data=df[df[target_col] == sp],
            x=col,
            kde=True,
            ax=axes[0, i],
            color=palette[sp],
            label=sp,
            stat="density",
            alpha=0.4,
            element="step"
        )
    axes[0, i].set_title(f"Distribution: {col.replace('_', ' ').title()}", fontweight='bold')
    axes[0, i].set_xlabel(f"{col.replace('_', ' ').title()} (cm)")
    axes[0, i].set_ylabel("Density")
    if i == 0:
        axes[0, i].legend(title="Species")
    else:
        axes[0, i].get_legend().remove() if axes[0, i].get_legend() else None

    # Bottom row: Boxplots across species
    sns.boxplot(
        data=df,
        x=target_col,
        y=col,
        ax=axes[1, i],
        palette=palette,
        hue=target_col,
        legend=False,
        boxprops=dict(alpha=0.8)
    )
    axes[1, i].set_title(f"Boxplot: {col.replace('_', ' ').title()} by Class", fontweight='bold')
    axes[1, i].set_xlabel("Species")
    axes[1, i].set_ylabel(f"{col.replace('_', ' ').title()} (cm)")

plt.suptitle("Datheon Task 1: Exploratory Feature Distributions & Boxplots", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
dist_img_path = os.path.join(IMG_DIR, 'feature_distributions.png')
plt.savefig(dist_img_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {dist_img_path}")

# 2.2 Seaborn Pairplot
pairplot_fig = sns.pairplot(
    df,
    hue=target_col,
    palette=palette,
    diag_kind="kde",
    markers=["o", "s", "D"],
    plot_kws={'alpha': 0.8, 's': 50}
)
pairplot_fig.fig.suptitle("Datheon Task 1: Pairplot of Iris Morphological Features", y=1.02, fontsize=15, fontweight='bold')
pairplot_path = os.path.join(IMG_DIR, 'pairplot.png')
pairplot_fig.savefig(pairplot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {pairplot_path}")

# 2.3 Feature Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df[feature_cols]
corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=1.2,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
    square=True
)
plt.title("Datheon Task 1: Feature Pearson Correlation Heatmap", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
corr_img_path = os.path.join(IMG_DIR, 'correlation_heatmap.png')
plt.savefig(corr_img_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {corr_img_path}")

# ==========================================
# 3. PREPROCESSING & TRAIN-TEST SPLIT
# ==========================================
print("\n" + "="*60)
print("3. DATA PREPROCESSING & 80/20 STRATIFIED SPLIT")
print("="*60)

X = df[feature_cols]
y = df[target_col]

# 80/20 train/test split with stratification and reproducible seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training set features shape: {X_train.shape}, target shape: {y_train.shape}")
print(f"Testing set features shape:  {X_test.shape}, target shape:  {y_test.shape}")
print("\nTrain class distribution:\n", y_train.value_counts())
print("\nTest class distribution:\n", y_test.value_counts())

# ==========================================
# 4. MODEL TRAINING VIA SCIKIT-LEARN PIPELINES
# ==========================================
print("\n" + "="*60)
print("4. TRAINING MODELS VIA SCIKIT-LEARN PIPELINES")
print("="*60)

# Pipeline 1: Logistic Regression with StandardScaler
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Pipeline 2: Decision Tree Classifier
pipe_dt = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=3))
])

# Train models
pipe_lr.fit(X_train, y_train)
pipe_dt.fit(X_train, y_train)

# Predictions
y_pred_lr = pipe_lr.predict(X_test)
y_pred_dt = pipe_dt.predict(X_test)

# ==========================================
# 5. MODEL EVALUATION & BENCHMARKING
# ==========================================
print("\n" + "="*60)
print("5. MODEL EVALUATION BENCHMARK")
print("="*60)

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro')
    prec_weighted = precision_score(y_true, y_pred, average='weighted')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    rec_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(y_true, y_pred))
    
    return {
        'Model': name,
        'Accuracy': acc,
        'Precision (Macro)': prec_macro,
        'Recall (Macro)': rec_macro,
        'F1-Score (Macro)': f1_macro,
        'Precision (Weighted)': prec_weighted,
        'Recall (Weighted)': rec_weighted,
        'F1-Score (Weighted)': f1_weighted
    }

metrics_lr = evaluate_model("Logistic Regression", y_test, y_pred_lr)
metrics_dt = evaluate_model("Decision Tree Classifier", y_test, y_pred_dt)

benchmark_df = pd.DataFrame([metrics_lr, metrics_dt])
print("\n" + "="*60)
print("MODEL BENCHMARK COMPARISON TABLE")
print("="*60)
print(benchmark_df.to_string(index=False))

# ==========================================
# 6. CONFUSION MATRICES VISUALIZATION
# ==========================================
print("\n" + "="*60)
print("6. GENERATING CONFUSION MATRICES")
print("="*60)

cm_lr = confusion_matrix(y_test, y_pred_lr, labels=classes)
cm_dt = confusion_matrix(y_test, y_pred_dt, labels=classes)

# 6.1 Logistic Regression Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes,
    cbar=True,
    annot_kws={'size': 14, 'weight': 'bold'}
)
plt.title(f"Logistic Regression Confusion Matrix\nAccuracy: {metrics_lr['Accuracy']:.2%}", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Predicted Species", fontsize=12, fontweight='bold')
plt.ylabel("True Species", fontsize=12, fontweight='bold')
plt.tight_layout()
cm_lr_path = os.path.join(IMG_DIR, 'lr_confusion_matrix.png')
plt.savefig(cm_lr_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cm_lr_path}")

# 6.2 Decision Tree Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_dt,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=classes,
    yticklabels=classes,
    cbar=True,
    annot_kws={'size': 14, 'weight': 'bold'}
)
plt.title(f"Decision Tree Classifier Confusion Matrix\nAccuracy: {metrics_dt['Accuracy']:.2%}", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Predicted Species", fontsize=12, fontweight='bold')
plt.ylabel("True Species", fontsize=12, fontweight='bold')
plt.tight_layout()
cm_dt_path = os.path.join(IMG_DIR, 'dt_confusion_matrix.png')
plt.savefig(cm_dt_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {cm_dt_path}")

# 6.3 Performance Comparison Bar Chart
plt.figure(figsize=(10, 6))
plot_metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
x_indices = np.arange(len(plot_metrics))
width = 0.35

lr_vals = [metrics_lr[m] for m in plot_metrics]
dt_vals = [metrics_dt[m] for m in plot_metrics]

bars1 = plt.bar(x_indices - width/2, lr_vals, width, label='Logistic Regression', color='#4C72B0', edgecolor='black', linewidth=1)
bars2 = plt.bar(x_indices + width/2, dt_vals, width, label='Decision Tree', color='#55A868', edgecolor='black', linewidth=1)

# Add values above bars
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('Evaluation Metrics', fontsize=13, fontweight='bold')
plt.ylabel('Score (0.0 to 1.0)', fontsize=13, fontweight='bold')
plt.title('Datheon Task 1: Benchmark Comparison — Logistic Regression vs Decision Tree', fontsize=14, fontweight='bold', pad=15)
plt.xticks(x_indices, plot_metrics, fontsize=11, fontweight='bold')
plt.ylim(0, 1.15)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
plt.tight_layout()
comp_path = os.path.join(IMG_DIR, 'model_comparison.png')
plt.savefig(comp_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {comp_path}")

# ==========================================
# 7. DECISION BOUNDARIES VISUALIZATION (PCA)
# ==========================================
print("\n" + "="*60)
print("7. GENERATING DECISION BOUNDARY VISUALIZATIONS (PCA 2D)")
print("="*60)

# We use PCA to project the 4D feature space onto 2 principal components
# preserving the maximum variance of the feature manifold for 2D visualization.
pca = PCA(n_components=2, random_state=42)
scaler_pca = StandardScaler()

X_scaled_all = scaler_pca.fit_transform(X)
X_pca = pca.fit_transform(X_scaled_all)
var_ratio = pca.explained_variance_ratio_
print(f"PCA Explained Variance Ratio: PC1 = {var_ratio[0]:.2%}, PC2 = {var_ratio[1]:.2%}, Total = {sum(var_ratio):.2%}")

# Train models on 2D PCA projection for decision boundary visualization
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(
    X_pca, y, test_size=0.20, random_state=42, stratify=y
)

# Label encoding for contour mapping
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
y_pca_train_idx = np.array([class_to_idx[val] for val in y_pca_train])
y_pca_test_idx = np.array([class_to_idx[val] for val in y_pca_test])

lr_pca = LogisticRegression(random_state=42, max_iter=1000)
lr_pca.fit(X_pca_train, y_pca_train_idx)

dt_pca = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_pca.fit(X_pca_train, y_pca_train_idx)

# Create dense meshgrid
x_min, x_max = X_pca[:, 0].min() - 1.0, X_pca[:, 0].max() + 1.0
y_min, y_max = X_pca[:, 1].min() - 1.0, X_pca[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Colormaps for background & points
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#E0ECF8', '#E2F0D9', '#FBE5D6'])
cmap_bold = ['#1F77B4', '#2CA02C', '#D62728']

# Plot 1: Logistic Regression Boundary
Z_lr = lr_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lr = Z_lr.reshape(xx.shape)
axes[0].contourf(xx, yy, Z_lr, cmap=cmap_light, alpha=0.8)
for idx, cls in enumerate(classes):
    axes[0].scatter(
        X_pca_test[y_pca_test_idx == idx, 0],
        X_pca_test[y_pca_test_idx == idx, 1],
        c=cmap_bold[idx],
        label=f"{cls} (Test)",
        edgecolor='black',
        s=80,
        linewidth=1.2
    )
axes[0].set_title(f"Logistic Regression Decision Boundary\n(Test Acc: {accuracy_score(y_pca_test_idx, lr_pca.predict(X_pca_test)):.2%})", fontsize=13, fontweight='bold')
axes[0].set_xlabel(f"Principal Component 1 ({var_ratio[0]:.1%} Variance)", fontsize=11, fontweight='bold')
axes[0].set_ylabel(f"Principal Component 2 ({var_ratio[1]:.1%} Variance)", fontsize=11, fontweight='bold')
axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

# Plot 2: Decision Tree Boundary
Z_dt = dt_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z_dt = Z_dt.reshape(xx.shape)
axes[1].contourf(xx, yy, Z_dt, cmap=cmap_light, alpha=0.8)
for idx, cls in enumerate(classes):
    axes[1].scatter(
        X_pca_test[y_pca_test_idx == idx, 0],
        X_pca_test[y_pca_test_idx == idx, 1],
        c=cmap_bold[idx],
        label=f"{cls} (Test)",
        edgecolor='black',
        s=80,
        linewidth=1.2
    )
axes[1].set_title(f"Decision Tree Decision Boundary\n(Test Acc: {accuracy_score(y_pca_test_idx, dt_pca.predict(X_pca_test)):.2%})", fontsize=13, fontweight='bold')
axes[1].set_xlabel(f"Principal Component 1 ({var_ratio[0]:.1%} Variance)", fontsize=11, fontweight='bold')
axes[1].set_ylabel(f"Principal Component 2 ({var_ratio[1]:.1%} Variance)", fontsize=11, fontweight='bold')
axes[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.suptitle("Datheon Task 1: 2D Decision Boundary Landscapes via PCA Projection", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
boundary_img_path = os.path.join(IMG_DIR, 'decision_boundaries.png')
plt.savefig(boundary_img_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {boundary_img_path}")

print("\n" + "="*60)
print("EXECUTION COMPLETED SUCCESSFULLY!")
print("="*60)
