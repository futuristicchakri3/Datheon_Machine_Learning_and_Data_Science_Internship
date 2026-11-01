"""
Task 4: NLP Sentiment Classification on Customer Reviews using TF-IDF
Script: Benchmark Training & Publication Visualizations
Internship: Datheon Machine Learning & Data Science Internship
Author: G. Chakravarthi Shiva Sai (Intern ID: DT-2026-UKTYN)
"""

import os
import sys
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.text_preprocessing import (
    load_and_prepare_dataset,
    preprocess_text,
    ensure_nltk_resources
)

# Ensure NLTK resources
ensure_nltk_resources()

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

img_dir = os.path.join(current_dir, 'images')
os.makedirs(img_dir, exist_ok=True)
csv_path = os.path.join(current_dir, 'dataset', 'sentiment_reviews.csv')

# 1. Load data
raw_df, binary_df, y_binary, neutral_df = load_and_prepare_dataset(csv_path)

# Add preprocessed text and length metrics
raw_df['clean_text'] = raw_df['review_text'].apply(preprocess_text)
raw_df['char_count'] = raw_df['review_text'].apply(len)
raw_df['word_count'] = raw_df['review_text'].apply(lambda x: len(x.split()))
raw_df['clean_word_count'] = raw_df['clean_text'].apply(lambda x: len(x.split()) if x else 0)

binary_df['clean_text'] = binary_df['review_text'].apply(preprocess_text)
y = binary_df['label'].values


# =====================================================================
# PLOT 1: Sentiment Distribution Bar Chart
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
counts = raw_df['sentiment'].str.capitalize().value_counts()
colors = {'Positive': '#2a9d8f', 'Negative': '#e76f51', 'Neutral': '#4a5568'}
bar_colors = [colors.get(cat, '#2b5c8f') for cat in counts.index]

bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor='black', alpha=0.9, width=0.45)
total_reviews = len(raw_df)

for bar in bars:
    height = bar.get_height()
    pct = (height / total_reviews) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.15,
        f"{int(height)} ({pct:.0f}%)\n[Included]" if bar.get_x() < 1.5 else f"{int(height)} ({pct:.0f}%)\n[Excluded from Binary]",
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

ax.set_title('Task 4: Customer Review Sentiment Class Distribution (N=10)', fontsize=13, fontweight='bold', pad=14)
ax.set_ylabel('Number of Reviews', fontsize=11, fontweight='bold')
ax.set_xlabel('Sentiment Class', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.35)
ax.axhline(0, color='black', linewidth=0.8)

# Context annotation box
note_text = (
    "Dataset Notice:\n"
    "• Total records: 10 (Positive: 5, Negative: 4, Neutral: 1)\n"
    "• Neutral record isolated from binary model training\n"
    "• Binary classification corpus: 9 records (5 Pos, 4 Neg)"
)
ax.text(
    0.97, 0.94, note_text, transform=ax.transAxes,
    fontsize=9, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.95)
)

plt.tight_layout()
p1_path = os.path.join(img_dir, 'sentiment_distribution.png')
plt.savefig(p1_path, dpi=300)
plt.close()
print(f"Generated Plot 1: {p1_path}")


# =====================================================================
# PLOT 2: Review Length Distribution
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# 2A: Character Length by Sentiment
palette = {'positive': '#2a9d8f', 'negative': '#e76f51', 'neutral': '#4a5568'}
sns.barplot(
    data=raw_df, x='sentiment', y='char_count', hue='sentiment',
    palette=palette, edgecolor='black', alpha=0.85, ax=axes[0], legend=False
)
axes[0].set_title('Mean Character Length per Sentiment', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sentiment Class', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Character Count (Raw Text)', fontsize=11, fontweight='bold')
for p in axes[0].patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f'{h:.1f}', (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# 2B: Word Count before vs after text preprocessing
x_indices = np.arange(len(raw_df))
width = 0.38
axes[1].bar(x_indices - width/2, raw_df['word_count'], width, label='Raw Word Count', color='#2b5c8f', edgecolor='black', alpha=0.85)
axes[1].bar(x_indices + width/2, raw_df['clean_word_count'], width, label='Cleaned Token Count', color='#f4a261', edgecolor='black', alpha=0.85)
axes[1].set_title('Word Count Reduction via Text Preprocessing', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Review Index (ID 1-10)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Word / Token Count', fontsize=11, fontweight='bold')
axes[1].set_xticks(x_indices)
axes[1].set_xticklabels([f"R{rid}" for rid in raw_df['review_id']])
axes[1].legend(loc='upper right', frameon=True, facecolor='white')

plt.suptitle('Review Length Diagnostics & Preprocessing Token Reduction', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
p2_path = os.path.join(img_dir, 'review_length_distribution.png')
plt.savefig(p2_path, dpi=300)
plt.close()
print(f"Generated Plot 2: {p2_path}")


# =====================================================================
# PLOT 3: Word Frequency Chart
# =====================================================================
# Extract all tokens from binary reviews
all_tokens = []
pos_tokens = []
neg_tokens = []

for _, row in binary_df.iterrows():
    tokens = row['clean_text'].split()
    all_tokens.extend(tokens)
    if row['sentiment'] == 'positive':
        pos_tokens.extend(tokens)
    else:
        neg_tokens.extend(tokens)

word_counts = Counter(all_tokens)
top_words = word_counts.most_common(12)

words = [w[0] for w in top_words][::-1]
freqs = [w[1] for w in top_words][::-1]

# Determine word affinity
colors_word = []
for w in words:
    in_pos = w in pos_tokens
    in_neg = w in neg_tokens
    if in_pos and in_neg:
        colors_word.append('#4a5568')  # Shared
    elif in_pos:
        colors_word.append('#2a9d8f')  # Positive
    else:
        colors_word.append('#e76f51')  # Negative

fig, ax = plt.subplots(figsize=(9, 5.8), dpi=300)
bars = ax.barh(words, freqs, color=colors_word, edgecolor='black', alpha=0.85, height=0.6)
ax.set_title('Top Lemmatized Word Frequencies in Preprocessed Corpus', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Frequency in Dataset (Occurrences)', fontsize=11, fontweight='bold')
ax.set_ylabel('Lemmatized Token', fontsize=11, fontweight='bold')
ax.set_xlim(0, max(freqs) + 1.2)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.08, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va='center', fontsize=10, fontweight='bold')

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2a9d8f', edgecolor='black', label='Positive Review Token'),
    Patch(facecolor='#e76f51', edgecolor='black', label='Negative Review Token'),
    Patch(facecolor='#4a5568', edgecolor='black', label='Appears in Both Classes')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', fontsize=9)

plt.tight_layout()
p3_path = os.path.join(img_dir, 'word_frequency.png')
plt.savefig(p3_path, dpi=300)
plt.close()
print(f"Generated Plot 3: {p3_path}")


# =====================================================================
# TF-IDF Feature Extraction & Model Fitting
# =====================================================================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
X_tfidf = tfidf.fit_transform(binary_df['clean_text'])

mnb = MultinomialNB(alpha=1.0)
lr = LogisticRegression(random_state=42, max_iter=1000)

mnb.fit(X_tfidf, y)
lr.fit(X_tfidf, y)

# Compute in-sample predictions
y_pred_mnb = mnb.predict(X_tfidf)
y_pred_lr = lr.predict(X_tfidf)

# Compute Leave-One-Out Cross-Validation predictions for honest generalization analysis
loo = LeaveOneOut()
loo_pred_mnb = cross_val_predict(mnb, X_tfidf, y, cv=loo)
loo_pred_lr = cross_val_predict(lr, X_tfidf, y, cv=loo)

# Calculate metrics
def compute_all_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

metrics_mnb_insample = compute_all_metrics(y, y_pred_mnb)
metrics_lr_insample = compute_all_metrics(y, y_pred_lr)

metrics_mnb_loocv = compute_all_metrics(y, loo_pred_mnb)
metrics_lr_loocv = compute_all_metrics(y, loo_pred_lr)

# Train/Test Split (7 train, 2 test) demonstration
X_train_txt, X_test_txt, y_train_split, y_test_split = train_test_split(
    binary_df['clean_text'], y, test_size=0.22, random_state=42, stratify=y
)
split_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
X_train_vec = split_vec.fit_transform(X_train_txt)
X_test_vec = split_vec.transform(X_test_txt)

mnb_split = MultinomialNB(alpha=1.0).fit(X_train_vec, y_train_split)
lr_split = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_vec, y_train_split)

pred_mnb_split = mnb_split.predict(X_test_vec)
pred_lr_split = lr_split.predict(X_test_vec)

metrics_mnb_split = compute_all_metrics(y_test_split, pred_mnb_split)
metrics_lr_split = compute_all_metrics(y_test_split, pred_lr_split)


# =====================================================================
# PLOT 4: Multinomial Naive Bayes Confusion Matrix
# =====================================================================
cm_mnb = confusion_matrix(y, y_pred_mnb, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
sns.heatmap(
    cm_mnb, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
    xticklabels=['Negative (0)', 'Positive (1)'],
    yticklabels=['Negative (0)', 'Positive (1)'],
    annot_kws={'fontsize': 14, 'fontweight': 'bold'}
)
ax.set_title(
    f"Multinomial Naive Bayes Confusion Matrix\n"
    f"(In-Sample Resubstitution: N=9 | Accuracy: {metrics_mnb_insample['Accuracy']*100:.0f}%)",
    fontsize=12, fontweight='bold', pad=12
)
ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax.set_ylabel('True Label', fontsize=11, fontweight='bold')

# Footnote on sample size
plt.figtext(
    0.5, 0.01,
    "Note: Resubstitution matrix demonstrates pipeline fit on 9 binary records (5 Pos, 4 Neg).\n"
    f"LOOCV Accuracy: {metrics_mnb_loocv['Accuracy']*100:.1f}% | 70/30 Split Test Accuracy: {metrics_mnb_split['Accuracy']*100:.1f}%",
    ha='center', fontsize=8, color='#555555'
)

plt.tight_layout()
p4_path = os.path.join(img_dir, 'confusion_matrix_naive_bayes.png')
plt.savefig(p4_path, dpi=300)
plt.close()
print(f"Generated Plot 4: {p4_path}")


# =====================================================================
# PLOT 5: Logistic Regression Confusion Matrix
# =====================================================================
cm_lr = confusion_matrix(y, y_pred_lr, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
sns.heatmap(
    cm_lr, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax,
    xticklabels=['Negative (0)', 'Positive (1)'],
    yticklabels=['Negative (0)', 'Positive (1)'],
    annot_kws={'fontsize': 14, 'fontweight': 'bold'}
)
ax.set_title(
    f"Logistic Regression Confusion Matrix\n"
    f"(In-Sample Resubstitution: N=9 | Accuracy: {metrics_lr_insample['Accuracy']*100:.0f}%)",
    fontsize=12, fontweight='bold', pad=12
)
ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax.set_ylabel('True Label', fontsize=11, fontweight='bold')

plt.figtext(
    0.5, 0.01,
    "Note: Resubstitution matrix demonstrates pipeline fit on 9 binary records (5 Pos, 4 Neg).\n"
    f"LOOCV Accuracy: {metrics_lr_loocv['Accuracy']*100:.1f}% | 70/30 Split Test Accuracy: {metrics_lr_split['Accuracy']*100:.1f}%",
    ha='center', fontsize=8, color='#555555'
)

plt.tight_layout()
p5_path = os.path.join(img_dir, 'confusion_matrix_logistic_regression.png')
plt.savefig(p5_path, dpi=300)
plt.close()
print(f"Generated Plot 5: {p5_path}")


# =====================================================================
# PLOT 6: Model Comparison Chart
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metric_names))
w = 0.35

# 6A: In-Sample Performance Benchmark (Demonstrating Pipeline Fit)
axes[0].bar(x - w/2, [metrics_mnb_insample[m] for m in metric_names], w,
            label='Multinomial Naive Bayes', color='#2b5c8f', edgecolor='black', alpha=0.85)
axes[0].bar(x + w/2, [metrics_lr_insample[m] for m in metric_names], w,
            label='Logistic Regression', color='#2a9d8f', edgecolor='black', alpha=0.85)

for p in axes[0].patches:
    h = p.get_height()
    axes[0].annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.02),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[0].set_title('In-Sample Fit Benchmark (N=9 Training Reviews)', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metric_names, fontsize=10, fontweight='bold')
axes[0].set_ylabel('Score (0.0 - 1.0)', fontsize=11, fontweight='bold')
axes[0].set_ylim(0, 1.2)
axes[0].legend(loc='lower right', frameon=True, facecolor='white')

# 6B: Leave-One-Out Cross-Validation Benchmark (Honest Generalization Assessment)
axes[1].bar(x - w/2, [metrics_mnb_loocv[m] for m in metric_names], w,
            label='Multinomial Naive Bayes (LOOCV)', color='#2b5c8f', edgecolor='black', alpha=0.85)
axes[1].bar(x + w/2, [metrics_lr_loocv[m] for m in metric_names], w,
            label='Logistic Regression (LOOCV)', color='#e76f51', edgecolor='black', alpha=0.85)

for p in axes[1].patches:
    h = p.get_height()
    axes[1].annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.02),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[1].set_title('Leave-One-Out Cross-Validation (LOOCV Generalization)', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metric_names, fontsize=10, fontweight='bold')
axes[1].set_ylabel('Score (0.0 - 1.0)', fontsize=11, fontweight='bold')
axes[1].set_ylim(0, 1.2)
axes[1].legend(loc='upper right', frameon=True, facecolor='white')

plt.suptitle('Model Performance Comparison: Naive Bayes vs Logistic Regression', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
p6_path = os.path.join(img_dir, 'model_comparison.png')
plt.savefig(p6_path, dpi=300)
plt.close()
print(f"Generated Plot 6: {p6_path}")

print("\n--- All 6 Publication Visualizations Generated Successfully ---")
