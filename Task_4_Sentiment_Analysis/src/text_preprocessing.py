"""
Task 4: NLP Sentiment Classification on Customer Reviews using TF-IDF
Module: Text Preprocessing and NLP Pipeline Utilities
Internship: Datheon Machine Learning & Data Science Internship
Author: G. Chakravarthi Shiva Sai (Intern ID: DT-2026-UKTYN)
"""

import os
import re
import warnings
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


# --- Programmatic NLTK Resource Management ---

def ensure_nltk_resources() -> None:
    """
    Verify and programmatically download required NLTK corpora and models.
    Ensures safe, headless, and offline-compatible execution.
    """
    required_packages = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in required_packages:
        try:
            nltk.data.find(path)
        except (LookupError, OSError):
            try:
                import socket
                socket.setdefaulttimeout(3.0)
                nltk.download(pkg, quiet=True)
            except Exception as e:
                warnings.warn(f"Notice: Failed to download NLTK resource '{pkg}': {e}")


# Run resource verification upon module import
ensure_nltk_resources()


# --- Domain Lexicon & Negation Constants ---

# Negation tokens are critical for preserving semantic polarity inversion in sentiment analysis
NEGATION_WORDS = frozenset({
    "not", "no", "nor", "neither", "never", "none", "cannot", "cant", "without",
    "barely", "hardly", "scarcely", "nothing", "nowhere", "hardly", "dont", "doesnt",
    "didnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt", "wont",
    "wouldnt", "shouldnt", "couldnt", "mustnt"
})

# Standard English stopwords excluding polarity-altering negation tokens
try:
    _raw_stopwords = set(stopwords.words("english"))
except Exception:
    _raw_stopwords = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
    }

STOP_WORDS = frozenset(_raw_stopwords - NEGATION_WORDS)
LEMMATIZER = WordNetLemmatizer()


# --- Linguistic Helpers ---

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Map Penn Treebank Part-of-Speech (POS) tags to WordNet POS constants.
    Enables context-aware lemmatization for verbs, adjectives, nouns, and adverbs.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_html(text: str) -> str:
    """
    Strip HTML tags, scripts, and entities from raw text.
    Uses BeautifulSoup with a robust regex fallback.
    """
    if not isinstance(text, str):
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)


def remove_punctuation_and_special(text: str) -> str:
    """
    Remove punctuation and non-alphabetic characters while preserving inter-word whitespace.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def preprocess_text(text: Union[str, Any]) -> str:
    """
    Complete end-to-end text preprocessing pipeline:
    1. Type validation and null checking.
    2. HTML tag removal.
    3. Lowercasing.
    4. Punctuation and non-alphabetic character removal.
    5. Tokenization via NLTK word_tokenize.
    6. POS-aware lemmatization via NLTK WordNetLemmatizer.
    7. Negation-preserving stopword filtering.
    8. Filtering out single-character tokens.

    Parameters:
        text (str): Raw review text or user prompt.

    Returns:
        str: Cleaned, normalized, space-delimited text ready for TF-IDF vectorization.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    # 1. Strip HTML tags
    cleaned = clean_html(text)

    # 2. Lowercase normalization
    cleaned = cleaned.lower()

    # 3. Punctuation removal
    cleaned = remove_punctuation_and_special(cleaned)

    # 4. Tokenization
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()

    if not tokens:
        return ""

    # 5. Context-aware POS tagging and lemmatization
    try:
        pos_tags = nltk.pos_tag(tokens)
    except Exception:
        pos_tags = [(tok, "NN") for tok in tokens]

    lemmatized_tokens: List[str] = []
    for word, tag in pos_tags:
        # Check stopword exclusion while strictly retaining negation tokens
        if word in STOP_WORDS:
            continue
        if len(word) <= 1:
            continue
        try:
            lemma = LEMMATIZER.lemmatize(word, pos=get_wordnet_pos(tag))
        except Exception:
            lemma = LEMMATIZER.lemmatize(word)
        lemmatized_tokens.append(lemma)

    return " ".join(lemmatized_tokens)


# --- Scikit-Learn Compatible Transformer ---

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer wrapping `preprocess_text`.
    Enables seamless integration inside `sklearn.pipeline.Pipeline`.
    """

    def __init__(self, preserve_negations: bool = True):
        self.preserve_negations = preserve_negations

    def fit(self, X: Any, y: Optional[Any] = None) -> "TextPreprocessor":
        return self

    def transform(self, X: Union[pd.Series, List[str], np.ndarray]) -> List[str]:
        if isinstance(X, pd.Series):
            return [preprocess_text(val) for val in X]
        elif isinstance(X, (list, tuple, np.ndarray)):
            return [preprocess_text(val) for val in X]
        return [preprocess_text(str(X))]


# --- Dataset Ingestion and Validation Utilities ---

def load_and_prepare_dataset(
    filepath: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the supplied `sentiment_reviews.csv`, validate columns and shapes,
    and split into binary classification data while isolating the neutral record.

    Parameters:
        filepath (str): Path to sentiment_reviews.csv

    Returns:
        raw_df (pd.DataFrame): The un-modified 10-record dataset.
        binary_df (pd.DataFrame): The 9-record subset (positive & negative).
        y_binary (pd.Series): Target labels (1 for positive, 0 for negative).
        neutral_df (pd.DataFrame): The isolated 1-record neutral sample.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    raw_df = pd.read_csv(filepath)

    expected_cols = {"review_id", "review_text", "sentiment"}
    if not expected_cols.issubset(raw_df.columns):
        raise ValueError(
            f"Dataset must contain columns: {expected_cols}. Found: {list(raw_df.columns)}"
        )

    # Separate neutral records
    neutral_mask = raw_df["sentiment"].str.strip().str.lower() == "neutral"
    neutral_df = raw_df[neutral_mask].copy()

    # Binary dataset for 2-class sentiment analysis
    binary_df = raw_df[~neutral_mask].copy()
    label_map = {"positive": 1, "negative": 0}
    binary_df["label"] = binary_df["sentiment"].str.strip().str.lower().map(label_map)

    # Validate mapping integrity
    if binary_df["label"].isnull().any():
        unknown_labels = binary_df[binary_df["label"].isnull()]["sentiment"].tolist()
        raise ValueError(f"Unrecognized sentiment labels detected: {unknown_labels}")

    binary_df["label"] = binary_df["label"].astype(int)
    y_binary = binary_df["label"]

    return raw_df, binary_df, y_binary, neutral_df
