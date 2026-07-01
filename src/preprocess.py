import re
from typing import Any
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

_PATTERNS = [
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), " <IP_ADDR> "), 
    (re.compile(r"\b\d{4}-\d{2}-\d{2}T[\d:]+Z?\b"), " <TIMESTAMP> "), 
    (re.compile(r"\b(txn|rec|usr|client)_[a-zA-Z0-9]+\b", re.I), " <ID> "), 
    (re.compile(r"\b\d+(\.\d+)?\s*(ms|s|mb|gb|rps|rpm|kb)\b", re.I), " <METRIC> "),       
    (re.compile(r"\b\d+/\d+\b"), " <RATIO> "),                                                
    (re.compile(r"\b\d+\b"), " <NUM> "),                                                      
]

def clean_log_message(text: str) -> str:
    """Tidies up a raw log message by replacing things like IP addresses, numbers,
    and timestamps with generic placeholders. This helps the model focus on the
    type of error rather than the specific values that change every time."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'[\[\]\(\):,\-]', ' ', text)    
    for pattern, substitution in _PATTERNS:
        text = pattern.sub(substitution, text)
    text = text.lower()
    return " ".join(text.split())


def _build_keyword_rows(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Creates extra training examples from the known keywords associated with each
    root cause category. This gives the model direct exposure to the vocabulary
    it should associate with each label, even if those exact words are rare in
    the real log dataset."""
    rows = []
    for _, row in labels_df.iterrows():
        keywords = str(row.get('example_keywords', ''))
        if keywords.strip():
            rows.append({
                'service': 'keyword_signal',
                'log_message': keywords,
                'root_cause_label': row['id']
            })
    return pd.DataFrame(rows)


def fit_transform_data(
    df: pd.DataFrame, labels_df: pd.DataFrame
) -> tuple[Any, Any, TfidfVectorizer, LabelEncoder]:
    """Prepares the training data and teaches the model what to look for.
    Combines real log examples with keyword hints from the root cause reference sheet,
    then builds the word-scoring and label-mapping tools the model needs."""
    keyword_rows = _build_keyword_rows(labels_df)
    df = pd.concat([df[['service', 'log_message', 'root_cause_label']], keyword_rows], ignore_index=True)

    df['cleaned_message'] = df['log_message'].apply(clean_log_message)
    df['combined_features'] = df['service'].str.lower() + " " + df['cleaned_message']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
    X = vectorizer.fit_transform(df['combined_features'])

    encoder = LabelEncoder()
    encoder.fit(labels_df['id'])
    y = encoder.transform(df['root_cause_label'])

    return X, y, vectorizer, encoder


def transform_data(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> Any:
    """Prepares incoming log records for prediction using the word-scoring tool
    that was built during training. Does not change or retrain anything."""
    df['cleaned_message'] = df['log_message'].apply(clean_log_message)
    df['combined_features'] = df['service'].str.lower() + " " + df['cleaned_message']
    return vectorizer.transform(df['combined_features'])


def prepare_data(df: pd.DataFrame, is_training: bool = True, vectorizer=None, encoder=None, labels_df=None):
    """General-purpose log data preparation used by the notebook.
    When training, it builds and returns the word-scoring and label-mapping tools.
    When predicting, it uses the already-built tools to prepare the incoming logs."""
    if is_training and labels_df is not None:
        keyword_rows = _build_keyword_rows(labels_df)
        df = pd.concat([df[['service', 'log_message', 'root_cause_label']], keyword_rows], ignore_index=True)

    df['cleaned_message'] = df['log_message'].apply(clean_log_message)
    df['combined_features'] = df['service'].str.lower() + " " + df['cleaned_message']

    if is_training:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
        X = vectorizer.fit_transform(df['combined_features'])
        encoder = LabelEncoder()
        if labels_df is not None:
            encoder.fit(labels_df['id'])
        y = encoder.transform(df['root_cause_label'])
        return X, y, vectorizer, encoder
    else:
        X = vectorizer.transform(df['combined_features'])
        return X