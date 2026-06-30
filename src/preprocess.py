import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import LABELS_PATH

_PATTERNS = [
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), " <IP_ADDR> "), 
    (re.compile(r"\b\d{4}-\d{2}-\d{2}T[\d:]+Z?\b"), " <TIMESTAMP> "), 
    (re.compile(r"\b(txn|rec|usr|client)_[a-zA-Z0-9]+\b", re.I), " <ID> "), 
    (re.compile(r"\b\d+(\.\d+)?\s*(ms|s|mb|gb|rps|rpm|kb)\b", re.I), " <METRIC> "),       
    (re.compile(r"\b\d+/\d+\b"), " <RATIO> "),                                                
    (re.compile(r"\b\d+\b"), " <NUM> "),                                                      
]

def clean_log_message(text: str) -> str:
    """Cleans raw log text payloads by normalizing variable parameters into standard tokens."""
    if not isinstance(text, str):
        return ""
    
    # 1. Standardize special bounding brackets out of string context
    text = re.sub(r'[\[\]\(\):,\-]', ' ', text)
    
    # 2. Iterate through and apply our regular expression mappings sequentially
    for pattern, substitution in _PATTERNS:
        text = pattern.sub(substitution, text)
        
    # 3. Lowercase and collapse variable whitespaces
    text = text.lower()
    return " ".join(text.split())


def prepare_data(df: pd.DataFrame, is_training: bool = True, vectorizer=None, encoder=None):
    """Preprocesses raw log data frames for model interaction."""
    df['cleaned_message'] = df['log_message'].apply(clean_log_message)
    df['combined_features'] = df['service'].str.lower() + " " + df['severity'].str.lower() + " " + df['cleaned_message']
    
    if is_training:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
        X = vectorizer.fit_transform(df['combined_features'])
        encoder = LabelEncoder()
        
        labels_df = pd.read_csv(LABELS_PATH) 
        
        encoder.fit(labels_df['id']) 
        y = encoder.transform(df['root_cause_label'])
        return X, y, vectorizer, encoder
    else:
        X = vectorizer.transform(df['combined_features'])
        return X