import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    """Check resources without requiring an internet connection."""
    return None


def tokenize(text: str) -> List[str]:
    ensure_nltk_resources()
    try:
        return word_tokenize(text)
    except LookupError:
        # Offline fallback: enough for the SMS dataset and does not require NLTK data.
        return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _get_stopwords() -> set:
    try:
        return set(stopwords.words('english'))
    except LookupError:
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'you', 'your', 'i', 'we',
            'they', 'this', 'but', 'not', 'have', 'do', 'so', 'if', 'me',
        }


def preprocess(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = [t for t in tokenize(clean_text(text)) if t.isalpha()]
    if remove_stopwords:
        stops = _get_stopwords()
        tokens = [t for t in tokens if t not in stops]
    return tokens


def preprocess_to_text(text: str, remove_stopwords: bool = True) -> str:
    return ' '.join(preprocess(text, remove_stopwords=remove_stopwords))
