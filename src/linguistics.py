from collections import Counter
from typing import Iterable

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer

from .preprocessing import ensure_nltk_resources, preprocess


def stems(tokens: Iterable[str]):
    porter, snowball = PorterStemmer(), SnowballStemmer('english')
    return [porter.stem(t) for t in tokens], [snowball.stem(t) for t in tokens]


def spacy_lemmas(texts: Iterable[str]):
    try:
        import spacy
        try:
            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
        except OSError:
            return None, 'spaCy model en_core_web_sm is not installed'
        docs = nlp.pipe(texts)
        return [[tok.lemma_ for tok in doc if tok.is_alpha] for doc in docs], None
    except ImportError:
        return None, 'spaCy is not installed'


def top_tokens(df, n=50):
    counter = Counter(token for text in df['message'] for token in preprocess(text))
    return counter.most_common(n)
