from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from gensim.models import FastText, Word2Vec


def mean_embedding_for_tokens(
    tokens: Iterable[str],
    model_name: str,
    model,
    glove_vectors: np.ndarray | None = None,
    glove_vocab: Dict[str, int] | None = None,
) -> np.ndarray:
    """Build one document vector as the mean of available word vectors."""
    vectors = []
    if model_name == "GloVe":
        if glove_vectors is None or glove_vocab is None:
            raise ValueError("GloVe vectors and vocabulary are required.")
        for token in tokens:
            idx = glove_vocab.get(token)
            if idx is not None:
                vectors.append(glove_vectors[idx])
    else:
        keyed = model.wv
        for token in tokens:
            if token in keyed.key_to_index:
                vectors.append(keyed[token])

    if not vectors:
        return np.zeros(model.vector_size if model_name != "GloVe" else glove_vectors.shape[1], dtype=np.float32)

    vec = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def build_document_matrix(
    token_lists: Iterable[Iterable[str]],
    model_name: str,
    model,
    glove_vectors: np.ndarray | None = None,
    glove_vocab: Dict[str, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return document matrix and mask for documents with at least one known word."""
    rows = [
        mean_embedding_for_tokens(tokens, model_name, model, glove_vectors, glove_vocab)
        for tokens in token_lists
    ]
    matrix = np.vstack(rows).astype(np.float32)
    known_mask = np.linalg.norm(matrix, axis=1) > 0
    return matrix, known_mask


def load_week5_models(models_dir):
    import json
    from pathlib import Path
    import numpy as np

    models_dir = Path(models_dir)
    w2v = Word2Vec.load(str(models_dir / "week5_word2vec.model"))
    fasttext = FastText.load(str(models_dir / "week5_fasttext.model"))
    glove_data = np.load(models_dir / "week5_glove.npz")
    glove_vectors = glove_data["vectors"].astype(np.float32)
    with open(models_dir / "week5_glove_vocab.json", "r", encoding="utf-8") as f:
        glove_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    return {
        "Word2Vec": w2v,
        "fastText": fasttext,
        "GloVe": (glove_vectors, glove_vocab),
    }
