from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def build_sentences(messages: Iterable[str], preprocess_fn) -> List[List[str]]:
    """Convert raw SMS messages into token lists used for all embedding models."""
    return [preprocess_fn(message) for message in messages]


def build_vocab(sentences: List[List[str]], min_count: int = 2):
    counts = Counter(word for sentence in sentences for word in sentence)
    vocab = {word: idx for idx, (word, count) in enumerate(
        sorted(((w, c) for w, c in counts.items() if c >= min_count),
               key=lambda x: (-x[1], x[0]))
    )}
    return vocab, counts


def build_cooccurrence(
    sentences: List[List[str]],
    vocab: Dict[str, int],
    window_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse GloVe co-occurrence triples (row, col, value).

    Distance weighting is 1/distance, as in the original GloVe formulation.
    """
    cooc = defaultdict(float)
    for sentence in sentences:
        ids = [vocab[w] for w in sentence if w in vocab]
        for i, center in enumerate(ids):
            left = max(0, i - window_size)
            right = min(len(ids), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context = ids[j]
                distance = abs(i - j)
                cooc[(center, context)] += 1.0 / distance

    rows = np.fromiter((k[0] for k in cooc), dtype=np.int32)
    cols = np.fromiter((k[1] for k in cooc), dtype=np.int32)
    values = np.fromiter(cooc.values(), dtype=np.float32)
    return rows, cols, values


def train_glove(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    vocab_size: int,
    vector_size: int = 50,
    epochs: int = 20,
    learning_rate: float = 0.05,
    x_max: float = 100.0,
    alpha: float = 0.75,
    seed: int = 42,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a compact GloVe model with mini-batch AdaGrad.

    The implementation keeps the corpus sparse and uses vectorized batches,
    which makes it practical for the SMS dataset without an external GloVe package.
    """
    rng = np.random.default_rng(seed)
    scale = 0.5 / vector_size
    W = rng.uniform(-scale, scale, size=(vocab_size, vector_size)).astype(np.float32)
    C = rng.uniform(-scale, scale, size=(vocab_size, vector_size)).astype(np.float32)
    bW = np.zeros(vocab_size, dtype=np.float32)
    bC = np.zeros(vocab_size, dtype=np.float32)

    gW = np.ones_like(W, dtype=np.float32)
    gC = np.ones_like(C, dtype=np.float32)
    gbW = np.ones_like(bW, dtype=np.float32)
    gbC = np.ones_like(bC, dtype=np.float32)

    order = np.arange(len(values))
    for _ in range(epochs):
        rng.shuffle(order)
        for begin in range(0, len(order), batch_size):
            idx = order[begin:begin + batch_size]
            r = rows[idx]
            c = cols[idx]
            x = values[idx]

            wi = W[r].copy()
            cj = C[c].copy()
            weight = np.minimum((x / x_max) ** alpha, 1.0)
            diff = np.sum(wi * cj, axis=1) + bW[r] + bC[c] - np.log(x)
            grad = (weight * diff).astype(np.float32)
            grad_w = grad[:, None] * cj
            grad_c = grad[:, None] * wi

            acc_w = np.zeros_like(W)
            acc_c = np.zeros_like(C)
            acc_bw = np.zeros_like(bW)
            acc_bc = np.zeros_like(bC)
            np.add.at(acc_w, r, grad_w)
            np.add.at(acc_c, c, grad_c)
            np.add.at(acc_bw, r, grad)
            np.add.at(acc_bc, c, grad)

            W -= learning_rate * acc_w / np.sqrt(gW)
            C -= learning_rate * acc_c / np.sqrt(gC)
            bW -= learning_rate * acc_bw / np.sqrt(gbW)
            bC -= learning_rate * acc_bc / np.sqrt(gbC)

            gW += acc_w * acc_w
            gC += acc_c * acc_c
            gbW += acc_bw * acc_bw
            gbC += acc_bc * acc_bc

    return W + C, bW + bC


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)
