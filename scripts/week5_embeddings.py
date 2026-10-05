from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec

from src.data import load_sms_data
from src.embeddings import (
    build_cooccurrence,
    build_sentences,
    build_vocab,
    normalize_rows,
    train_glove,
)
from src.preprocessing import preprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
IMAGES = ROOT / "outputs" / "images"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
VECTOR_SIZE = 30
MIN_COUNT = 2
WINDOW = 4


def train_models(sentences):
    results = []

    start = time.perf_counter()
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=1,
        sg=1,
        negative=10,
        epochs=10,
        seed=SEED,
    )
    w2v_time = time.perf_counter() - start
    w2v.save(str(MODELS / "week5_word2vec.model"))
    w2v.wv.save(str(MODELS / "week5_word2vec.kv"))
    results.append({
        "model": "Word2Vec",
        "vector_size": VECTOR_SIZE,
        "vocabulary": len(w2v.wv),
        "training_seconds": round(w2v_time, 3),
        "notes": "Gensim Word2Vec, skip-gram",
    })

    start = time.perf_counter()
    fasttext = FastText(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=1,
        sg=1,
        negative=10,
        epochs=10,
        min_n=3,
        max_n=5,
        bucket=20000,
        seed=SEED,
    )
    ft_time = time.perf_counter() - start
    fasttext.save(str(MODELS / "week5_fasttext.model"))
    fasttext.wv.save(str(MODELS / "week5_fasttext.kv"))
    results.append({
        "model": "fastText",
        "vector_size": VECTOR_SIZE,
        "vocabulary": len(fasttext.wv),
        "training_seconds": round(ft_time, 3),
        "notes": "Gensim FastText with character n-grams",
    })

    start = time.perf_counter()
    vocab, counts = build_vocab(sentences, min_count=MIN_COUNT)
    rows, cols, values = build_cooccurrence(sentences, vocab, window_size=WINDOW)
    glove_vectors, glove_bias = train_glove(
        rows, cols, values,
        vocab_size=len(vocab),
        vector_size=VECTOR_SIZE,
        epochs=10,
        learning_rate=0.05,
        seed=SEED,
    )
    glove_time = time.perf_counter() - start
    np.savez_compressed(
        MODELS / "week5_glove.npz",
        vectors=glove_vectors.astype(np.float32),
        bias=glove_bias.astype(np.float32),
    )
    with open(MODELS / "week5_glove_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    results.append({
        "model": "GloVe",
        "vector_size": VECTOR_SIZE,
        "vocabulary": len(vocab),
        "training_seconds": round(glove_time, 3),
        "notes": "Own compact GloVe implementation, weighted co-occurrence matrix",
    })

    return w2v, fasttext, glove_vectors, vocab, counts, pd.DataFrame(results)


def cosine_from_vectors(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def nearest(model_name, model, word, topn=10, normalized_glove=None, glove_inverse=None):
    if model_name == "GloVe":
        vectors, vocab = model
        if word not in vocab:
            return []
        matrix = normalized_glove if normalized_glove is not None else normalize_rows(vectors)
        idx = vocab[word]
        scores = matrix @ matrix[idx]
        scores[idx] = -np.inf
        ids = np.argsort(-scores)[:topn]
        inv = glove_inverse if glove_inverse is not None else {i: w for w, i in vocab.items()}
        return [(inv[int(i)], float(scores[int(i)])) for i in ids]
    try:
        return [(w, float(score)) for w, score in model.wv.most_similar(word, topn=topn)]
    except KeyError:
        return []


def vector_for(model_name, model, word):
    if model_name == "GloVe":
        vectors, vocab = model
        return vectors[vocab[word]] if word in vocab else None
    try:
        return model.wv[word]
    except KeyError:
        return None


def semantic_examples(models):
    anchors = ["free", "money", "call", "phone", "mobile", "prize", "offer", "win", "good", "love"]
    glove_model = models["GloVe"]
    glove_norm = normalize_rows(glove_model[0])
    glove_inverse = {i: w for w, i in glove_model[1].items()}
    rows = []
    for model_name, model in models.items():
        for anchor in anchors:
            for rank, (word, score) in enumerate(
                nearest(model_name, model, anchor, topn=8, normalized_glove=glove_norm, glove_inverse=glove_inverse),
                start=1,
            ):
                rows.append({
                    "model": model_name,
                    "anchor": anchor,
                    "rank": rank,
                    "neighbor": word,
                    "cosine": round(score, 6),
                })
    return pd.DataFrame(rows)


def pair_quality(models):
    # Domain-specific related pairs. This is an intrinsic proxy, not an absolute benchmark.
    pairs = [
        ("free", "offer"),
        ("free", "prize"),
        ("money", "cash"),
        ("call", "phone"),
        ("phone", "mobile"),
        ("win", "winner"),
        ("good", "great"),
        ("offer", "deal"),
        ("customer", "service"),
        ("text", "message"),
        ("home", "house"),
        ("love", "like"),
    ]
    rows = []
    for a, b in pairs:
        row = {"word_a": a, "word_b": b}
        for model_name, model in models.items():
            va = vector_for(model_name, model, a)
            vb = vector_for(model_name, model, b)
            row[f"{model_name}_cosine"] = round(cosine_from_vectors(va, vb), 6) if va is not None and vb is not None else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    long = df.melt(id_vars=["word_a", "word_b"], var_name="model", value_name="cosine")
    long["model"] = long["model"].str.replace("_cosine", "", regex=False)
    return df, long


def arithmetic_examples(models):
    triplets = [
        ("great", "good", "bad", "great - good + bad"),
        ("phone", "mobile", "call", "phone - mobile + call"),
        ("money", "cash", "prize", "money - cash + prize"),
        ("offer", "free", "price", "offer - free + price"),
    ]
    rows = []
    for model_name, model in models.items():
        for a, b, c, expression in triplets:
            va, vb, vc = [vector_for(model_name, model, w) for w in (a, b, c)]
            if va is None or vb is None or vc is None:
                rows.append({"model": model_name, "expression": expression, "result": "missing word", "cosine": np.nan})
                continue
            target = va - vb + vc
            if model_name == "GloVe":
                vectors, vocab = model
                norm = normalize_rows(vectors)
                target = target / max(np.linalg.norm(target), 1e-12)
                scores = norm @ target
                for w in (a, b, c):
                    if w in vocab:
                        scores[vocab[w]] = -np.inf
                idx = int(np.argmax(scores))
                inv = {i: w for w, i in vocab.items()}
                result, score = inv[idx], float(scores[idx])
            else:
                # Gensim accepts positive and negative terms directly.
                try:
                    found = model.wv.most_similar(positive=[a, c], negative=[b], topn=1)
                    result, score = found[0]
                except KeyError:
                    result, score = "missing word", np.nan
            rows.append({
                "model": model_name,
                "expression": expression,
                "result": result,
                "cosine": round(float(score), 6) if pd.notna(score) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_pair_quality(long_df):
    summary = long_df.dropna().groupby("model")["cosine"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(summary.index, summary.values)
    ax.set_xlabel("Средняя cosine similarity")
    ax.set_title("Week 5 — сравнение семантического качества векторов")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMAGES / "week5_model_quality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_neighbors(neighbor_df):
    if neighbor_df.empty:
        return
    selected = neighbor_df[(neighbor_df["anchor"] == "free") & (neighbor_df["rank"] <= 5)].copy()
    if selected.empty:
        return
    selected["label"] = selected["model"] + ": " + selected["neighbor"]
    selected = selected.sort_values("cosine")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(selected["label"], selected["cosine"])
    ax.set_xlabel("Cosine similarity")
    ax.set_title('Week 5 — ближайшие слова для "free"')
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMAGES / "week5_free_neighbors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_sms_data()
    sentences = build_sentences(df["message"], preprocess)
    sentences = [s for s in sentences if s]

    vocab_counts = pd.Series([w for s in sentences for w in s]).value_counts()
    pd.DataFrame({
        "metric": ["messages", "non_empty_sentences", "tokens", "unique_tokens", "unique_tokens_min_count_2"],
        "value": [len(df), len(sentences), int(vocab_counts.sum()), len(vocab_counts), int((vocab_counts >= MIN_COUNT).sum())],
    }).to_csv(REPORTS / "week5_vocab_stats.csv", index=False)

    w2v, fasttext, glove_vectors, glove_vocab, counts, comparison = train_models(sentences)
    comparison.to_csv(REPORTS / "week5_model_comparison.csv", index=False)

    models = {
        "Word2Vec": w2v,
        "fastText": fasttext,
        "GloVe": (glove_vectors, glove_vocab),
    }

    neighbors = semantic_examples(models)
    neighbors.to_csv(REPORTS / "week5_semantic_neighbors.csv", index=False)

    pair_wide, pair_long = pair_quality(models)
    pair_wide.to_csv(REPORTS / "week5_similarity_pairs.csv", index=False)
    pair_long.to_csv(REPORTS / "week5_similarity_pairs_long.csv", index=False)

    arithmetic = arithmetic_examples(models)
    arithmetic.to_csv(REPORTS / "week5_vector_arithmetic.csv", index=False)

    plot_pair_quality(pair_long)
    plot_neighbors(neighbors)

    print("Week 5 завершена.")
    print(comparison.to_string(index=False))
    print("\nСемантические примеры для free:")
    print(neighbors[neighbors.anchor == "free"].head(24).to_string(index=False))
    print("\nАрифметика векторов:")
    print(arithmetic.to_string(index=False))
    print(f"\nМодели: {MODELS}")
    print(f"CSV-отчёты: {REPORTS}")
    print(f"Изображения: {IMAGES}")


if __name__ == "__main__":
    main()
