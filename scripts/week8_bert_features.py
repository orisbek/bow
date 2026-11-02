"""Week 8: frozen pretrained BERT features vs the project's TF-IDF baseline."""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from src.data import load_sms_data
from scripts.week7_transformer_attention import MODEL, REVISION

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MAX_LENGTH = 96
BATCH_SIZE = 64


def split_indices(labels):
    ids = np.arange(len(labels))
    return train_test_split(ids, test_size=0.2, random_state=SEED, stratify=labels)


def metrics(name, y, pred, prob, train_size):
    return {"model": name, "train_size": train_size, "test_size": len(y),
            "accuracy": accuracy_score(y, pred),
            "precision_spam": precision_score(y, pred, zero_division=0),
            "recall_spam": recall_score(y, pred, zero_division=0),
            "f1_spam": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob)}


def get_embeddings(messages, cache_path: Path):
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["features"]) == len(messages):
            return cached["features"]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    encoder = AutoModel.from_pretrained(MODEL, revision=REVISION).eval()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rows = []
    with torch.inference_mode():
        for start in range(0, len(messages), BATCH_SIZE):
            batch = tok(messages[start:start+BATCH_SIZE], padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
            states = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            rows.append(((states * mask).sum(1) / mask.sum(1)).numpy())
            if start % (BATCH_SIZE * 20) == 0:
                print(f"BERT: {min(start+BATCH_SIZE, len(messages))}/{len(messages)}", flush=True)
    features = np.vstack(rows)
    cache_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    return features


def main():
    torch.manual_seed(SEED)
    df = load_sms_data().reset_index(drop=True)
    labels = (df.label == "spam").astype(int).to_numpy()
    train, test = split_indices(labels)
    reports, models = ROOT / "reports", ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    messages = df.message.tolist()

    # TF-IDF vocabulary and IDF are fit on train only, never on held-out texts.
    tfidf = TfidfVectorizer()
    x_train = tfidf.fit_transform([messages[i] for i in train])
    x_test = tfidf.transform([messages[i] for i in test])
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    baseline.fit(x_train, labels[train])
    base_prob = baseline.predict_proba(x_test)[:, 1]
    base_pred = baseline.predict(x_test)

    # The BERT encoder is never updated; only a small linear head learns labels.
    features = get_embeddings(messages, ROOT / "artifacts" / "week8_frozen_features.npz")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(features[train], labels[train])
    bert_prob = clf.predict_proba(features[test])[:, 1]
    bert_pred = clf.predict(features[test])

    summary = pd.DataFrame([
        metrics("TF-IDF + LogisticRegression", labels[test], base_pred, base_prob, len(train)),
        metrics("Frozen BERT mean pooling + LogisticRegression", labels[test], bert_pred, bert_prob, len(train)),
    ])
    summary.to_csv(reports / "week8_comparison.csv", index=False)
    pd.DataFrame({"source_row": np.arange(len(df)), "label": df.label,
                  "split": np.where(np.isin(np.arange(len(df)), test), "test", "train")}).to_csv(
                      reports / "week8_split.csv", index=False)
    pd.DataFrame({"source_row": test, "true_label": df.label.iloc[test].to_numpy(),
                  "baseline_pred": np.where(base_pred, "spam", "ham"),
                  "baseline_probability_spam": base_prob,
                  "bert_pred": np.where(bert_pred, "spam", "ham"),
                  "bert_probability_spam": bert_prob}).to_csv(reports / "week8_predictions.csv", index=False)
    joblib.dump(clf, models / "week8_frozen_bert_logreg.joblib")
    joblib.dump((tfidf, baseline), models / "week8_tfidf_baseline.joblib")
    (reports / "week8_config.json").write_text(json.dumps({
        "encoder": MODEL, "revision": REVISION, "max_length": MAX_LENGTH,
        "pooling": "attention-mask mean of final hidden states, including CLS/SEP",
        "encoder_updated": False, "trainable_head": "balanced LogisticRegression",
        "split": {"test_size": 0.2, "stratified": True, "seed": SEED},
        "positive_class": "spam", "threshold": 0.5,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
