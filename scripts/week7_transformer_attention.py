"""Week 7: contextual BERT embeddings and a reproducible attention example."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from src.data import load_sms_data

ROOT = Path(__file__).resolve().parents[1]
MODEL = "prajjwal1/bert-tiny"  # English SMS; 2-layer BERT, 128 hidden dimensions.
REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
SEED = 42


def select_messages(df: pd.DataFrame) -> pd.DataFrame:
    # Guarantee a real contrast for the same surface word, then sample the rest.
    fixed = [2, 75]  # "Free entry ..." (spam), "once you free" (ham).
    assert df.loc[fixed, "label"].tolist() == ["spam", "ham"]
    rest = df.drop(index=fixed)
    chosen = pd.concat([
        df.loc[fixed],
        rest[rest.label == "ham"].sample(n=9, random_state=SEED),
        rest[rest.label == "spam"].sample(n=9, random_state=SEED),
    ]).reset_index(names="source_row")
    return chosen


def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    df = select_messages(load_sms_data())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, attn_implementation="eager").eval()
    encoded = tokenizer(df.message.tolist(), padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
    with torch.inference_mode():
        out = model(**encoded, output_attentions=True)
    mask = encoded["attention_mask"].unsqueeze(-1)
    pooled = ((out.last_hidden_state * mask).sum(1) / mask.sum(1)).numpy()
    tfidf = TfidfVectorizer().fit_transform(df.message)
    tfidf_sim = cosine_similarity(tfidf)
    bert_sim = cosine_similarity(pooled)
    df.to_csv(reports / "week7_sample.csv", index=False)
    pd.DataFrame(pooled, columns=[f"bert_dim_{i}" for i in range(pooled.shape[1])]).assign(
        source_row=df.source_row.to_numpy(), label=df.label.to_numpy()
    ).to_csv(reports / "week7_embeddings.csv", index=False)

    pairs = [(0, 1)] + [(i, j) for i in range(len(df)) for j in range(i + 1, len(df)) if (i, j) != (0, 1)]
    pd.DataFrame([{
        "source_row_a": int(df.source_row[i]), "source_row_b": int(df.source_row[j]),
        "label_a": df.label[i], "label_b": df.label[j],
        "tfidf_cosine": float(tfidf_sim[i, j]), "bert_cosine": float(bert_sim[i, j]),
    } for i, j in pairs]).to_csv(reports / "week7_similarity.csv", index=False)

    # Compare the token 'free' in two actual SMS, averaged over its WordPieces.
    token_vectors = []
    attention_rows = []
    for i in (0, 1):
        valid_len = int(encoded.attention_mask[i].sum())
        tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids[i, :valid_len])
        free_positions = [j for j, token in enumerate(tokens) if token.lower() == "free"]
        if not free_positions:
            raise ValueError(f"No 'free' token in source row {df.source_row[i]}")
        pos = free_positions[0]
        token_vectors.append(out.last_hidden_state[i, pos].numpy())
        # Last layer, average over heads; weights describe this query token's attention.
        weights = out.attentions[-1][i, :, pos, :valid_len].mean(0).numpy()
        for j, (token, weight) in enumerate(zip(tokens, weights)):
            attention_rows.append({"source_row": int(df.source_row[i]), "label": df.label[i],
                                   "query_token": "free", "key_position": j,
                                   "key_token": token, "attention_weight": float(weight)})
    pd.DataFrame(attention_rows).to_csv(reports / "week7_attention_free.csv", index=False)
    context_cosine = float(cosine_similarity([token_vectors[0]], [token_vectors[1]])[0, 0])
    summary = {"model": MODEL, "revision": REVISION, "sample_size": len(df), "ham": int((df.label == "ham").sum()),
               "spam": int((df.label == "spam").sum()),
               "focus_rows": [2, 75], "tfidf_document_cosine": float(tfidf_sim[0, 1]),
               "bert_document_cosine": float(bert_sim[0, 1]),
               "bert_free_token_cosine": context_cosine,
               "attention_note": "Final-layer attention averaged across heads; attention is not a causal explanation."}
    (reports / "week7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
