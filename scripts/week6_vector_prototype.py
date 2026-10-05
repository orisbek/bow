from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data import load_sms_data
from src.preprocessing import preprocess
from src.vector_classifier import build_document_matrix, load_week5_models

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
IMAGES = ROOT / "outputs" / "images"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
for path in (REPORTS, IMAGES, MODELS, ARTIFACTS):
    path.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
MAX_ITER = 1000


def plot_confusion(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["ham", "spam"],
        columns=["ham", "spam"],
    ).to_csv(REPORTS / "week6_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Week 6 — confusion matrix ({model_name})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(IMAGES / "week6_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics_df: pd.DataFrame):
    plot_df = metrics_df.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["model"], plot_df["f1"])
    ax.set_xlabel("F1-score")
    ax.set_title("Week 6 — сравнение классификаторов на document embeddings")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMAGES / "week6_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pca(matrix: np.ndarray, labels: np.ndarray, model_name: str):
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(matrix)
    explained = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "label": np.where(labels == 1, "spam", "ham"),
    })
    pca_df.to_csv(REPORTS / "week6_pca.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for label_name in ["ham", "spam"]:
        part = pca_df[pca_df["label"] == label_name]
        ax.scatter(part["pc1"], part["pc2"], s=16, alpha=0.35, label=label_name)
    ax.set_title(f"Week 6 — PCA document embeddings ({model_name})")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(IMAGES / "week6_pca_embeddings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": explained,
    }).to_csv(REPORTS / "week6_pca_explained_variance.csv", index=False)


def train_and_evaluate(model_name, matrix, labels, ids, train_idx, test_idx):
    # Logistic Regression is the small applied prototype built on vector features.
    clf = LogisticRegression(max_iter=MAX_ITER, random_state=SEED, class_weight="balanced")
    clf.fit(matrix[train_idx], labels[train_idx])

    pred = clf.predict(matrix[test_idx])
    prob = clf.predict_proba(matrix[test_idx])[:, 1]

    metrics = {
        "model": model_name,
        "classifier": "LogisticRegression",
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "known_vector_test_examples": int(np.linalg.norm(matrix[test_idx], axis=1).astype(bool).sum()),
        "accuracy": accuracy_score(labels[test_idx], pred),
        "precision": precision_score(labels[test_idx], pred, zero_division=0),
        "recall": recall_score(labels[test_idx], pred, zero_division=0),
        "f1": f1_score(labels[test_idx], pred, zero_division=0),
        "roc_auc": roc_auc_score(labels[test_idx], prob),
    }

    predictions = pd.DataFrame({
        "id": np.asarray(ids)[test_idx],
        "label_true": np.where(labels[test_idx] == 1, "spam", "ham"),
        "label_pred": np.where(pred == 1, "spam", "ham"),
        "probability_spam": prob,
        "model": model_name,
    })
    return metrics, predictions, clf


def build_document_embeddings_report(df, token_lists, matrices):
    rows = pd.DataFrame({
        "id": np.arange(len(df)),
        "label": df["label"].values,
        "message": df["message"].values,
        "tokens": [" ".join(tokens) for tokens in token_lists],
    })
    for model_name, matrix in matrices.items():
        safe = model_name.lower().replace("2", "2").replace("fasttext", "fasttext")
        for i in range(matrix.shape[1]):
            rows[f"{safe}_dim_{i+1}"] = matrix[:, i]
    rows.to_csv(REPORTS / "week6_document_embeddings.csv", index=False)


def predict_text(text: str, best_model_name: str, classifier, models):
    tokens = preprocess(text)
    model = models[best_model_name]
    if best_model_name == "GloVe":
        glove_vectors, glove_vocab = model
        vector = build_document_matrix([tokens], best_model_name, None, glove_vectors, glove_vocab)[0]
    else:
        vector = build_document_matrix([tokens], best_model_name, model)[0]
    label = int(classifier.predict(vector)[0])
    prob = float(classifier.predict_proba(vector)[0, 1])
    return ("spam" if label == 1 else "ham"), prob


def main():
    print("[Week 6] Загрузка SMSSpamCollection...")
    df = load_sms_data()
    token_lists = [preprocess(text) for text in df["message"]]
    labels = df["label"].map({"ham": 0, "spam": 1}).to_numpy(dtype=np.int64)
    ids = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        ids,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )
    pd.DataFrame({
        "id": ids,
        "split": np.where(np.isin(ids, test_idx), "test", "train"),
        "label": df["label"].values,
    }).to_csv(REPORTS / "week6_split.csv", index=False)

    print("[Week 6] Загрузка моделей Week 5...")
    models = load_week5_models(MODELS)

    matrices = {}
    for model_name, model in models.items():
        print(f"[Week 6] Построение document embeddings: {model_name}...")
        if model_name == "GloVe":
            glove_vectors, glove_vocab = model
            matrix, mask = build_document_matrix(token_lists, model_name, None, glove_vectors, glove_vocab)
        else:
            matrix, mask = build_document_matrix(token_lists, model_name, model)
        matrices[model_name] = matrix
        pd.DataFrame({
            "model": [model_name],
            "documents": [len(matrix)],
            "dimensions": [matrix.shape[1]],
            "documents_with_nonzero_vectors": [int(mask.sum())],
            "zero_vector_documents": [int((~mask).sum())],
        }).to_csv(REPORTS / f"week6_{model_name.lower()}_vector_stats.csv", index=False)

    build_document_embeddings_report(df, token_lists, matrices)

    metric_rows = []
    prediction_frames = []
    fitted = {}
    for model_name, matrix in matrices.items():
        print(f"[Week 6] Обучение Logistic Regression: {model_name}...")
        metrics, predictions, clf = train_and_evaluate(
            model_name, matrix, labels, ids, train_idx, test_idx
        )
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        fitted[model_name] = clf
        joblib.dump(clf, MODELS / f"week6_{model_name.lower()}_logreg.joblib")

    metrics_df = pd.DataFrame(metric_rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(REPORTS / "week6_model_comparison.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(REPORTS / "week6_predictions.csv", index=False)

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_clf = fitted[best_model_name]
    with open(ARTIFACTS / "week6_best_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": best_model_name,
            "classifier": "LogisticRegression",
            "random_state": SEED,
            "test_size": TEST_SIZE,
            "note": "Document vectors are mean pooled from Week 5 word embeddings trained on the same SMSSpamCollection corpus.",
        }, f, ensure_ascii=False, indent=2)

    best_preds = pd.concat(prediction_frames, ignore_index=True)
    best_preds = best_preds[best_preds["model"] == best_model_name]
    plot_confusion(
        best_preds["label_true"].map({"ham": 0, "spam": 1}).to_numpy(),
        best_preds["label_pred"].map({"ham": 0, "spam": 1}).to_numpy(),
        best_model_name,
    )
    plot_model_comparison(metrics_df)

    # Visualize the same held-out sample space with PCA for the best embedding model.
    plot_pca(matrices[best_model_name][test_idx], labels[test_idx], best_model_name)

    examples = [
        "Congratulations! You have won a free prize. Call now to claim.",
        "Are we still meeting at 6 pm today?",
        "Free entry to win cash, reply now.",
        "Thanks for your message, see you tomorrow.",
    ]
    demo_rows = []
    for text in examples:
        label, prob = predict_text(text, best_model_name, best_clf, models)
        demo_rows.append({
            "message": text,
            "prediction": label,
            "probability_spam": round(prob, 6),
            "embedding_model": best_model_name,
        })
    pd.DataFrame(demo_rows).to_csv(REPORTS / "week6_demo_predictions.csv", index=False)

    print("\nWeek 6 завершена.")
    print("Лучший вариант:")
    print(metrics_df.iloc[0].to_string())
    print("\nСравнение:")
    print(metrics_df.to_string(index=False))
    print(f"\nЛучшее embedding-модель: {best_model_name}")
    print(f"CSV: {REPORTS}")
    print(f"Изображения: {IMAGES}")
    print(f"Классификаторы: {MODELS}")


if __name__ == "__main__":
    main()
