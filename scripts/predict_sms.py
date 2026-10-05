from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess
from src.vector_classifier import build_document_matrix, load_week5_models

MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"


def main():
    if len(sys.argv) < 2:
        print('Использование: python scripts/predict_sms.py "Your SMS text here"')
        raise SystemExit(1)

    text = " ".join(sys.argv[1:])
    with open(ARTIFACTS / "week6_best_model.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    model_name = config["model"]
    classifier = joblib.load(MODELS / f"week6_{model_name.lower()}_logreg.joblib")
    models = load_week5_models(MODELS)

    tokens = preprocess(text)
    model = models[model_name]
    if model_name == "GloVe":
        vectors, vocab = model
        matrix, _ = build_document_matrix([tokens], model_name, None, vectors, vocab)
    else:
        matrix, _ = build_document_matrix([tokens], model_name, model)

    prediction = int(classifier.predict(matrix)[0])
    probability = float(classifier.predict_proba(matrix)[0, 1])
    label = "spam" if prediction == 1 else "ham"

    print(f"Embedding model: {model_name}")
    print(f"Prediction: {label}")
    print(f"Probability spam: {probability:.4f}")


if __name__ == "__main__":
    main()
