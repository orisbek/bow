"""Recreate and persist the best assignment Week 4 configuration for the UI.

Run once: python -m scripts.prepare_app_model
Does not change Week 10 reports or fall back to another model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from scripts.week10_experiments import frozen_features, run_training, scored
from scripts.week7_transformer_attention import MODEL, REVISION
from scripts.week8_bert_features import MAX_LENGTH, SEED, split_indices
from scripts.week9_bert_finetune import SmsDataset
from src.app_inference import MODEL_DIR
from src.data import load_sms_data

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_best(results):
    # Stable tie-breakers: F1 spam, accuracy, then experiment name.
    return results.sort_values(["f1_spam", "accuracy", "experiment"],
                               ascending=[False, False, True]).iloc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL,
                        help="Pinned Hub model or a local copy of the same revision")
    parser.add_argument("--force", action="store_true", help="Rebuild the application model")
    args = parser.parse_args()
    if (MODEL_DIR / "metadata.json").exists() and not args.force:
        print("Application model already prepared. Use --force to rebuild.", flush=True)
        return

    results_path = ROOT / "reports" / "week10_experiments.csv"
    best = select_best(pd.read_csv(results_path))
    if best.experiment != "lr_5e-5_full" or best.kind != "fresh_training":
        raise ValueError("Week 4 winner changed; review training setup before rebuilding the app.")
    print(f"Selected: {best.experiment}; recorded F1={best.f1_spam:.6f}", flush=True)
    source_config = json.loads((ROOT / "reports" / "week10_config.json").read_text(encoding="utf-8"))
    # Git on Windows may check out LF text files as CRLF. Compare canonical LF
    # bytes so the same corpus does not fail the recorded source integrity check.
    dataset_bytes = (ROOT / "data" / "SMSSpamCollection").read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes.replace(b"\r\n", b"\n")).hexdigest()
    if dataset_hash != source_config["dataset_sha256"]:
        raise ValueError("Dataset differs from the Week 4 experiment.")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = load_sms_data().reset_index(drop=True)
    labels = (df.label == "spam").astype(int).to_numpy()
    train, test = split_indices(labels)
    split = pd.read_csv(ROOT / "reports" / "week10_split.csv")
    if not np.array_equal(np.sort(test), split.loc[split.split == "test", "source_row"]):
        raise ValueError("Train/test split differs from Week 4.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=REVISION)
    fingerprint = hashlib.sha256((dataset_hash + MODEL + REVISION + str(MAX_LENGTH)).encode()).hexdigest()
    features = frozen_features(df.message.tolist(), tokenizer, args.model_path, fingerprint)
    test_loader = DataLoader(SmsDataset(df.message.iloc[test].tolist(), labels[test], tokenizer), batch_size=64)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / "classifier.pt"
    probabilities, history = run_training(
        str(best.experiment), float(best.learning_rate), train, labels, df, features,
        tokenizer, args.model_path, test_loader, checkpoint_path=checkpoint,
    )
    tokenizer.save_pretrained(MODEL_DIR)
    AutoConfig.from_pretrained(args.model_path, revision=REVISION).save_pretrained(MODEL_DIR)
    clean_mask = ~df.message.iloc[test].isin(set(df.message.iloc[train])).to_numpy()
    actual = scored(str(best.experiment), labels[test], probabilities, len(train), clean_mask)
    old = pd.read_csv(ROOT / "reports" / "week10_predictions.csv")
    old = old[old.experiment == best.experiment].set_index("source_row").loc[test]
    metadata = {
        "assignment_week": 5, "repository_stage": 11,
        "experiment": str(best.experiment), "model": MODEL, "revision": REVISION,
        "learning_rate": float(best.learning_rate), "epochs": int(best.epochs),
        "batch_size": 32, "max_length": MAX_LENGTH, "seed": SEED,
        "threshold": .5, "train_size": len(train), "test_size": len(test),
        "recorded_metrics": {key: float(best[key]) for key in
                             ["accuracy", "precision_spam", "recall_spam", "f1_spam"]},
        "reproduced_metrics": actual,
        "matching_test_predictions": int(np.sum((probabilities >= .5) == (old.predicted_label == "spam").to_numpy())),
        "max_probability_difference": float(np.max(np.abs(probabilities - old.probability_spam.to_numpy()))),
        "dataset_sha256": dataset_hash,
        "dataset_file_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_hash_normalization": "CRLF to LF for cross-platform Git checkouts",
        "selection_csv_sha256": sha256(results_path),
        "checkpoint_sha256": sha256(checkpoint), "history": history,
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "numpy", "pandas", "scikit-learn", "streamlit"]},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "reports" / "week11_model_reproduction.json"
    report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"source_row": test, "true_label": df.label.iloc[test].to_numpy(),
                  "predicted_label": np.where(probabilities >= .5, "spam", "ham"),
                  "probability_spam": probabilities}).to_csv(ROOT / "reports" / "week11_test_predictions.csv", index=False)
    print(json.dumps(actual, indent=2), flush=True)
    print(f"Model ready for offline inference: {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
