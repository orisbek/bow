"""Offline inference for the configuration selected from assignment Week 4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "week11_best_bert"
MAX_CHARACTERS = 4000


def validate_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Введите текст SMS перед анализом.")
    if len(text) > MAX_CHARACTERS:
        raise ValueError(f"Максимальная длина — {MAX_CHARACTERS} символов.")
    if not any("a" <= char.lower() <= "z" for char in text):
        raise ValueError("Введите SMS на английском языке: модель обучена на английском корпусе.")
    return text


class AppClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        return self.head((states * mask).sum(1) / mask.sum(1)).squeeze(-1)


class SpamDetector:
    def __init__(self, directory: Path = MODEL_DIR):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        checkpoint = directory / "classifier.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.metadata["checkpoint_sha256"]:
            raise ValueError("Контрольная сумма модели не совпала. Повторите подготовку модели.")
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AppClassifier(config)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()
        self.lock = RLock()

    def analyze(self, text: str) -> dict:
        text = validate_text(text)
        max_length = int(self.metadata["max_length"])
        with self.lock, torch.inference_mode():
            token_count = len(self.tokenizer(text, truncation=False)["input_ids"])
            batch = self.tokenizer(text, padding="max_length", truncation=True,
                                   max_length=max_length, return_tensors="pt")
            probability_spam = float(torch.sigmoid(self.model(**batch)).item())
        label = "spam" if probability_spam >= .5 else "ham"
        return {
            "text": text,
            "label": label,
            "confidence": probability_spam if label == "spam" else 1 - probability_spam,
            "probability_spam": probability_spam,
            "probability_ham": 1 - probability_spam,
            "token_count": token_count,
            "truncated": token_count > max_length,
            "max_length": max_length,
            "experiment": self.metadata["experiment"],
        }
