"""Week 9: fine-tune BERT with the Week 8 linear head on the same SMS split."""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data import load_sms_data
from scripts.week7_transformer_attention import MODEL, REVISION
from scripts.week8_bert_features import ROOT, SEED, MAX_LENGTH, metrics, split_indices

EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 2e-5


class SmsDataset(Dataset):
    def __init__(self, messages, labels, tokenizer):
        self.encoding = tokenizer(messages, padding="max_length", truncation=True,
                                  max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{k: v[index] for k, v in self.encoding.items()},
                "labels": self.labels[index]}


class BertSmsClassifier(nn.Module):
    def __init__(self, pretrained_head):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL, revision=REVISION)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(pretrained_head.coef_, dtype=torch.float32))
            self.head.bias.copy_(torch.tensor(pretrained_head.intercept_, dtype=torch.float32))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return self.head(pooled).squeeze(-1)


def predict(model, loader):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            probabilities.append(torch.sigmoid(model(**batch)).numpy())
            batch["labels"] = labels
    return np.concatenate(probabilities)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    df = load_sms_data().reset_index(drop=True)
    labels = (df.label == "spam").astype(int).to_numpy()
    train_idx, test_idx = split_indices(labels)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    head_file = ROOT / "models" / "week8_frozen_bert_logreg.joblib"
    if not head_file.exists():
        from scripts.week8_bert_features import main as run_week8
        run_week8()
    head = joblib.load(head_file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    train_data = SmsDataset(df.message.iloc[train_idx].tolist(), labels[train_idx], tokenizer)
    test_data = SmsDataset(df.message.iloc[test_idx].tolist(), labels[test_idx], tokenizer)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)
    classifier = BertSmsClassifier(head)

    before_prob = predict(classifier, test_loader)
    # Both use identical BERT embeddings, pooling and the fitted linear head.
    # Confirm that the before state agrees with the Week 8 recorded prediction.
    week8 = pd.read_csv(reports / "week8_predictions.csv") if (reports / "week8_predictions.csv").exists() else None
    if week8 is not None:
        if not np.array_equal(week8.source_row.to_numpy(), test_idx):
            raise ValueError("Week 8 and Week 9 splits differ")
        if not np.allclose(before_prob, week8.bert_probability_spam, atol=3e-5):
            raise ValueError("Frozen model predictions differ from Week 8")

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)
    counts = np.bincount(labels[train_idx], minlength=2)
    weights = torch.tensor(len(train_idx) / (2 * counts), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(weights[1] / weights[0]))
    history = []
    for epoch in range(EPOCHS):
        classifier.train()
        losses = []
        for batch in train_loader:
            y = batch.pop("labels").float()
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(**batch)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        history.append(row)
        print(row, flush=True)
    after_prob = predict(classifier, test_loader)
    before_pred = (before_prob >= 0.5).astype(int)
    after_pred = (after_prob >= 0.5).astype(int)
    comparison = pd.DataFrame([
        metrics("Before: frozen BERT + fitted linear head", labels[test_idx], before_pred,
                before_prob, len(train_idx)),
        metrics("After: end-to-end fine-tuned BERT + linear head", labels[test_idx], after_pred,
                after_prob, len(train_idx)),
    ])
    comparison.to_csv(reports / "week9_before_after.csv", index=False)
    pd.DataFrame({"source_row": test_idx, "true_label": df.label.iloc[test_idx].to_numpy(),
                  "before_pred": np.where(before_pred, "spam", "ham"),
                  "before_probability_spam": before_prob,
                  "after_pred": np.where(after_pred, "spam", "ham"),
                  "after_probability_spam": after_prob}).to_csv(reports / "week9_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(reports / "week9_training_history.csv", index=False)
    (reports / "week9_config.json").write_text(json.dumps({
        "encoder": MODEL, "revision": REVISION, "initialized_head": "week8_frozen_bert_logreg.joblib",
        "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE, "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss with train-derived positive class weight",
        "seed": SEED, "test_size": 0.2, "stratified": True,
        "selection": "fixed hyperparameters; test evaluated before and after, never used for training",
    }, indent=2), encoding="utf-8")
    checkpoint = ROOT / "models" / "week9_finetuned_bert.pt"
    torch.save(classifier.state_dict(), checkpoint)
    print(comparison.to_string(index=False))
    print(f"Model checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
