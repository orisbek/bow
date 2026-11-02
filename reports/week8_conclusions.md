# Неделя 8 — замороженный BERT как извлекатель признаков

Источник: `data/SMSSpamCollection`, 5 572 SMS. Разбиение 4 457 train / 1 115 test, стратифицированное, `random_state=42`; список строк хранится в `week8_split.csv`. Модель `prajjwal1/bert-tiny` закреплённой ревизии; ограничение 96 WordPiece токенов, mean pooling с mask по последнему слою. Веса BERT не изменяются. Линейная голова Logistic Regression обучена на train, что делает подход *feature extraction*, а не zero-shot. Для сравнения TF-IDF + Logistic Regression обучена на той же train-части с таким же `class_weight="balanced"`; словарь TF-IDF не видит test.

| Модель | Accuracy | Precision spam | Recall spam | F1 spam | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.983857 | 0.951724 | 0.926174 | 0.938776 | 0.990169 |
| Frozen BERT + Logistic Regression | 0.978475 | 0.883436 | 0.966443 | 0.923077 | 0.995442 |

Числа получены запуском `python -m scripts.week8_bert_features` и хранятся с полной точностью в `week8_comparison.csv`; индивидуальные ответы — `week8_predictions.csv`. На этой тестовой выборке TF-IDF выше по F1, а замороженный BERT имеет более высокий recall и ROC-AUC. Последнее не заменяет F1: при пороге 0.5 ложных тревог у BERT больше. Это один split и маленький BERT; преимущества полного BERT или других корпусов из этих чисел не следуют.

Код: `scripts/week8_bert_features.py`, ноутбук: `notebooks/08_bert_features.ipynb`. После запуска сохраняются также `models/week8_frozen_bert_logreg.joblib`, `models/week8_tfidf_baseline.joblib` и кэш признаков `artifacts/week8_frozen_features.npz` (последние генерируются повторно и игнорируются Git).
