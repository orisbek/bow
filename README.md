# Spam Detector — учебный NLP-проект (недели 1–10)

Проект работает **только с исходным датасетом `data/SMSSpamCollection`** и постепенно проходит путь от EDA и preprocessing до векторных представлений и прикладной классификации SMS.

## Неделя 6 — мини-задача на основе векторов

Прикладная задача выбрана из темы проекта: **классификация SMS на `spam` и `ham`**.

Для каждого сообщения строится document embedding как среднее (mean pooling) векторов его слов из моделей Week 5:

- Word2Vec;
- GloVe;
- fastText.

Затем для каждого пространства отдельно обучается Logistic Regression. Используется один и тот же stratified train/test split (`test_size=0.2`, `random_state=42`), поэтому сравнение моделей честно по одним и тем же примерам.

Для визуализации document embeddings используется PCA до двух компонент. PCA не меняет обучаемую модель, а только проецирует 30-мерные документы на 2D для просмотра структуры `spam` и `ham`.

### Важное замечание о данных

Word2Vec/GloVe/fastText в Week 5 обучены без использования меток на том же корпусе `SMSSpamCollection`. В Week 6 эти готовые embedding-модели преобразуют сообщения в векторы. Это учебный прототип на одном корпусе, а не строгий production benchmark с отдельным внешним корпусом предварительного обучения.

## Что создаётся после запуска

### CSV

- `reports/week6_split.csv` — фиксированное train/test-разбиение;
- `reports/week6_document_embeddings.csv` — document embeddings;
- `reports/week6_model_comparison.csv` — Accuracy, Precision, Recall, F1 и ROC-AUC для Word2Vec/GloVe/fastText;
- `reports/week6_predictions.csv` — предсказания на тестовой части;
- `reports/week6_confusion_matrix.csv` — числовая confusion matrix лучшей модели;
- `reports/week6_demo_predictions.csv` — несколько демонстрационных SMS;
- `reports/week6_pca.csv` — координаты документов после PCA;
- `reports/week6_pca_explained_variance.csv` — доля дисперсии двух компонент;
- `reports/week6_*_vector_stats.csv` — статистика document vectors.

### Изображения

- `outputs/images/week6_model_comparison.png` — сравнение F1-score;
- `outputs/images/week6_confusion_matrix.png` — confusion matrix лучшей embedding-модели;
- `outputs/images/week6_pca_embeddings.png` — PCA-визуализация пространства документов лучшей модели.

### Модели

- `models/week6_word2vec_logreg.joblib`;
- `models/week6_glove_logreg.joblib`;
- `models/week6_fasttext_logreg.joblib`.

В `artifacts/week6_best_model.json` сохраняется название лучшей комбинации embedding + classifier.

## Фактический результат запуска

На текущем `SMSSpamCollection` получился лучший вариант **fastText + Logistic Regression**. При фиксированном split 80/20: Accuracy = **0.9623**, Precision = **0.8092**, Recall = **0.9396**, F1 = **0.8696**, ROC-AUC = **0.9848**. Word2Vec дал F1 = **0.8589**, GloVe — **0.6494**. Это результаты одного воспроизводимого учебного эксперимента на нашем корпусе.

## Как это работает

```text
SMSSpamCollection
        ↓
Week 5 word embeddings
        ↓
mean pooling слов → один вектор документа
        ↓
30-мерный document embedding
        ↓
Logistic Regression
        ↓
spam / ham
```

Отдельно:

```text
30-мерные document embeddings
        ↓
PCA
        ↓
2D
        ↓
визуализация ham / spam
```

## Установка

Для полного запуска проекта используйте **64-битный Python 3.12 или 3.13**. Для `gensim 4.4.0` есть готовые Windows x64 пакеты под эти версии; под Python 3.14 готового Windows x64 пакета нет, и `pip` пытается собрать `gensim` из исходников. В командной строке Windows проверьте установленные версии командой `py -0p`.

Windows (PowerShell, пример с Python 3.13):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --only-binary=gensim -r requirements.txt
```

Ключ `--only-binary=gensim` не позволяет незаметно перейти к сборке `gensim` из исходников: если подходящего пакета нет, установка сразу сообщит об этом. При наличии Python 3.12 замените `-3.13` на `-3.12`. Если команда `py -3.13` не найдена, установите 64-битный Python 3.13 и затем повторите команды. Все команды запуска проекта выполняйте через `.venv\Scripts\python.exe` либо активируйте это окружение.

Linux/macOS:

```bash
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Запуск Week 6

Модели Week 5 должны уже находиться в `models/`.

```bash
python -m scripts.week6_vector_prototype
```

## Запуск всего проекта

```bash
python main.py
```

## Быстрая проверка нового SMS

После выполнения Week 6 можно подать собственный текст:

```bash
python scripts/predict_sms.py "Congratulations! You won a free cash prize. Call now!"
```

## Неделя 7 — Transformer / Attention

`python -m scripts.week7_transformer_attention` анализирует 20 исходных SMS (по 10 каждого класса), сохраняет контекстные векторы BERT, сравнение с TF-IDF и веса attention для слова `free` в двух контекстах. Notebook: `notebooks/07_transformer_attention.ipynb`. Фактические результаты и ограничения: `reports/week7_conclusions.md`, данные: `reports/week7_*.csv` и `reports/week7_summary.json`. Входной корпус английский; используется `prajjwal1/bert-tiny` с закреплённой ревизией. Веса скачиваются при первом запуске.

## Неделя 8 — BERT без дообучения encoder

`python -m scripts.week8_bert_features` вычисляет mean-pooled признаки замороженного BERT на SMS, обучает линейный классификатор и сравнивает с TF-IDF + Logistic Regression на том же стратифицированном split. Результаты: `reports/week8_comparison.csv`, `reports/week8_predictions.csv`, `reports/week8_conclusions.md`. Notebook: `notebooks/08_bert_features.ipynb`. Линейная голова обучается с метками, поэтому этот этап называется feature extraction, не zero-shot. Для воспроизведения требуется скачать BERT-веса при первом запуске.

## Неделя 9 — Fine-tuning BERT

`python -m scripts.week9_bert_finetune` инициализирует линейную голову из недели 8 и дообучает вместе с ней весь encoder две эпохи на том же train split. Сравнение до/после на фиксированном test: `reports/week9_before_after.csv`, индивидуальные результаты: `reports/week9_predictions.csv`, выводы: `reports/week9_conclusions.md`; notebook: `notebooks/09_bert_finetune.ipynb`. Перед запуском недели 9 модель недели 8 создаётся автоматически при отсутствии её файла. Чекпойнт сохраняется в `models/week9_finetuned_bert.pt`.

## Неделя 10 — эксперименты и анализ ошибок (четвёртая неделя задания)

```bash
python -m scripts.week10_experiments
```

Пять независимых обучений исследуют learning rate `1e-5 / 2e-5 / 5e-5` при 100% train
и доли `50% / 75% / 100%` при LR `2e-5`. Все запуски — две эпохи, batch size 32, seed 42.
Каждый вариант начинает с исходного BERT; линейная голова обучается только на своей доле train.
Готовый checkpoint недели 9 оценивается отдельно и не используется для инициализации этих запусков.

Результаты уже сохранены: [аналитический отчёт](reports/week10_report.md),
[таблица экспериментов](reports/week10_experiments.csv), [все 15 ошибок базовой модели](reports/week10_errors.csv),
[график метрик](outputs/images/week10_metric_dynamics.png),
[ноутбук](notebooks/10_experiments_error_analysis.ipynb).

| Эксперимент | Доля train | F1 spam |
|---|---:|---:|
| Сохранённая модель недели 9 | 100% | 0.949153 |
| Новое обучение, LR 1e-5 | 100% | 0.934641 |
| Новое обучение, LR 2e-5 | 100% | 0.940789 |
| Новое обучение, LR 5e-5 | 100% | 0.953020 |
| Новое обучение, LR 2e-5 | 50% | 0.930693 |
| Новое обучение, LR 2e-5 | 75% | 0.934641 |

Источники чисел — `reports/week10_experiments.csv` и индивидуальные вероятности
`reports/week10_predictions.csv`. Новый порядок батчей фиксирован отдельным генератором;
повторно обученная базовая конфигурация отличается от сохранённого checkpoint, поэтому обе строки показаны отдельно.
Результаты относятся к одному seed и одному разбиению, без подбора параметров по test.

В прежнем test обнаружены 128 строк с точным текстовым совпадением в train.
Для сопоставимости сохранён прежний test, дополнительно все модели оценены на одних и тех же 987 строках
без точных совпадений в полном train (`clean_*` в таблице). Это не заменяет независимый внешний корпус.
Причины ошибок в отчёте — ручные гипотезы по тексту, исходные метки не изменены.

Для обновления отчёта и рисунков без обучения:

```bash
python -m scripts.week10_experiments --report-only
```

Проверки протокола:

```bash
python -m unittest discover -s tests -v
```
