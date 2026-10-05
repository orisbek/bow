# Spam Detector — учебный NLP-проект (недели 1–5)

Проект анализирует датасет `data/SMSSpamCollection` и постепенно готовит его к классификации SMS на `spam` и `ham`.

## Неделя 5 — Word2Vec, GloVe, fastText

В Week 5 реализованы все задачи из плана:

- обучение собственной модели **Word2Vec** с помощью Gensim;
- поиск ближайших слов по cosine similarity;
- арифметика векторов;
- собственная компактная реализация **GloVe** на разреженной матрице совместной встречаемости;
- обучение **fastText** через Gensim с символьными n-граммами;
- сравнение Word2Vec, GloVe и fastText на одинаковом корпусе и размерности векторов;
- сохранение моделей, CSV-отчётов и PNG-визуализаций;
- семантические примеры для домена SMS.

Важно: сравнение качества в `week5_similarity_pairs.csv` — это **внутренний, доменно-ориентированный proxy-тест**, а не официальный универсальный benchmark. Пары слов заранее заданы как связанные по смыслу термины из SMS-словаря.

## Что создаётся после запуска

### Модели

- `models/week5_word2vec.model` — полная модель Word2Vec Gensim;
- `models/week5_word2vec.kv` — KeyedVectors Word2Vec;
- `models/week5_fasttext.model` — полная модель fastText Gensim;
- `models/week5_fasttext.kv` — KeyedVectors fastText;
- `models/week5_glove.npz` — веса GloVe;
- `models/week5_glove_vocab.json` — словарь GloVe.

### CSV

- `reports/week5_vocab_stats.csv` — статистика корпуса;
- `reports/week5_model_comparison.csv` — размерность, словарь и время обучения;
- `reports/week5_semantic_neighbors.csv` — ближайшие слова для выбранных anchors;
- `reports/week5_similarity_pairs.csv` — cosine similarity для семантических пар;
- `reports/week5_similarity_pairs_long.csv` — та же оценка в long-формате;
- `reports/week5_vector_arithmetic.csv` — результаты арифметики векторов.

### Изображения

- `outputs/images/week5_model_quality.png` — сравнение средней cosine similarity;
- `outputs/images/week5_free_neighbors.png` — ближайшие слова к `free` для трёх моделей.

## Параметры Week 5

Для сопоставимости модели обучаются на одном и том же корпусе после общего preprocessing:

- `vector_size = 30`;
- `window = 4`;
- `min_count = 2`;
- `seed = 42`;
- Word2Vec: skip-gram, 10 epochs;
- fastText: skip-gram, 10 epochs, character n-grams 3–5;
- GloVe: окно 4, 10 epochs, weighted co-occurrence, AdaGrad.

## Установка

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

При необходимости NLTK:

```bash
python -m nltk.downloader punkt punkt_tab stopwords
```

Для Week 3, если нужна spaCy-лемматизация:

```bash
python -m spacy download en_core_web_sm
```

## Запуск

Все недели:

```bash
python main.py
```

Только Week 5:

```bash
python -m scripts.week5_embeddings
```

После запуска модели, CSV и изображения обновляются автоматически.
