# Spam Detector — учебный проект (недели 1–4)

Проект анализирует датасет `data/SMSSpamCollection` и готовит данные для классификации SMS-сообщений на `spam` и `ham`.

## Что формируется после запуска

После выполнения `python main.py` проект создаёт:

- CSV-таблицы с результатами анализа в папке `reports/`;
- PNG-фотографии графиков и облаков слов в папке `outputs/images/`;
- отдельные CSV-файлы с обработанным датасетом, train/test-выборками и характеристиками матриц.

### Основные изображения

- `outputs/images/week1_class_distribution.png` — распределение классов;
- `outputs/images/week4_bow_tfidf_top20.png` — топ-20 слов BOW и TF-IDF;
- `outputs/images/week4_bow_wordcloud.png` — облако слов BOW;
- `outputs/images/week4_tfidf_wordcloud.png` — облако слов TF-IDF.

### Основные CSV-файлы

- `reports/week1_summary.csv`;
- `reports/week1_grouped_stats.csv`;
- `reports/week2_before_after.csv`;
- `reports/week2_length_comparison.csv`;
- `reports/week3_stemming_examples.csv`;
- `reports/week3_spacy_lemmas.csv` — если установлена модель spaCy;
- `reports/week3_top50_tokens.csv`;
- `reports/week3_top50_porter.csv`;
- `reports/week3_top50_snowball.csv`;
- `reports/week4_top100_bow.csv`;
- `reports/week4_top100_tfidf.csv`;
- `reports/week4_matrix_comparison.csv`;
- `reports/week4_processed_dataset.csv`;
- `reports/week4_train.csv`;
- `reports/week4_test.csv`.

## Установка и запуск

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

Ресурсы NLTK:

```bash
python -m nltk.downloader punkt punkt_tab stopwords
```

Модель spaCy (необходима для лемматизации):

```bash
python -m spacy download en_core_web_sm
```

Запуск всех этапов:

```bash
python main.py
```

Все результаты будут созданы автоматически. При повторном запуске CSV и изображения обновляются.
