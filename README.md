# Spam Detector — учебный NLP-проект (недели 1–6)

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

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
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
