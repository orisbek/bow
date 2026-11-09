"""Week 10 (assignment week 4): controlled BERT experiments and error analysis.

Run: python -m scripts.week10_experiments
The Week 9 checkpoint is evaluated, never used to initialize fraction experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.data import load_sms_data
from scripts.week7_transformer_attention import MODEL, REVISION
from scripts.week8_bert_features import MAX_LENGTH, SEED, metrics, split_indices
from scripts.week9_bert_finetune import SmsDataset, predict

ROOT = Path(__file__).resolve().parents[1]
EPOCHS, BATCH_SIZE = 2, 32
CONFIGS = [('lr_1e-5_full', 1e-5, 1.0), ('lr_2e-5_full', 2e-5, 1.0),
           ('lr_5e-5_full', 5e-5, 1.0), ('data_50', 2e-5, .5), ('data_75', 2e-5, .75)]

# Manual, text-based hypotheses for ALL errors of the saved Week 9 checkpoint.
# They are not verified causal explanations, and do not change original labels.
ANNOTATIONS = {
    5: ('Неоднозначный контекст', 'Реклама платного сервиса оформлена как личное сообщение; указана цена £1.50.', 'Добавить в train подобные замаскированные рекламные SMS; сохранить цены и сокращения.'),
    1086: ('Нестандартное написание', 'Поздравительный текст содержит FR\'NDSHIP, V, nt, 4few; причина ошибки предположительная.', 'Добавить размеченные личные SMS с сокращениями; проверить символьные признаки.'),
    5449: ('Неоднозначность разметки', 'По видимому тексту это короткая шутка; контекст рассылки отсутствует, метка spam требует проверки.', 'Проверить происхождение сообщения и правила разметки; не менять метку без подтверждения.'),
    263: ('Признаки, похожие на спам', 'Личное сообщение содержит номер телефона и верхний регистр.', 'Добавить обычные SMS с телефонами и верхним регистром как трудные отрицательные примеры.'),
    1430: ('Неоднозначный контекст', 'Объявление For sale заканчивается шуткой; из одного текста намерение отправителя неясно.', 'Собрать дополнительные размеченные объявления и шутки; проверить контекст источника.'),
    1266: ('Нестандартное написание', 'Личное сообщение содержит слитные слова, сокращения и необычные символы.', 'Проверить нормализацию кодировки; добавить примеры с неформальным написанием.'),
    5046: ('Признаки, похожие на спам', 'Деловое сообщение использует Customer Service, mail id и contact us.', 'Добавить легитимные рабочие и сервисные сообщения как трудные отрицательные примеры.'),
    2804: ('Короткое сообщение', 'Короткая реклама мелодий содержит FreeMsg и Reply; информации мало.', 'Добавить короткие рекламные сообщения и проверить ошибки по длине текста.'),
    3539: ('Признаки, похожие на спам', 'Уведомление об установке интернета похоже на массовую сервисную рассылку.', 'Разграничить транзакционные уведомления и нежелательную рекламу в инструкции разметки.'),
    2823: ('Неоднозначный контекст', 'Сообщение с префиксом ROMCAPspam выглядит как гороскоп или личное пожелание.', 'Проверить служебный префикс и происхождение; добавить примеры некоммерческого спама.'),
    751: ('Неоднозначность разметки', 'Шутка не содержит явного предложения; без контекста нельзя подтвердить причину метки spam.', 'Проверить правила для массовых шуток и цепочных сообщений, сохранив исходную метку.'),
    3094: ('Короткое сообщение', 'Сообщение состоит из URL учебной страницы; URL сам по себе не определяет класс.', 'Добавить легитимные ссылки и рекламные ссылки; проверить признаки домена и контекста.'),
    3302: ('Нестандартное написание', 'Короткий ответ с RCT, THNQ и именами выглядит как личная переписка.', 'Проверить источник и разметку; добавить спам без типичных рекламных слов.'),
    2247: ('Неоднозначный контекст', 'Текст содержит предупреждение о мошенничестве и цитату приманки; текущая метка spam требует контекста.', 'Проверить, размечается ли предупреждение или сама рассылка; дополнить инструкцию.'),
    1507: ('Неоднозначный контекст', 'Реклама караоке начинается с благодарности за голосование и содержит FREE и reply.', 'Добавить рекламные сообщения с нейтральным вступлением; проверить достаточно ли контекстных примеров.'),
}


class Classifier(nn.Module):
    """Exactly the encoder, masked mean pooling and linear head of Week 9."""
    def __init__(self, head, model_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, revision=REVISION)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(head.coef_, dtype=torch.float32))
            self.head.bias.copy_(torch.tensor(head.intercept_, dtype=torch.float32))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        return self.head((states * mask).sum(1) / mask.sum(1)).squeeze(-1)


def nested_subsets(train, labels):
    """Random nested subsets; original full train ordering is retained at 100%."""
    result = {1.0: np.asarray(train)}
    for fraction in (.75, .5):
        # Each smaller sample is stratified WITHIN the next larger one.
        parent = result[1.0 if fraction == .75 else .75]
        count = int(np.floor(len(train) * fraction))
        ids, _ = train_test_split(parent, train_size=count, random_state=SEED,
                                  stratify=labels[parent])
        result[fraction] = ids
    return result


def frozen_features(messages, tokenizer, model_path, fingerprint):
    cache = ROOT / 'artifacts' / 'week10_features.npz'
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['fingerprint']) == fingerprint:
                return saved['features']
    encoder = AutoModel.from_pretrained(model_path, revision=REVISION).eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(messages), 64):
            batch = tokenizer(messages[start:start + 64], padding=True, truncation=True,
                              max_length=MAX_LENGTH, return_tensors='pt')
            states = encoder(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            rows.append(((states * mask).sum(1) / mask.sum(1)).numpy())
    features = np.vstack(rows)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, features=features, fingerprint=fingerprint)
    return features


def scored(name, y, prob, train_size, clean_mask):
    row = metrics(name, y, (prob >= .5).astype(int), prob, train_size)
    tn, fp, fn, tp = confusion_matrix(y, prob >= .5, labels=[0, 1]).ravel()
    row.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    # Same non-overlapping evaluation rows for every run (against FULL train).
    clean = metrics(name, y[clean_mask], (prob[clean_mask] >= .5).astype(int),
                    prob[clean_mask], train_size)
    row.update({f'clean_{k}': v for k, v in clean.items() if k not in ('model', 'train_size')})
    return row


def run_training(name, lr, ids, labels, df, features, tokenizer, model_path, test_loader):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    head = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
    head.fit(features[ids], labels[ids])
    classifier = Classifier(head, model_path)
    train_data = SmsDataset(df.message.iloc[ids].tolist(), labels[ids], tokenizer)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    counts = np.bincount(labels[ids], minlength=2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(counts[0] / counts[1], dtype=torch.float32))
    history = []
    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        total, n = 0., 0
        for batch in loader:
            y = batch.pop('labels').float()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(classifier(**batch), y)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), 1.)
            optimizer.step()
            total += float(loss.item()) * len(y)
            n += len(y)
        history.append({'experiment': name, 'epoch': epoch, 'train_loss': total / n})
        print(json.dumps(history[-1]), flush=True)
    # No checkpoint or parameter selection on test: one evaluation after fixed epochs.
    probability = predict(classifier, test_loader)
    return probability, history


def error_analysis(df, test, probability, reports):
    part = df.iloc[test].reset_index(names='source_row')
    part['predicted_label'] = np.where(probability >= .5, 'spam', 'ham')
    part['probability_spam'] = probability
    part = part[part.label != part.predicted_label].copy()
    if len(part) < 10:
        raise ValueError('The reference checkpoint has fewer than 10 errors; do not invent errors.')
    if set(part.source_row) != set(ANNOTATIONS):
        raise ValueError('Reference errors changed; manually review annotations before publishing.')
    part['error_type'] = np.where(part.label == 'spam', 'FN: пропущенный спам', 'FP: ложная тревога')
    part['category_hypothesis'] = part.source_row.map(lambda i: ANNOTATIONS[i][0])
    part['reason_hypothesis'] = part.source_row.map(lambda i: ANNOTATIONS[i][1])
    part['proposed_solution'] = part.source_row.map(lambda i: ANNOTATIONS[i][2])
    part['annotation_status'] = 'Ручная гипотеза по тексту; причинность не проверена; исходная метка сохранена'
    part.to_csv(reports / 'week10_errors.csv', index=False)
    groups = part.groupby('category_hypothesis').size().reset_index(name='error_count')
    groups.to_csv(reports / 'week10_error_groups.csv', index=False)
    return part, groups


def plots(results, reports, images):
    fresh = results[results.kind == 'fresh_training']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, part, x, xlabel in [
        (axes[0], fresh[fresh.train_fraction == 1.].sort_values('learning_rate'), 'learning_rate', 'Learning rate'),
        (axes[1], fresh[fresh.learning_rate == 2e-5].sort_values('train_fraction'), 'train_fraction', 'Train fraction')]:
        for metric in ['accuracy', 'precision_spam', 'recall_spam', 'f1_spam']:
            ax.plot(part[x], part[metric], marker='o', label=metric.replace('_spam', ''))
        ax.set_xticks(part[x], [f'{v:g}' for v in part[x]])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Score (zoomed axis; positive class: spam)')
        ax.set_ylim(.85, 1.005)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Controlled experiments: 2 epochs, seed 42, fixed test')
    fig.tight_layout()
    fig.savefig(images / 'week10_metric_dynamics.png', dpi=180)
    plt.close(fig)
    baseline = results[results.kind == 'saved_checkpoint'].iloc[0]
    cm = np.array([[baseline.tn, baseline.fp], [baseline.fn, baseline.tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'],
           xlabel='Predicted', ylabel='True', title='Saved Week 9 checkpoint')
    fig.tight_layout()
    fig.savefig(images / 'week10_confusion_matrix.png', dpi=180)
    plt.close(fig)
    pd.DataFrame(cm, index=['true_ham', 'true_spam'], columns=['pred_ham', 'pred_spam']).to_csv(reports / 'week10_confusion_matrix.csv')


def markdown_table(frame):
    """Render a small Markdown table without an optional tabulate dependency."""
    columns = list(frame.columns)
    def cell(value):
        if isinstance(value, (float, np.floating)):
            value = f'{value:.6f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    rows = ['| ' + ' | '.join(map(cell, columns)) + ' |',
            '| ' + ' | '.join('---' for _ in columns) + ' |']
    rows.extend('| ' + ' | '.join(cell(value) for value in row) + ' |'
                for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def report(results, errors, groups, audit, config, reports):
    fresh = results[results.kind == 'fresh_training']
    lr = fresh[fresh.train_fraction == 1.].sort_values('learning_rate')
    size = fresh[fresh.learning_rate == 2e-5].sort_values('train_fraction')
    best_lr = lr.loc[lr.f1_spam.idxmax()]
    baseline = results[results.kind == 'saved_checkpoint'].iloc[0]
    columns = ['experiment', 'learning_rate', 'train_fraction', 'train_size', 'accuracy', 'precision_spam', 'recall_spam', 'f1_spam']
    clean_cols = ['experiment', 'clean_test_size', 'clean_accuracy', 'clean_precision_spam', 'clean_recall_spam', 'clean_f1_spam']
    text = f'''# Экспериментальное исследование и анализ ошибок

Это четвёртая неделя задания; в структуре существующего проекта — Week 10.
Все числа ниже получены запуском `python -m scripts.week10_experiments`.

## Базовая модель и данные

Исходный корпус: `data/SMSSpamCollection`, {audit['rows']} SMS; ham={audit['ham']}, spam={audit['spam']}.
Положительный класс — spam, порог 0.5. Исходное стратифицированное разбиение недели 8:
{audit['train_rows']} train / {audit['test_rows']} test, seed=42. Метки и source_row сверены с сохранёнными предсказаниями.
Базовая модель — сохранённый `models/week9_finetuned_bert.pt`: BERT-tiny, 2 эпохи, LR=2e-5.
Повторная оценка checkpoint сверена с `reports/week9_predictions.csv`: максимальное абсолютное
расхождение вероятностей {config['reference_max_probability_difference']:.8g}; число несовпадающих классов 0.
Accuracy={baseline.accuracy:.6f}, Precision={baseline.precision_spam:.6f}, Recall={baseline.recall_spam:.6f}, F1={baseline.f1_spam:.6f}.
Ошибок {len(errors)}: FP={int(baseline.fp)}, FN={int(baseline.fn)}.

## Методика

Пять независимых запусков: LR=1e-5 / 2e-5 / 5e-5 на полном train и 50% / 75% / 100% train при LR=2e-5.
Общий вариант 100%, LR=2e-5 используется в обоих сравнениях, поэтому достаточно пяти обучений.
Каждый запуск начинает с одной закреплённой исходной ревизии `{MODEL}` ({REVISION}), а не с checkpoint, уже видевшего весь train.
Как в неделе 9, голова инициализируется Logistic Regression на замороженных BERT-признаках.
Но она всякий раз обучена только на разрешённой доле train; одинаковая начальная голова используется для всех трёх LR.
Подвыборки вложенные и стратифицированные: 50% входит в 75%, 75% — в 100%; размеры округлены вниз.
Batch=32, 2 эпохи, max_length=96, masked mean pooling включая CLS/SEP, AdamW (weight_decay=0.01),
BCEWithLogitsLoss с положительным весом ham/spam из текущего train, clip_grad_norm=1.
Seed=42, отдельный генератор порядка батчей, CPU, не более четырёх потоков.
Этот порядок батчей зафиксирован явно; новое обучение базовой конфигурации не обязано побитово повторить прежний checkpoint.
Исходные SMS передаются токенизатору BERT без прежнего удаления стоп-слов.
Тест не используется для обучения, подбора порога, числа эпох или остановки.
Это заранее заданное описательное сравнение: максимальный test F1 не является независимой оценкой выбранной модели.
Сокращение train меняет и число шагов оптимизации при фиксированных эпохах; эффект объёма включает это изменение.

## Сводная таблица

{markdown_table(results[columns])}

Полная точность, длительность, train/test counts и confusion counts — `week10_experiments.csv`.
Сохранённая модель — отдельная строка reference; заново обученная базовая конфигурация — отдельная строка.

## Динамика метрик

![Динамика метрик](../outputs/images/week10_metric_dynamics.png)

На полном train максимальный наблюдаемый F1 среди заданных LR: {best_lr.learning_rate:g}, F1={best_lr.f1_spam:.6f}.
Это описание данного запуска, а не доказательство универсально лучшего LR.
При 50%, 75% и 100% train получены соответственно F1={size.f1_spam.iloc[0]:.6f}, {size.f1_spam.iloc[1]:.6f}, {size.f1_spam.iloc[2]:.6f}.
Разница 100% − 50%: {size.f1_spam.iloc[2] - size.f1_spam.iloc[0]:+.6f}.
Одна реализация seed и одно разбиение не позволяют оценить статистическую устойчивость различий.

## Дубликаты и дополнительная оценка

В корпусе {audit['duplicate_excess']} повторных строк по точному тексту сверх первого вхождения.
Между train и test пересекаются {audit['shared_unique_messages']} уникальных текстов;
{audit['test_rows_overlapping_train']} тестовых строк имеют точное совпадение в полном train.
Основная таблица сохраняет прежний test для сопоставимости. Дополнительно один и тот же фиксированный
набор из {audit['clean_test_rows']} тестовых строк без точных совпадений в ПОЛНОМ train оценён для всех моделей:

{markdown_table(results[clean_cols])}

Этот дополнительный набор исключает точное пересечение текстов, но не гарантирует отсутствие близких шаблонов
и не заменяет внешний корпус. Для следующего исследования нужно отдельное разбиение по группам одинаковых
и близких сообщений до всех этапов обучения. Метрики текущей модели нельзя объявлять проверкой на независимом внешнем корпусе.

## Анализ всех ошибок исходной дообученной модели

Выбраны все {len(errors)} ошибок сохранённого checkpoint, а не только удобные примеры.
`week10_errors.csv` содержит исходный текст, source_row, метки, вероятность, FP/FN,
предполагаемую категорию, объяснение и предложенное решение. Полная таблица приведена ниже.
Категории — ручные гипотезы по видимому тексту; причинность не проверена.
Неоднозначность метки не означает доказанную ошибку разметчика. Исходные метки сохранены.

{markdown_table(errors[['source_row', 'message', 'label', 'predicted_label', 'probability_spam', 'category_hypothesis', 'reason_hypothesis', 'proposed_solution']])}

### Группы

{markdown_table(groups)}

Дисбаланс классов: доля spam={audit['spam']/audit['rows']:.2%}; это свойство корпуса, а не установленная причина каждого промаха.
Точность всегда предсказывать ham равна {audit['test_ham']/audit['test_rows']:.6f} на данном test,
поэтому Accuracy рассматривается вместе с Precision/Recall/F1 и матрицей ошибок.
Текущий weighted loss уже учитывает соотношение классов. Эффект иных весов или oversampling здесь не проверялся.

## Предлагаемые улучшения

1. Проверить неоднозначные метки и происхождение сообщений, не исправляя их по догадке.
2. Добавить в обучающий корпус легитимные ссылки, уведомления и личные SMS с телефонами,
   а также рекламу, оформленную как личная переписка. Не переносить тестовые ошибки в train при сохранении текущей оценки.
3. Отдельно проверить влияние SMS-сокращений, кодировки и длины, используя новую размеченную выборку.
4. Подбирать LR, порог и веса классов на validation внутри train; после выбора оценивать на новом независимом test.
5. Повторить эксперименты с несколькими seed и групповым разбиением, чтобы проверить устойчивость наблюдений.

## Воспроизведение и источники

Команда: `python -m scripts.week10_experiments`. Для заранее скачанной той же ревизии:
`python -m scripts.week10_experiments --model-path /path/to/bert-tiny`.
Локальный каталог должен содержать исходные config.json, vocab.txt, pytorch_model.bin именно указанной ревизии.
Конфигурация, SHA256 исходных данных/checkpoint, версии библиотек и время запуска — `week10_config.json`.
Разбиение и принадлежность подвыборкам — `week10_split.csv`; все вероятности — `week10_predictions.csv`;
потери по эпохам — `week10_training_history.csv`; метрики по обоим классам — `week10_classification_reports.json`.
Входы: исходный корпус, checkpoint и результаты Week 9; новый код: `scripts/week10_experiments.py`.
'''
    (reports / 'week10_report.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default=MODEL, help='Pinned Hub model, or a local copy of that revision')
    parser.add_argument('--report-only', action='store_true', help='Rebuild report and figures from completed CSV/JSON results')
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.use_deterministic_algorithms(True)
    reports, images = ROOT / 'reports', ROOT / 'outputs' / 'images'
    reports.mkdir(exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        config = json.loads((reports / 'week10_config.json').read_text(encoding='utf-8'))
        results = pd.read_csv(reports / 'week10_experiments.csv')
        errors = pd.read_csv(reports / 'week10_errors.csv')
        groups = pd.read_csv(reports / 'week10_error_groups.csv')
        if len(results) != 6 or set(results.experiment) != {'week9_saved_reference', *(c[0] for c in CONFIGS)}:
            raise ValueError('Reports require all five completed experiments and the saved reference.')
        plots(results, reports, images)
        report(results, errors, groups, config['audit'], config, reports)
        print('Report and figures rebuilt from completed experimental results.')
        return
    df = load_sms_data().reset_index(drop=True)
    labels = (df.label == 'spam').astype(int).to_numpy()
    train, test = split_indices(labels)
    saved_split = pd.read_csv(reports / 'week8_split.csv')
    assert np.array_equal(np.sort(test), saved_split.loc[saved_split.split == 'test', 'source_row'])
    subsets = nested_subsets(train, labels)
    train_text = set(df.message.iloc[train])
    clean = ~df.message.iloc[test].isin(train_text).to_numpy()
    test_text = set(df.message.iloc[test])
    audit = {'rows': len(df), 'ham': int((labels == 0).sum()), 'spam': int(labels.sum()),
             'train_rows': len(train), 'test_rows': len(test), 'test_ham': int((labels[test] == 0).sum()),
             'duplicate_excess': int(df.duplicated('message').sum()),
             'shared_unique_messages': len(train_text & test_text),
             'test_rows_overlapping_train': int((~clean).sum()), 'clean_test_rows': int(clean.sum())}
    config = {'model': MODEL, 'revision': REVISION, 'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
              'max_length': MAX_LENGTH, 'optimizer': 'AdamW', 'weight_decay': .01, 'threshold': .5,
              'initialization': 'Original pretrained BERT; LogisticRegression fitted ONLY on current train subset',
              'selection': 'Five prespecified configurations; no test-based tuning or early stopping',
              'dataset_sha256': hashlib.sha256((ROOT / 'data' / 'SMSSpamCollection').read_bytes()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256((ROOT / 'models' / 'week9_finetuned_bert.pt').read_bytes()).hexdigest(),
              'python': platform.python_version(), 'versions': {p: importlib.metadata.version(p) for p in
                  ['torch', 'transformers', 'numpy', 'pandas', 'scikit-learn', 'matplotlib']},
              'device': 'cpu', 'threads': torch.get_num_threads(), 'started_utc': pd.Timestamp.now(tz='UTC').isoformat(),
              'experiments': CONFIGS, 'audit': audit}
    print(json.dumps(audit), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=REVISION)
    fingerprint = hashlib.sha256((config['dataset_sha256'] + MODEL + REVISION + str(MAX_LENGTH)).encode()).hexdigest()
    features = frozen_features(df.message.tolist(), tokenizer, args.model_path, fingerprint)
    reference_head = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED).fit(features[train], labels[train])
    reference = Classifier(reference_head, args.model_path)
    reference.load_state_dict(torch.load(ROOT / 'models' / 'week9_finetuned_bert.pt', map_location='cpu', weights_only=True))
    test_loader = DataLoader(SmsDataset(df.message.iloc[test].tolist(), labels[test], tokenizer), batch_size=64)
    reference_prob = predict(reference, test_loader)
    old = pd.read_csv(reports / 'week9_predictions.csv')
    assert np.array_equal(old.source_row.to_numpy(), test)
    assert np.array_equal(old.true_label.to_numpy(), df.label.iloc[test].to_numpy())
    assert np.array_equal((reference_prob >= .5), (old.after_pred == 'spam').to_numpy())
    assert np.allclose(reference_prob, old.after_probability_spam.to_numpy(), atol=3e-5)
    config['reference_max_probability_difference'] = float(np.max(np.abs(reference_prob - old.after_probability_spam.to_numpy())))
    rows, histories, predictions, class_reports = [], [], [], {}

    def record(name, probability, size, lr, fraction, kind, seconds):
        row = scored(name, labels[test], probability, size, clean)
        row.update(experiment=name, learning_rate=lr, train_fraction=fraction, epochs=EPOCHS,
                   kind=kind, training_seconds=seconds)
        rows.append(row)
        predictions.append(pd.DataFrame({'experiment': name, 'source_row': test, 'true_label': df.label.iloc[test].to_numpy(),
                            'predicted_label': np.where(probability >= .5, 'spam', 'ham'),
                            'probability_spam': probability, 'no_exact_match_in_full_train': clean}))
        class_reports[name] = classification_report(labels[test], probability >= .5, labels=[0, 1],
                                                      target_names=['ham', 'spam'], output_dict=True, zero_division=0)
        pd.DataFrame(rows).to_csv(reports / 'week10_experiments.csv', index=False)
        pd.concat(predictions).to_csv(reports / 'week10_predictions.csv', index=False)

    record('week9_saved_reference', reference_prob, len(train), 2e-5, 1., 'saved_checkpoint', 0.)
    errors, groups = error_analysis(df, test, reference_prob, reports)
    split = pd.DataFrame({'source_row': np.arange(len(df)), 'label': df.label,
                          'split': np.where(np.isin(np.arange(len(df)), test), 'test', 'train')})
    for fraction, ids in subsets.items():
        split[f'train_{int(fraction * 100)}pct'] = split.source_row.isin(ids)
    split.to_csv(reports / 'week10_split.csv', index=False)
    for name, lr, fraction in CONFIGS:
        started = time.perf_counter()
        probability, history = run_training(name, lr, subsets[fraction], labels, df, features,
                                             tokenizer, args.model_path, test_loader)
        histories.extend(history)
        record(name, probability, len(subsets[fraction]), lr, fraction, 'fresh_training', time.perf_counter() - started)
        pd.DataFrame(histories).to_csv(reports / 'week10_training_history.csv', index=False)
        print(json.dumps(rows[-1]), flush=True)
    config['finished_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
    (reports / 'week10_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (reports / 'week10_classification_reports.json').write_text(json.dumps(class_reports, indent=2), encoding='utf-8')
    results = pd.DataFrame(rows)
    plots(results, reports, images)
    report(results, errors, groups, audit, config, reports)
    print(results[['experiment', 'train_size', 'accuracy', 'precision_spam', 'recall_spam', 'f1_spam']].to_string(index=False))


if __name__ == '__main__':
    main()
