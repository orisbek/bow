"""Build the assignment Week 5 hand-in report after real browser screenshots."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = json.loads((ROOT / "reports" / "week11_model_reproduction.json").read_text(encoding="utf-8"))
    controls = json.loads((ROOT / "reports" / "week11_control_tests.json").read_text(encoding="utf-8"))
    for item in controls:
        if not (ROOT / item["screenshot"]).exists():
            raise FileNotFoundError(f'Capture the actual UI first: {item["screenshot"]}')
    recorded, actual = metadata["recorded_metrics"], metadata["reproduced_metrics"]
    table = "\n".join(
        f'| {item["id"]} | {item["title"]} | {item["expected"]} | {item["label"]} | '
        f'{item["confidence"]:.2%} | {item["probability_spam"]:.6f} |'
        for item in controls)
    screenshots = "\n\n".join(
        f'### Тест {item["id"]}. {item["title"]}\n\n'
        f'> {item["text"]}\n\n'
        f'Ожидаемый класс: **{item["expected"]}**. Предсказание: **{item["label"]}**, '
        f'уверенность {item["confidence"]:.2%}.\n\n'
        f'![Тест {item["id"]}](../{item["screenshot"]})'
        for item in controls)
    text = f'''# Неделя 5. Пользовательское NLP-приложение

## Результат

Создано приложение **Streamlit** (`app.py`) для классификации английских SMS на
**ham** (обычная переписка) и **spam** (нежелательная рассылка).
Это Week 11 в нумерации исходного репозитория: предшествующие эксперименты
Week 10 являются 4-й неделей задания. Существующий этап word embeddings Week 5 сохранён.

## Выбор и воспроизведение модели

Источник выбора — `reports/week10_experiments.csv`. Из шести вариантов выбран
максимальный F1 spam: **{metadata["experiment"]}**. Это закреплённый
`{metadata["model"]}`, revision `{metadata["revision"]}`, LR=5e-5, две эпохи,
batch size 32, seed 42, max_length 96. Используются masked mean pooling и
линейная голова; порог spam — 0,5. Классификатор инициализирован Logistic Regression
на замороженных признаках только train, затем дообучен вместе с encoder.

У победителя отсутствовали сохранённые веса. `python -m scripts.prepare_app_model`
повторяет **только выбранную конфигурацию** через исходную функцию обучения
`run_training`, дополненную необязательным сохранением checkpoint. Отчёты и
метрики Week 10 при этом не перезаписываются, модель Week 9 не используется как замена.

| Метрика | Результат 4-й недели | Воспроизведённая модель приложения |
| --- | ---: | ---: |
| Accuracy | {recorded["accuracy"]:.6f} | {actual["accuracy"]:.6f} |
| Precision spam | {recorded["precision_spam"]:.6f} | {actual["precision_spam"]:.6f} |
| Recall spam | {recorded["recall_spam"]:.6f} | {actual["recall_spam"]:.6f} |
| F1 spam | {recorded["f1_spam"]:.6f} | {actual["f1_spam"]:.6f} |

Train/test: {metadata["train_size"]}/{metadata["test_size"]}, прежнее разбиение сверено.
Совпали классы **{metadata["matching_test_predictions"]}/{metadata["test_size"]}** тестовых SMS;
максимальная разница вероятности spam — **{metadata["max_probability_difference"]:.8g}**.
На 987 тестовых SMS без точного совпадения с полным train получен F1
**{actual["clean_f1_spam"]:.6f}**. Полные параметры, контрольные суммы, потери по эпохам
и версии библиотек: `week11_model_reproduction.json`; все новые test-предсказания:
`week11_test_predictions.csv`. Различие LF/CRLF при Git checkout учтено при проверке корпуса.

## Функционал

- Многострочное поле для английского SMS и кнопка «Анализировать».
- Пять встроенных контрольных сообщений, которые заполняют поле ввода.
- Предсказанный класс ham/spam и его вероятность в процентах.
- Горизонтальный график P(ham) и P(spam), числовые значения обеих вероятностей.
- Скачать результат JSON с исходным проанализированным текстом и идентификатором модели.
- Проверки пустого ввода, отсутствия английских букв и предела в 4 000 символов.
- Предупреждение об обрезке при превышении 96 токенов и о смешанном тексте с кириллицей.
- Понятная подсказка, если отсутствуют файлы модели. Веб-интерфейс не запускает обучение.
- Модель кешируется между запросами и после подготовки загружается исключительно локально.

Вероятность выбранного класса: P(spam), если P(spam) ≥ 0,5, иначе 1 − P(spam).
Тексты не записываются приложением на диск и не передаются внешним API;
при запуске на другом сервере браузер отправляет текст этому серверу Streamlit.
Сохранение JSON выполняется пользователем по кнопке. Результат формы относится к
последнему нажатию «Анализировать».

## Пять новых контрольных текстов

Ожидаемые метки заданы вручную до проверки. Тексты не совпадают ни с одной строкой
исходного корпуса без учёта регистра. Для каждого примера выбрана опция в форме
Streamlit, нажата кнопка анализа; UI-результат сверяется с прямым вызовом той же модели.
После этого те же пять текстов проверены в настоящем браузере и сохранены скриншоты.
Пройдено по ожидаемым меткам: **{sum(item["correct"] for item in controls)}/5**.

| № | Сценарий | Ожидается | Получено | Уверенность | P(spam) |
| --- | --- | --- | --- | ---: | ---: |
{table}

CSV/JSON с полными значениями: `week11_control_tests.csv`, `week11_control_tests.json`.
Это функциональные примеры; пять вручную подобранных SMS не являются надёжной оценкой качества.

## Скриншоты работающего UI

{screenshots}

## Установка и запуск

Из корня репозитория (Windows PowerShell):

```powershell
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements-app.txt
.venv\\Scripts\\python.exe -m streamlit run app.py
```

Linux/macOS: используйте `.venv/bin/python`. При отсутствии подготовленных файлов:
`python -m scripts.prepare_app_model`; для той же ревизии из локального каталога:
`python -m scripts.prepare_app_model --model-path artifacts/bert-tiny`.
Подготовленная модель включена в сдачу (`models/week11_best_bert/`).
Проверенное окружение: Python {metadata["python"]}; точные зависимости закреплены
в `requirements-app-lock.txt`. Для иных платформ используйте диапазоны `requirements-app.txt`.

Проверки:

```powershell
.venv\\Scripts\\python.exe -m scripts.test_app_controls
.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.venv\\Scripts\\python.exe -m scripts.build_app_report
```

Тесты покрывают согласованность метрик Week 10, все пять сценариев формы,
расчёт вероятностей и уверенности, очистку устаревшего результата после некорректного
ввода и предупреждение об обрезке. Скриншоты требуют отдельного запуска браузера.

## Ограничения

Модель обучена на английских SMS Spam Collection. Для русского/казахского языка,
других доменов и современных схем спама качество не проверено. Вероятности не
калибровались. Часть исторического test имеет дубликаты в train; выбор лучшей конфигурации
по тому же test не даёт независимой оценки её обобщения. Порог и модель не подбирались
по пяти новым контрольным сообщениям.

## Источники

- [Исходный репозиторий](https://github.com/orisbek/bow), commit `d3127a7a608f8d1fe6e163d0a5e32505379de540`.
- [Streamlit: форма и кнопка анализа](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form_submit_button).
- [Streamlit: кеширование модели](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).
- [Streamlit: AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
- [Transformers: Auto classes и локальная загрузка](https://huggingface.co/docs/transformers/model_doc/auto).
'''
    (ROOT / "reports" / "week11_app_report.md").write_text(text, encoding="utf-8")
    print("reports/week11_app_report.md")


if __name__ == "__main__":
    main()
