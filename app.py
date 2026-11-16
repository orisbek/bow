"""Assignment Week 5: Streamlit UI for the best Week 4 SMS classifier."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_examples import CONTROL_TEXTS

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "week11_best_bert"

st.set_page_config(page_title="SMS · Spam Detector", page_icon="✉️", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1180px; padding-top: 4rem; padding-bottom: 1.5rem;}
h1 {font-size: clamp(1.75rem, 3.2vw, 2.4rem) !important; line-height: 1.2 !important;}
[data-testid="stSidebar"] {border-right: 1px solid #e5eaf3;}
textarea {font-size: 16px !important;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Загружаем модель…")
def load_detector():
    from src.app_inference import SpamDetector
    return SpamDetector(MODEL_DIR)


def select_example():
    choice = st.session_state.example
    st.session_state.sms = next((item["text"] for item in CONTROL_TEXTS
                                 if f'{item["id"]} · {item["title"]}' == choice), "")
    st.session_state.pop("result", None)


with st.sidebar:
    st.markdown("### ✉️ Spam Detector")
    st.caption("Учебный NLP-проект · Неделя 5")
    st.divider()
    st.markdown("**BERT-tiny · лучшая конфигурация**")
    st.caption("Выбрана по F1 класса spam в экспериментах 4-й недели (Week 10 в репозитории).")
    source = pd.read_csv(ROOT / "reports" / "week10_experiments.csv")
    best = source.sort_values(["f1_spam", "accuracy", "experiment"], ascending=[False, False, True]).iloc[0]
    st.metric("F1 · результаты 4-й недели", f"{best.f1_spam:.4f}")
    st.caption(f"Accuracy {best.accuracy:.2%} · LR {best.learning_rate:g}")
    st.caption("2 эпохи · 4 457 обучающих SMS")
    if (MODEL_DIR / "metadata.json").exists():
        meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        st.success("Модель подготовлена")
        st.caption(f'F1 воспроизведения: {meta["reproduced_metrics"]["f1_spam"]:.4f}')
    else:
        st.warning("Сначала подготовьте модель")
        st.code("python -m scripts.prepare_app_model", language="bash")
    st.divider()
    st.caption("Английские SMS · ham / spam")
    st.caption("После подготовки анализ работает локально, без обращения к внешним API.")

st.caption("АНАЛИЗ ТЕКСТА / КЛАССИФИКАЦИЯ SMS")
st.title("Это спам или обычное сообщение?")
st.write("Введите SMS на английском языке. Модель определит класс и покажет вероятности.")

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.subheader("Сообщение")
    st.selectbox("Контрольный пример", ["Свой текст"] +
                 [f'{item["id"]} · {item["title"]}' for item in CONTROL_TEXTS],
                 key="example", on_change=select_example)
    with st.form("sms_form"):
        text = st.text_area("Текст SMS", key="sms", height=180, max_chars=4000,
                            placeholder="Hey, are we still meeting at the library at six?")
        submitted = st.form_submit_button("Анализировать", type="primary", use_container_width=True)
    st.caption("До 4 000 символов. Модель использует первые 96 токенов, включая служебные.")

if submitted:
    st.session_state.pop("result", None)
    try:
        # Validate before loading the checkpoint, so empty input gets a useful message.
        from src.app_inference import validate_text
        clean_text = validate_text(text)
        st.session_state.result = load_detector().analyze(clean_text)
    except ValueError as error:
        st.error(str(error))
    except FileNotFoundError:
        st.error("Файлы модели не найдены. Выполните: python -m scripts.prepare_app_model")
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError):
        st.error("Не удалось загрузить модель. Повторите подготовку: python -m scripts.prepare_app_model --force")

with right:
    st.subheader("Результат анализа")
    result = st.session_state.get("result")
    if result is None:
        st.info("Результат появится после нажатия «Анализировать».")
        st.caption("ham — обычная переписка. spam — нежелательная рассылка.")
    else:
        if result["label"] == "spam":
            st.error("SPAM · Нежелательное сообщение")
        else:
            st.success("HAM · Обычное сообщение")
        st.metric("Уверенность в предсказанном классе", f'{result["confidence"]:.2%}')
        chart = pd.DataFrame({"Класс": ["ham", "spam"],
                              "Вероятность": [result["probability_ham"], result["probability_spam"]]})
        st.bar_chart(chart, x="Класс", y="Вероятность", horizontal=True,
                     color="#2563eb", height=150)
        a, b = st.columns(2)
        a.caption(f'P(ham) = {result["probability_ham"]:.4f}')
        b.caption(f'P(spam) = {result["probability_spam"]:.4f}')
        if result["truncated"]:
            st.warning(f'Текст содержит {result["token_count"]} токенов: использованы первые 96. Конец сообщения не анализируется.')
        if any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in result["text"]):
            st.warning("В тексте есть кириллица. Качество для смешанного языка не оценивалось.")
        st.download_button("Скачать результат · JSON", json.dumps(result, ensure_ascii=False, indent=2),
                           file_name="sms_analysis.json", mime="application/json")

st.divider()
st.caption("Порог spam: 0,5. Вероятность модели не является гарантией: возможны ошибки. "
           "Метрики получены на учебном SMS Spam Collection; выбор по тому же test ограничивает их независимость.")
