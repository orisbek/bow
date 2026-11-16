"""Run five new SMS through the actual Streamlit form and save the results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.app_examples import CONTROL_TEXTS
from src.app_inference import SpamDetector
from src.data import load_sms_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    corpus = set(load_sms_data().message.str.strip().str.casefold())
    detector = SpamDetector()
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    if app.exception:
        raise RuntimeError(app.exception[0].message)
    rows = []
    for item in CONTROL_TEXTS:
        if item["text"].strip().casefold() in corpus:
            raise ValueError(f'Control {item["id"]} already exists in the dataset')
        option = f'{item["id"]} · {item["title"]}'
        app.selectbox(key="example").select(option).run()
        if app.text_area(key="sms").value != item["text"]:
            raise AssertionError("Example did not populate the input")
        app.button[0].click().run()
        if app.exception:
            raise RuntimeError(app.exception[0].message)
        ui = dict(app.session_state["result"])
        direct = detector.analyze(item["text"])
        if ui != direct:
            raise AssertionError("UI result differs from offline inference")
        rows.append({**item, **ui, "correct": ui["label"] == item["expected"],
                     "new_to_dataset": True,
                     "screenshot": f'outputs/images/week11_app/test_{item["id"]}.jpg'})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    reports = ROOT / "reports"
    pd.DataFrame(rows).to_csv(reports / "week11_control_tests.csv", index=False)
    (reports / "week11_control_tests.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'Correct: {sum(row["correct"] for row in rows)}/5. Screenshots must be captured in the browser.', flush=True)


if __name__ == "__main__":
    main()
