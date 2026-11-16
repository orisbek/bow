"""Integration checks for offline inference and the real Streamlit form."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from src.app_examples import CONTROL_TEXTS
from src.app_inference import MODEL_DIR, SpamDetector, validate_text

ROOT = Path(__file__).resolve().parents[1]


class InputTest(unittest.TestCase):
    def test_invalid_inputs(self):
        for text in ("", "  \n", "12345!!!", "Привет, как дела?", "a" * 4001):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                validate_text(text)
        self.assertEqual(validate_text("  Hello there!  "), "Hello there!")


@unittest.skipUnless((MODEL_DIR / "metadata.json").exists(), "Run python -m scripts.prepare_app_model")
class ApplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = SpamDetector()

    def test_offline_probabilities_and_truncation(self):
        for item in CONTROL_TEXTS:
            result = self.detector.analyze(item["text"])
            self.assertGreaterEqual(result["probability_spam"], 0)
            self.assertLessEqual(result["probability_spam"], 1)
            self.assertAlmostEqual(result["probability_ham"] + result["probability_spam"], 1)
            self.assertEqual(result["label"], "spam" if result["probability_spam"] >= .5 else "ham")
            self.assertEqual(result["confidence"], result["probability_" + result["label"]])
            self.assertFalse(result["truncated"])
        self.assertTrue(self.detector.analyze("hello " * 200)["truncated"])

    def test_ui_controls_and_invalid_input_clear_result(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        for item in CONTROL_TEXTS:
            app.selectbox(key="example").select(f'{item["id"]} · {item["title"]}').run()
            self.assertFalse(app.exception)
            self.assertNotIn("result", app.session_state)
            self.assertEqual(app.text_area(key="sms").value, item["text"])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(dict(app.session_state["result"]), self.detector.analyze(item["text"]))
        app.text_area(key="sms").set_value("   ")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("result", app.session_state)
        self.assertTrue(any("Введите текст" in message.value for message in app.error))
        app.text_area(key="sms").set_value("hello " * 200)
        app.button[0].click().run()
        self.assertTrue(any("первые 96" in message.value for message in app.warning))


if __name__ == "__main__":
    unittest.main()
