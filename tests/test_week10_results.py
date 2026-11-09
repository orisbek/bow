"""Check that the published Week 10 tables agree with individual predictions."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


class Week10ResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = pd.read_csv(REPORTS / "week10_experiments.csv")
        cls.predictions = pd.read_csv(REPORTS / "week10_predictions.csv")
        cls.split = pd.read_csv(REPORTS / "week10_split.csv")

    def test_metrics_match_predictions(self):
        self.assertEqual(len(self.results), 6)
        test_rows = set(self.split.loc[self.split.split == "test", "source_row"])
        for result in self.results.itertuples():
            part = self.predictions[self.predictions.experiment == result.experiment]
            self.assertEqual(len(part), result.test_size)
            self.assertEqual(set(part.source_row), test_rows)
            self.assertFalse(part.source_row.duplicated().any())
            true = part.true_label.eq("spam")
            pred = part.probability_spam.ge(.5)
            self.assertTrue(np.array_equal(pred, part.predicted_label.eq("spam")))
            for name, metric in (("accuracy", accuracy_score),
                                 ("precision_spam", precision_score),
                                 ("recall_spam", recall_score),
                                 ("f1_spam", f1_score)):
                self.assertAlmostEqual(metric(true, pred), getattr(result, name), places=12)
                clean = part.no_exact_match_in_full_train
                self.assertAlmostEqual(metric(true[clean], pred[clean]),
                                       getattr(result, "clean_" + name), places=12)
            self.assertEqual(int((~pred & true).sum()), result.fn)
            self.assertEqual(int((pred & ~true).sum()), result.fp)

    def test_subsets_and_reference_errors(self):
        split = self.split
        train = split.split.eq("train")
        self.assertEqual(int(split.train_50pct.sum()), 2228)
        self.assertEqual(int(split.train_75pct.sum()), 3342)
        self.assertEqual(int(split.train_100pct.sum()), 4457)
        self.assertTrue((split.train_50pct <= split.train_75pct).all())
        self.assertTrue((split.train_75pct <= split.train_100pct).all())
        self.assertTrue((split.train_100pct == train).all())
        errors = pd.read_csv(REPORTS / "week10_errors.csv")
        baseline = self.predictions[self.predictions.experiment == "week9_saved_reference"]
        wrong = baseline[baseline.true_label != baseline.predicted_label]
        self.assertEqual(len(errors), 15)
        self.assertEqual(set(errors.source_row), set(wrong.source_row))
        self.assertTrue(errors.category_hypothesis.notna().all())
        self.assertTrue(errors.proposed_solution.notna().all())


if __name__ == "__main__":
    unittest.main()
