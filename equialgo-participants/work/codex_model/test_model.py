"""Behavior checks for clean inference, persistence, and the validation split."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model import ROOT, features, write_predictions

HERE = Path(__file__).resolve().parent


class ModelBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=2)
        cls.model = joblib.load(HERE / "artifacts/model.joblib")
        cls.apps = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
        cls.sample = cls.apps.iloc[:17].copy()

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_identifiers_and_supplied_labels_do_not_affect_scores(self):
        modified = self.sample.copy()
        modified["id_candidat"] = [f"never_seen_{i}" for i in range(len(modified))]
        modified["decision_octroi"] = np.arange(len(modified)) % 2
        np.testing.assert_array_equal(self.model.score(modified), self.model.score(self.sample))

    def test_score_is_independent_of_cohort_and_order(self):
        expected = self.model.score(self.sample)
        reverse = self.sample.iloc[::-1]
        np.testing.assert_allclose(self.model.score(reverse)[::-1], expected, atol=1e-13, rtol=0)
        np.testing.assert_allclose(self.model.score(self.sample.iloc[[8]]), expected[[8]], atol=1e-13, rtol=0)

    def test_counterfactual_scoring_neutralizes_nuisance_features(self):
        altered = self.sample.copy()
        altered["region_administrative"] = "Montreal"
        altered["revenu_familial_estime"] = 150000
        altered["distance_domicile_campus_km"] = 2.0
        altered["code_postal_3"] = "NEW"
        altered["programme_etudes"] = "New programme"
        altered["premiere_generation_universitaire"] = 1
        np.testing.assert_array_equal(self.model.score(altered), self.model.score(self.sample))

    def test_scores_respond_to_new_academic_profiles(self):
        applicants = pd.concat([self.sample.iloc[[0]]] * 3, ignore_index=True)
        applicants["id_candidat"] = ["new_low", "new_mid", "new_high"]
        applicants["cote_r_equivalent"] = [20.0, 28.0, 36.0]
        scores = self.model.score(applicants)
        self.assertTrue(np.all(np.diff(scores) > 0), scores)

    def test_serialization_and_export_reproduce_predictions(self):
        expected = pd.read_csv(ROOT / "upload_model/predictions.csv")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            joblib.dump(self.model, directory / "model.joblib")
            restored = joblib.load(directory / "model.joblib")
            report = write_predictions(restored, self.apps, directory / "predictions.csv")
            actual = pd.read_csv(directory / "predictions.csv")
            pd.testing.assert_frame_equal(actual, expected)
            self.assertEqual(report["rows"], 4000)
            self.assertEqual(report["grants"], 1600)
            self.assertTrue(actual.decision_octroi.isin([0, 1]).all())

    def test_holdout_and_eval_excluded_from_development_split(self):
        split = pd.read_csv(HERE / "artifacts/validation_split.csv")
        train_ids = set(split.loc[split.split == "development", "id_candidat"])
        holdout_ids = set(split.loc[split.split == "holdout", "id_candidat"])
        self.assertEqual(len(train_ids), 8000)
        self.assertEqual(len(holdout_ids), 2000)
        self.assertFalse(train_ids & holdout_ids)
        self.assertFalse((train_ids | holdout_ids) & set(self.apps.id_candidat))
        self.assertTrue((split.loc[split.split == "holdout", "cv_fold"] == -1).all())
        self.assertEqual(set(split.loc[split.split == "development", "cv_fold"]), set(range(5)))
        report = json.loads((HERE / "artifacts/training_report.json").read_text())
        self.assertIsNone(report["reference_accuracy"])

    def test_invalid_input_is_rejected(self):
        bad = self.sample.copy()
        bad.loc[bad.index[0], "revenu_familial_estime"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            features(bad)
        with self.assertRaisesRegex(ValueError, "Missing application fields"):
            features(self.sample.drop(columns=["cote_r_equivalent"]))
        repeated = self.sample.copy()
        repeated["id_candidat"] = "duplicate"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                write_predictions(self.model, repeated, Path(directory) / "out.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
