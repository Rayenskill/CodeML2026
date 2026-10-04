"""Verify new structural models, single-applicant inference, and scored files."""
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model import ROOT, digest, write_predictions

HERE = Path(__file__).resolve().parent


def main():
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    out = HERE / "structural"
    submissions = json.loads((out / "submissions.json").read_text())
    verified = []
    with tempfile.TemporaryDirectory() as temp:
        for row in submissions:
            model = joblib.load(ROOT / row["artifact"])
            destination = Path(temp) / f'{row["model"]}.csv'
            regenerated = write_predictions(model, applicants, destination)
            assert regenerated["sha256"] == row["sha256"], row["model"]
            assert destination.read_bytes() == Path(row["file"]).read_bytes()
            data = pd.read_csv(destination)
            assert list(data.columns) == ["id_candidat", "decision_octroi"]
            assert len(data) == 4000 and data.id_candidat.is_unique
            np.testing.assert_array_equal(data.id_candidat.to_numpy(), applicants.id_candidat.to_numpy())
            assert data.decision_octroi.isin([0, 1]).all() and data.decision_octroi.sum() == 1600
            sample = applicants.iloc[:13].copy()
            scores = model.score(sample)
            sample["id_candidat"] = [f"new_applicant_{i}" for i in range(len(sample))]
            sample["decision_octroi"] = np.arange(len(sample)) % 2
            np.testing.assert_array_equal(scores, model.score(sample))
            np.testing.assert_allclose(model.score(sample.iloc[[7]]), scores[[7]], rtol=0, atol=1e-13)
            np.testing.assert_allclose(model.score(sample.iloc[::-1])[::-1], scores, rtol=0, atol=1e-13)
            assert {"id_candidat", "decision_octroi"}.isdisjoint(model.profiles_.columns)
            # Every family must respond to a previously unseen academic profile.
            different = sample.copy()
            different["cote_r_equivalent"] = np.clip(different.cote_r_equivalent + 2, 15, 40)
            assert np.any(np.abs(model.score(different) - scores) > 1e-5)
            if row["model"].startswith("region_correction_"):
                # The new correction must actually use region, with equal merit.
                probe = pd.concat([sample.iloc[[0]]] * 2, ignore_index=True)
                probe["region_administrative"] = ["Montreal", "Cote-Nord"]
                effect = np.diff(model.score(probe))[0]
                assert abs(effect) > 1e-5
            verified.append({"model": row["model"], "rows": 4000, "grants": 1600,
                             "prediction_sha256": regenerated["sha256"],
                             "model_sha256": digest(ROOT / row["artifact"]),
                             "reproduces_exported_predictions": True,
                             "ignores_identifiers_and_supplied_labels": True,
                             "single_applicant_and_order_invariance": True,
                             "responds_to_new_academic_profiles": True})
            print(f'{row["model"]}: passed', flush=True)
    board = pd.read_csv(HERE / "platform_results.csv")
    for row in board.itertuples():
        assert digest(ROOT / row.file) == row.sha256, row.file
    best = json.loads((HERE / "best_model.json").read_text())
    assert digest(ROOT / best["model_file"]) == best["model_sha256"]
    assert digest(ROOT / "upload_model_best/predictions.csv") == best["scored_prediction_sha256"]
    evidence = {"models": verified, "previously_scored_files_unchanged": True,
                "best_verified_accuracy_percent": best["observed_accuracy_percent"],
                "new_models_reference_accuracy_verified": False,
                "source_sha256": {p.name: digest(p) for p in
                    [HERE / "model.py", HERE / "structural.py", HERE / "verify_structural.py"]}}
    (out / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("All structural models passed; verified best and scored files preserved.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
