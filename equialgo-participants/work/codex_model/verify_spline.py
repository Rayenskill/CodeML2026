"""Check round-three artifacts and preserve all previously scored predictions."""
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
    folder = HERE / "spline_tuning"
    submissions = json.loads((folder / "submissions.json").read_text())
    results = []
    with tempfile.TemporaryDirectory() as temp:
        for row in submissions:
            if "file" not in row:
                continue
            model = joblib.load(ROOT / row["artifact"])
            output = Path(temp) / f'{row["model"]}.csv'
            regenerated = write_predictions(model, applicants, output)
            assert regenerated["sha256"] == row["sha256"], row["model"]
            assert output.read_bytes() == Path(row["file"]).read_bytes()
            data = pd.read_csv(output)
            assert list(data.columns) == ["id_candidat", "decision_octroi"]
            assert len(data) == 4000 and data.id_candidat.is_unique
            np.testing.assert_array_equal(data.id_candidat, applicants.id_candidat)
            assert data.decision_octroi.isin([0, 1]).all()
            assert 0.36 <= data.decision_octroi.mean() <= 0.44
            sample = applicants.iloc[:9].copy()
            scores = model.score(sample)
            sample["id_candidat"] = [f"brand_new_{i}" for i in range(len(sample))]
            sample["decision_octroi"] = np.arange(len(sample)) % 2
            np.testing.assert_array_equal(scores, model.score(sample))
            np.testing.assert_allclose(model.score(sample.iloc[[4]]), scores[[4]], atol=1e-13, rtol=0)
            if row["allocation_rule"] == "training_threshold":
                np.testing.assert_array_equal(model.predict(applicants), data.decision_octroi)
                np.testing.assert_array_equal(model.predict(sample.iloc[[4]]), model.predict(sample)[[4]])
            results.append({"model": row["model"], "sha256": regenerated["sha256"],
                            "rows": len(data), "grants": int(data.decision_octroi.sum()),
                            "reproduced_byte_for_byte": True,
                            "ignores_identifiers_and_supplied_labels": True,
                            "scores_single_new_applicant": True})
            print(f'{row["model"]}: passed', flush=True)
    board = pd.read_csv(HERE / "platform_results.csv")
    for row in board.itertuples():
        assert digest(ROOT / row.file) == row.sha256, row.file
    best = json.loads((HERE / "best_model.json").read_text())
    assert digest(ROOT / best["model_file"]) == best["model_sha256"]
    assert digest(ROOT / "upload_model_best/predictions.csv") == best["scored_prediction_sha256"]
    summary = {"models_checked": results, "previously_scored_files_unchanged": True,
               "default_prediction_reproduces_verified_best": True,
               "best_observed_accuracy_percent": best["observed_accuracy_percent"],
               "new_models_reference_accuracy_verified": False,
               "source_sha256": {p.name: digest(p) for p in
                   [HERE / "model.py", HERE / "tune_spline.py", HERE / "verify_spline.py"]}}
    (folder / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("All new models verified; all previously scored files preserved.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
