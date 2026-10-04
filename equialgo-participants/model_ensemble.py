"""Train or run the historical-data-only EquiAlgo model and ensemble.

    python model_ensemble.py train
    python model_ensemble.py predict
    python model_ensemble.py refine
    python model_ensemble.py tune-spline
    python model_ensemble.py structural

Training selects models on five-fold development validation and reports a
reserved historical holdout. Prediction loads the best verified clean model
when available and writes upload_model_best/predictions.csv.
"""
from pathlib import Path
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "predict", "refine", "tune-spline", "structural"}:
        print(__doc__)
        return
    command = sys.argv.pop(1)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "work/codex_model"))
    if command == "train":
        from train import main as run
    elif command == "refine":
        from refine import main as run
    elif command == "tune-spline":
        from tune_spline import main as run
    elif command == "structural":
        from structural import main as run
    else:
        from model import main as run
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        run()


if __name__ == "__main__":
    main()
