"""
Train the model once and persist all artifacts the API needs at serve time.

Saves the calibrated model and the fitted feature metadata to disk so the API
loads a ready model at startup instead of retraining on every boot. Run this
as a build/deploy step:  python scripts/train_and_save.py
"""

import pickle
from pathlib import Path

try:
    from src.model import build_split, train_model
    from src.calibration import calibrate
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model import build_split, train_model
    from src.calibration import calibrate

ARTIFACT_DIR = Path("artifacts")


def main(sample_path: str = "data/sample.csv"):
    ARTIFACT_DIR.mkdir(exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_split(sample_path)
    model = train_model(X_train, y_train, X_val, y_val)
    calibrated = calibrate(model, X_val, y_val)

    # The raw XGBoost model is needed for SHAP; the calibrated wrapper is
    # needed for well-calibrated PDs. Persist both, plus the feature schema.
    with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ARTIFACT_DIR / "calibrated.pkl", "wb") as f:
        pickle.dump(calibrated, f)
    with open(ARTIFACT_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(list(X_train.columns), f)

    print(f"Saved artifacts to {ARTIFACT_DIR}/")
    print(f"  model.pkl, calibrated.pkl, feature_columns.pkl")
    print(f"Feature count: {len(X_train.columns)}")


if __name__ == "__main__":
    main()
