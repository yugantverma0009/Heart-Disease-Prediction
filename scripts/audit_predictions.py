import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import FEATURE_COLUMNS, MODEL_PATH, PREPROCESSOR_PATH, get_risk_band


HEALTHY_PATIENT = {
    "age": 30,
    "sex": 0,
    "cp": 1,
    "trestbps": 110,
    "chol": 170,
    "fbs": 0,
    "restecg": 0,
    "thalach": 190,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}

HIGH_RISK_PATIENT = {
    "age": 72,
    "sex": 1,
    "cp": 3,
    "trestbps": 190,
    "chol": 420,
    "fbs": 1,
    "restecg": 2,
    "thalach": 90,
    "exang": 1,
    "oldpeak": 5.0,
    "slope": 0,
    "ca": 4,
    "thal": 3,
}

ONE_FEATURE_CHANGES = {
    "age": 72,
    "sex": 1,
    "cp": 0,
    "trestbps": 190,
    "chol": 420,
    "fbs": 1,
    "restecg": 2,
    "thalach": 90,
    "exang": 1,
    "oldpeak": 5.0,
    "slope": 0,
    "ca": 4,
    "thal": 3,
}


def make_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def disease_probability(pipeline, values: dict) -> float:
    return float(pipeline.predict_proba(make_frame(values))[0][1])


def assert_risk_case(pipeline, label: str, values: dict, expected_band: str) -> None:
    probability = disease_probability(pipeline, values)
    band, _ = get_risk_band(probability)
    prediction = int(pipeline.predict(make_frame(values))[0])
    print(f"{label}: prediction={prediction}, probability={probability:.3%}, band={band}")
    if band != expected_band:
        raise AssertionError(f"{label} expected {expected_band} Risk, got {band} Risk.")


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessing artifact: {PREPROCESSOR_PATH}")

    artifact = joblib.load(MODEL_PATH)
    saved_preprocessor = joblib.load(PREPROCESSOR_PATH)
    pipeline = artifact["pipeline"]

    if list(artifact["feature_columns"]) != FEATURE_COLUMNS:
        raise AssertionError("Artifact feature order does not match config FEATURE_COLUMNS.")
    if list(pipeline.named_steps["preprocess"].transformers[0][2]) != FEATURE_COLUMNS:
        raise AssertionError("Pipeline preprocessor feature order does not match training config.")
    if list(saved_preprocessor.transformers[0][2]) != FEATURE_COLUMNS:
        raise AssertionError("Saved preprocessor feature order does not match training config.")
    if artifact.get("positive_class") != 1 or artifact.get("positive_class_label") != "Disease":
        raise AssertionError("Class 1 must be Disease for predict_proba(X)[0][1].")

    assert_risk_case(pipeline, "Healthy patient", HEALTHY_PATIENT, "Low")
    assert_risk_case(pipeline, "High-risk patient", HIGH_RISK_PATIENT, "High")

    base_probability = disease_probability(pipeline, HEALTHY_PATIENT)
    unchanged = []
    for feature, changed_value in ONE_FEATURE_CHANGES.items():
        changed = HEALTHY_PATIENT.copy()
        changed[feature] = changed_value
        new_probability = disease_probability(pipeline, changed)
        if abs(new_probability - base_probability) <= 1e-9:
            unchanged.append(feature)
        print(f"Change {feature}: {base_probability:.3%} -> {new_probability:.3%}")

    if unchanged:
        raise AssertionError(f"Changing these features did not affect probability: {unchanged}")

    print("Prediction audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
