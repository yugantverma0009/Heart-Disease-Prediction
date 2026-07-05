import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CATEGORICAL_INPUTS, FEATURE_COLUMNS, MODEL_PATH, PREPROCESSOR_PATH, get_risk_band


@dataclass(frozen=True)
class TestCase:
    name: str
    inputs: dict[str, float | int]
    expected_risk: str
    expected_probability: tuple[float, float]


TEST_CASES = [
    TestCase(
        "TEST CASE 1 - VERY HEALTHY",
        {
            "age": 25,
            "sex": 0,
            "cp": 2,
            "trestbps": 110,
            "chol": 170,
            "fbs": 0,
            "restecg": 0,
            "thalach": 195,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 2,
            "ca": 0,
            "thal": 2,
        },
        "Low",
        (0.05, 0.20),
    ),
    TestCase(
        "TEST CASE 2 - HEALTHY ADULT",
        {
            "age": 35,
            "sex": 1,
            "cp": 2,
            "trestbps": 118,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 185,
            "exang": 0,
            "oldpeak": 0.2,
            "slope": 2,
            "ca": 0,
            "thal": 2,
        },
        "Low",
        (0.10, 0.30),
    ),
    TestCase(
        "TEST CASE 3 - MODERATE RISK",
        {
            "age": 50,
            "sex": 1,
            "cp": 1,
            "trestbps": 140,
            "chol": 240,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.5,
            "slope": 1,
            "ca": 1,
            "thal": 1,
        },
        "Moderate",
        (0.40, 0.65),
    ),
    TestCase(
        "TEST CASE 4 - HIGH RISK",
        {
            "age": 65,
            "sex": 1,
            "cp": 3,
            "trestbps": 170,
            "chol": 340,
            "fbs": 1,
            "restecg": 2,
            "thalach": 105,
            "exang": 1,
            "oldpeak": 3.8,
            "slope": 0,
            "ca": 3,
            "thal": 3,
        },
        "High",
        (0.80, 0.95),
    ),
    TestCase(
        "TEST CASE 5 - VERY HIGH RISK",
        {
            "age": 75,
            "sex": 1,
            "cp": 3,
            "trestbps": 190,
            "chol": 450,
            "fbs": 1,
            "restecg": 2,
            "thalach": 90,
            "exang": 1,
            "oldpeak": 5.5,
            "slope": 0,
            "ca": 4,
            "thal": 3,
        },
        "High",
        (0.90, 0.99),
    ),
    TestCase(
        "TEST CASE 6 - YOUNG WITH SOME RISK",
        {
            "age": 28,
            "sex": 1,
            "cp": 3,
            "trestbps": 150,
            "chol": 260,
            "fbs": 0,
            "restecg": 0,
            "thalach": 180,
            "exang": 0,
            "oldpeak": 0.5,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        },
        "Moderate",
        (0.30, 0.50),
    ),
    TestCase(
        "TEST CASE 7 - ELDERLY BUT HEALTHY",
        {
            "age": 72,
            "sex": 0,
            "cp": 2,
            "trestbps": 120,
            "chol": 190,
            "fbs": 0,
            "restecg": 0,
            "thalach": 170,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 2,
            "ca": 0,
            "thal": 2,
        },
        "Low to Moderate",
        (0.20, 0.45),
    ),
]


BASELINE = TEST_CASES[0].inputs
SENSITIVITY_TESTS = {
    "Age": ("age", [25, 45, 65, 80], "gradually increase"),
    "Cholesterol": ("chol", [170, 220, 300, 450], "increase slightly"),
    "Major Vessels (CA)": ("ca", [0, 1, 2, 3, 4], "significant increase"),
    "Exercise Angina": ("exang", [0, 1], "noticeable increase"),
    "Maximum Heart Rate": ("thalach", [190, 170, 140, 100], "increase as heart rate decreases"),
    "Oldpeak": ("oldpeak", [0, 1, 2, 3, 5], "steadily increase"),
}


def make_frame(values: dict[str, float | int]) -> pd.DataFrame:
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def disease_probability(pipeline, values: dict[str, float | int]) -> float:
    return float(pipeline.predict_proba(make_frame(values))[0][1])


def risk_matches(actual: str, expected: str) -> bool:
    if expected == "Low to Moderate":
        return actual in {"Low", "Moderate"}
    return actual == expected


def format_table(rows: list[list[str]]) -> str:
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        line = "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"
        lines.append(line)
        if row_index == 0:
            lines.append("| " + " | ".join("-" * widths[index] for index in range(len(row))) + " |")
    return "\n".join(lines)


def verify_artifacts(artifact: dict, saved_preprocessor) -> list[tuple[str, bool, str]]:
    pipeline = artifact["pipeline"]
    checks = [
        (
            "Feature order is identical to training dataset",
            list(artifact.get("feature_columns", [])) == FEATURE_COLUMNS,
            str(artifact.get("feature_columns", [])),
        ),
        (
            "Pipeline preprocessing feature order matches config",
            list(pipeline.named_steps["preprocess"].transformers[0][2]) == FEATURE_COLUMNS,
            str(list(pipeline.named_steps["preprocess"].transformers[0][2])),
        ),
        (
            "Saved preprocessing artifact matches pipeline",
            list(saved_preprocessor.transformers[0][2]) == list(pipeline.named_steps["preprocess"].transformers[0][2]),
            str(list(saved_preprocessor.transformers[0][2])),
        ),
        (
            "Positive class probability uses predict_proba(X)[0][1]",
            artifact.get("positive_class") == 1 and artifact.get("positive_class_label") == "Disease",
            str({key: artifact.get(key) for key in ["positive_class", "positive_class_label"]}),
        ),
        (
            "Random Forest is the saved production estimator",
            artifact.get("model_name") == "Random Forest"
            and pipeline.named_steps["model"].__class__.__name__ == "ClinicallyCalibratedRandomForest"
            and pipeline.named_steps["model"].estimator.__class__.__name__ == "RandomForestClassifier",
            f"{artifact.get('model_name')} / {pipeline.named_steps['model'].__class__.__name__}",
        ),
        (
            "Categorical encoding matches UCI training codes",
            CATEGORICAL_INPUTS["cp"]["options"]["Asymptomatic"] == 0
            and CATEGORICAL_INPUTS["cp"]["options"]["Typical angina"] == 3
            and CATEGORICAL_INPUTS["slope"]["options"]["Downsloping"] == 0
            and CATEGORICAL_INPUTS["slope"]["options"]["Upsloping"] == 2
            and CATEGORICAL_INPUTS["thal"]["options"]["Normal"] == 2
            and CATEGORICAL_INPUTS["thal"]["options"]["Reversible defect"] == 3,
            str({key: CATEGORICAL_INPUTS[key]["options"] for key in ["cp", "slope", "thal"]}),
        ),
    ]
    return checks


def run_case_table(pipeline) -> tuple[list[list[str]], bool]:
    rows = [["Test Case", "Predicted Probability", "Predicted Risk", "Expected Result", "PASS/FAIL", "Reason if Failed"]]
    all_passed = True
    for case in TEST_CASES:
        probability = disease_probability(pipeline, case.inputs)
        risk, _ = get_risk_band(probability)
        low, high = case.expected_probability
        risk_ok = risk_matches(risk, case.expected_risk)
        probability_ok = low <= probability <= high
        passed = risk_ok and probability_ok
        all_passed = all_passed and passed
        reason = ""
        if not risk_ok:
            reason = f"Expected risk {case.expected_risk}, got {risk}."
        if not probability_ok:
            range_text = f"{low:.0%}-{high:.0%}"
            reason = f"{reason} Expected probability {range_text}, got {probability:.1%}.".strip()
        rows.append(
            [
                case.name,
                f"{probability:.2%}",
                risk,
                f"{case.expected_risk}, {low:.0%}-{high:.0%}",
                "PASS" if passed else "FAIL",
                reason or "-",
            ]
        )
    return rows, all_passed


def run_sensitivity_table(pipeline) -> tuple[list[list[str]], bool]:
    rows = [["Sensitivity Test", "Values", "Probabilities", "Expected", "PASS/FAIL", "Reason if Failed"]]
    all_passed = True
    for label, (feature, values, expected) in SENSITIVITY_TESTS.items():
        probabilities = []
        for value in values:
            changed = dict(BASELINE)
            changed[feature] = value
            probabilities.append(disease_probability(pipeline, changed))

        if feature == "thalach":
            passed = all(left <= right for left, right in zip(probabilities, probabilities[1:]))
        elif feature == "chol":
            passed = probabilities[-1] >= probabilities[0] and max(abs(right - left) for left, right in zip(probabilities, probabilities[1:])) <= 0.20
        else:
            passed = all(left <= right for left, right in zip(probabilities, probabilities[1:]))

        if feature == "ca":
            passed = passed and (probabilities[-1] - probabilities[0]) >= 0.10
        if feature == "exang":
            passed = passed and (probabilities[-1] - probabilities[0]) >= 0.03

        all_passed = all_passed and passed
        rows.append(
            [
                label,
                " -> ".join(str(value) for value in values),
                " -> ".join(f"{probability:.2%}" for probability in probabilities),
                expected,
                "PASS" if passed else "FAIL",
                "-" if passed else "Observed probability trend does not match expectation.",
            ]
        )
    return rows, all_passed


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessing artifact: {PREPROCESSOR_PATH}")

    artifact = joblib.load(MODEL_PATH)
    saved_preprocessor = joblib.load(PREPROCESSOR_PATH)
    pipeline = artifact["pipeline"]

    print(f"Model: {artifact.get('model_name')}")
    print(f"Estimator: {pipeline.named_steps['model'].__class__.__name__}")
    print()

    artifact_rows = [["Verification", "PASS/FAIL", "Evidence"]]
    artifact_passed = True
    for label, passed, evidence in verify_artifacts(artifact, saved_preprocessor):
        artifact_passed = artifact_passed and passed
        artifact_rows.append([label, "PASS" if passed else "FAIL", evidence])
    print(format_table(artifact_rows))
    print()

    case_rows, cases_passed = run_case_table(pipeline)
    print(format_table(case_rows))
    print()

    sensitivity_rows, sensitivity_passed = run_sensitivity_table(pipeline)
    print(format_table(sensitivity_rows))
    print()

    overall_passed = artifact_passed and cases_passed and sensitivity_passed
    print(f"Overall QA Result: {'PASS' if overall_passed else 'FAIL'}")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
