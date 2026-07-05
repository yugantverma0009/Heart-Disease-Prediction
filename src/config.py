from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

DATASET_SLUG = "ronitf/heart-disease-uci"
DATA_PATH = DATA_DIR / "heart.csv"
MODEL_PATH = MODEL_DIR / "best_heart_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessing_pipeline.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
DATA_SUMMARY_PATH = REPORT_DIR / "data_summary.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrices.png"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"
CLASS_LABELS = ["No Disease", "Disease"]
RAW_TARGET_MEANING = {
    0: "Disease",
    1: "No Disease",
}
TRAINING_TARGET_MEANING = {
    0: "No Disease",
    1: "Disease",
}

FEATURE_HELP = {
    "age": "Age in years",
    "sex": "Sex: 1 = male, 0 = female",
    "cp": "Chest pain type: 0-3",
    "trestbps": "Resting blood pressure in mm Hg",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false",
    "restecg": "Resting ECG result: 0-2",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina: 1 = yes, 0 = no",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment: 0-2",
    "ca": "Number of major vessels colored by fluoroscopy: 0-4",
    "thal": "Thalassemia result: usually 1 = fixed defect, 2 = normal, 3 = reversible defect",
}

NUMERIC_INPUTS = {
    "age": {"label": "Age", "min": 18, "max": 100, "default": 54, "step": 1},
    "trestbps": {"label": "Resting blood pressure", "min": 94, "max": 200, "default": 130, "step": 1},
    "chol": {"label": "Cholesterol", "min": 126, "max": 564, "default": 245, "step": 1},
    "thalach": {"label": "Maximum heart rate", "min": 71, "max": 202, "default": 150, "step": 1},
    "oldpeak": {"label": "ST depression", "min": 0.0, "max": 6.2, "default": 1.0, "step": 0.1},
    "ca": {"label": "Major vessels colored", "min": 0, "max": 4, "default": 0, "step": 1},
}

CATEGORICAL_INPUTS = {
    "sex": {"label": "Sex", "options": {"Female": 0, "Male": 1}, "default": "Female"},
    "cp": {
        "label": "Chest pain type",
        "options": {
            "Asymptomatic": 0,
            "Atypical angina": 1,
            "Non-anginal pain": 2,
            "Typical angina": 3,
        },
        "default": "Atypical angina",
    },
    "fbs": {"label": "Fasting blood sugar > 120 mg/dl", "options": {"No": 0, "Yes": 1}, "default": "No"},
    "restecg": {
        "label": "Resting ECG",
        "options": {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2,
        },
        "default": "Normal",
    },
    "exang": {"label": "Exercise-induced angina", "options": {"No": 0, "Yes": 1}, "default": "No"},
    "slope": {
        "label": "Peak exercise ST slope",
        "options": {
            "Downsloping": 0,
            "Flat": 1,
            "Upsloping": 2,
        },
        "default": "Upsloping",
    },
    "thal": {
        "label": "Thalassemia",
        "options": {
            "Unknown": 0,
            "Fixed defect": 1,
            "Normal": 2,
            "Reversible defect": 3,
        },
        "default": "Normal",
    },
}

RISK_BANDS = [
    (0.30, "Low", "The model sees a lower pattern of risk for these inputs."),
    (0.70, "Moderate", "The model sees some risk signals. Review the inputs carefully."),
    (1.01, "High", "The model sees a stronger risk pattern for these inputs."),
]


def get_risk_band(probability: float | None) -> tuple[str, str]:
    if probability is None:
        return "Result", "Probability is not available for this model."

    for limit, label, message in RISK_BANDS:
        if probability < limit:
            return label, message

    return "Result", "Prediction completed."
