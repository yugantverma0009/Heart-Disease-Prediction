import json
import sys
import traceback
from pathlib import Path
from textwrap import dedent

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_INPUTS,
    CLASS_LABELS,
    CONFUSION_MATRIX_PATH,
    FEATURE_COLUMNS,
    get_risk_band,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_INPUTS,
    PREPROCESSOR_PATH,
    TRAINING_TARGET_MEANING,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f5f7fb;
            --panel: #ffffff;
            --panel-soft: #f8fafc;
            --sidebar: #eef3f8;
            --border: #d8e0ea;
            --text: #111827;
            --muted: #5f6f85;
            --input-bg: #ffffff;
            --input-text: #111827;
            --input-muted: #64748b;
            --accent: #2563eb;
            --accent-soft: rgba(37, 99, 235, 0.09);
            --success: #0f9f6e;
            --warning: #b7791f;
            --danger: #be123c;
            --button-hover: #1d4ed8;
            --shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
            --card-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(37, 99, 235, 0.05), transparent 18rem),
                var(--app-bg) !important;
            color: var(--text) !important;
        }
        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(180deg, rgba(37, 99, 235, 0.05), transparent 18rem),
                var(--app-bg) !important;
        }
        [data-testid="stHeader"] {
            background: rgba(245, 247, 251, 0.86) !important;
            backdrop-filter: blur(10px);
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        section[data-testid="stSidebar"] {
            background: var(--sidebar) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text) !important;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--text) !important;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        label {
            color: var(--text);
        }
        .hero-panel {
            border: 1px solid var(--border);
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.11), rgba(20, 184, 166, 0.08)),
                var(--panel);
            border-radius: 8px;
            padding: 1.45rem 1.55rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
        }
        .hero-panel h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.05;
            color: var(--text) !important;
        }
        .hero-panel p {
            color: var(--muted) !important;
            margin: 0.7rem 0 0;
            max-width: 760px;
            font-size: 1.02rem;
            line-height: 1.65;
        }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.9rem 0 1rem;
        }
        .stat-card {
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 8px;
            padding: 1rem;
            min-height: 112px;
            box-shadow: var(--card-shadow);
        }
        .stat-card .label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.45rem;
        }
        .stat-card .value {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .stat-card .detail {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.45rem;
        }
        .sidebar-card {
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--card-shadow);
        }
        .sidebar-card .label {
            color: var(--muted);
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }
        .sidebar-card .value {
            color: var(--text);
            font-size: 1.15rem;
            font-weight: 700;
        }
        .risk-callout {
            border-radius: 8px;
            border: 1px solid rgba(190, 18, 60, 0.26);
            background: rgba(190, 18, 60, 0.08);
            padding: 0.95rem 1rem;
            margin: 0.9rem 0;
            color: #881337;
            font-weight: 600;
        }
        .risk-callout.low {
            border-color: rgba(15, 159, 110, 0.28);
            background: rgba(15, 159, 110, 0.08);
            color: #065f46;
        }
        .risk-callout.moderate {
            border-color: rgba(183, 121, 31, 0.3);
            background: rgba(183, 121, 31, 0.09);
            color: #7c4a03;
        }
        .section-note {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
        }
        div[data-testid="stTabs"] button p {
            font-weight: 650;
            color: var(--muted) !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: var(--accent) !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {
            background-color: var(--border) !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--accent) !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stHorizontalBlock"] {
            row-gap: 1rem;
        }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stSlider"] label,
        [data-testid="stSelectbox"] label {
            color: var(--muted) !important;
            opacity: 1 !important;
        }
        [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: var(--accent) !important;
            font-weight: 700;
        }
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            border-radius: 8px;
            border-color: var(--border) !important;
            background-color: var(--input-bg) !important;
            color: var(--input-text) !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] svg {
            color: var(--input-text) !important;
            fill: var(--input-text) !important;
        }
        div[data-baseweb="popover"] div[role="listbox"] {
            background: var(--panel);
            border: 1px solid var(--border);
        }
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] div[role="option"] {
            color: var(--input-text) !important;
            background: var(--panel) !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        .stButton {
            margin-top: 0.15rem;
        }
        .stButton > button {
            min-height: 2.85rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--panel);
            color: var(--text) !important;
            font-weight: 700;
            letter-spacing: 0;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }
        .stButton > button p,
        .stButton > button span {
            color: inherit !important;
            font-weight: 700;
        }
        .stButton > button[kind="primary"],
        .stButton button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            border-color: #1d4ed8 !important;
            color: #ffffff !important;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
        }
        .stButton > button[kind="primary"] *,
        .stButton button[data-testid="stBaseButton-primary"] *,
        button[data-testid="stBaseButton-primary"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(135deg, var(--button-hover) 0%, #1e40af 100%) !important;
            border-color: #1e40af !important;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.3);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stExpander"] summary p {
            color: var(--text) !important;
            font-weight: 650;
        }
        .risk-meter {
            --value: 0%;
            position: relative;
            height: 0.55rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.78);
            margin: 1.4rem 0 2rem;
            outline: none;
        }
        .risk-meter-fill {
            width: var(--value);
            height: 100%;
            border-radius: inherit;
            background: var(--accent);
        }
        .risk-meter-label {
            position: absolute;
            left: clamp(1.5rem, var(--value), calc(100% - 1.5rem));
            bottom: calc(100% + 0.45rem);
            transform: translateX(-50%);
            border-radius: 999px;
            background: var(--text);
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 700;
            line-height: 1;
            padding: 0.28rem 0.45rem;
            opacity: 0;
            pointer-events: none;
            transition: opacity 120ms ease;
            white-space: nowrap;
        }
        .risk-meter:hover .risk-meter-label,
        .risk-meter:focus .risk-meter-label,
        .risk-meter:active .risk-meter-label {
            opacity: 1;
        }
        @media (max-width: 900px) {
            .card-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_card(label: str, value: str, detail: str = "") -> str:
    return dedent(f"""
    <div class="stat-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="detail">{detail}</div>
    </div>
    """).strip()


def sidebar_card(label: str, value: str) -> None:
    st.sidebar.markdown(
        dedent(f"""
        <div class="sidebar-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def format_percent(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{decimals}%}"


def format_score(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def slider(column: str):
    settings = NUMERIC_INPUTS[column]
    return st.slider(
        settings["label"],
        min_value=settings["min"],
        max_value=settings["max"],
        value=settings["default"],
        step=settings["step"],
        help=f"Model feature: {column}",
    )


def selectbox(column: str) -> int:
    settings = CATEGORICAL_INPUTS[column]
    label = settings["label"]
    options = list(settings["options"])
    default_label = settings.get("default", options[0])
    selected = st.selectbox(
        label,
        options,
        index=options.index(default_label),
        help=f"Model feature: {column}",
    )
    return int(settings["options"][selected])


def risk_label(label: str) -> str:
    if label in {"Low", "Moderate", "High"}:
        return f"{label} Risk"
    return label


def risk_meter(probability: float) -> str:
    bounded_probability = min(max(probability, 0.0), 1.0)
    percent = bounded_probability * 100
    percent_text = f"{percent:.1f}%"
    return dedent(f"""
    <div class="risk-meter" style="--value: {percent:.1f}%;" role="meter" aria-label="Estimated heart disease probability" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent:.1f}" tabindex="0">
        <div class="risk-meter-fill"></div>
        <span class="risk-meter-label">{percent_text}</span>
    </div>
    """).strip()


def build_input() -> pd.DataFrame:
    st.subheader("Patient Details")
    st.markdown(
        '<div class="section-note">Enter the patient clinical details below to estimate the likelihood of heart disease using the trained machine learning model.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")

    with left:
        age = slider("age")
        sex = selectbox("sex")
        cp = selectbox("cp")
        trestbps = slider("trestbps")
        chol = slider("chol")
        fbs = selectbox("fbs")
        restecg = selectbox("restecg")

    with right:
        thalach = slider("thalach")
        exang = selectbox("exang")
        oldpeak = slider("oldpeak")
        slope = selectbox("slope")
        ca = slider("ca")
        thal = selectbox("thal")

    values = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def validate_patient(patient: pd.DataFrame) -> list[str]:
    errors = []
    if list(patient.columns) != FEATURE_COLUMNS:
        errors.append("The model input columns are not in the required training order.")

    if patient.columns.duplicated().any():
        errors.append("The model input contains duplicated feature columns.")

    missing_columns = [column for column in FEATURE_COLUMNS if column not in patient.columns]
    if missing_columns:
        errors.append(f"The model input is missing: {', '.join(missing_columns)}.")

    for column, settings in NUMERIC_INPUTS.items():
        value = patient.at[0, column]
        if not settings["min"] <= value <= settings["max"]:
            errors.append(f"{settings['label']} must be between {settings['min']} and {settings['max']}.")

    for column, settings in CATEGORICAL_INPUTS.items():
        allowed = set(settings["options"].values())
        if patient.at[0, column] not in allowed:
            errors.append(f"{settings['label']} has an unsupported encoded value.")

    return errors


def coerce_input_types(patient: pd.DataFrame) -> pd.DataFrame:
    typed = patient.copy()
    integer_columns = {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "slope", "ca", "thal"}
    for column in FEATURE_COLUMNS:
        if column in integer_columns:
            typed[column] = typed[column].astype("int64")
        else:
            typed[column] = typed[column].astype("float64")
    return typed[FEATURE_COLUMNS]


def validate_artifact(artifact: dict, saved_preprocessor) -> None:
    required_keys = {
        "pipeline",
        "model_name",
        "test_accuracy",
        "feature_columns",
        "target_mapping",
        "positive_class",
        "positive_class_label",
    }
    missing_keys = required_keys - set(artifact)
    if missing_keys:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing_keys)}")

    pipeline = artifact["pipeline"]
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Model artifact pipeline must be a scikit-learn Pipeline.")
    if "preprocess" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Model pipeline must contain preprocess and model steps.")
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Model pipeline must support predict_proba for probability display.")

    saved_columns = list(artifact["feature_columns"])
    if saved_columns != FEATURE_COLUMNS:
        raise ValueError("Model feature columns do not match the current application config.")

    preprocessor = pipeline.named_steps["preprocess"]
    transformer_columns = list(preprocessor.transformers[0][2])
    if transformer_columns != FEATURE_COLUMNS:
        raise ValueError("Preprocessing columns do not match the application feature order.")

    saved_transformer_columns = list(saved_preprocessor.transformers[0][2])
    if saved_transformer_columns != transformer_columns:
        raise ValueError("Saved preprocessing artifact does not match the model pipeline preprocessor.")

    if artifact["positive_class"] != 1 or artifact["positive_class_label"] != TRAINING_TARGET_MEANING[1]:
        raise ValueError("Model target mapping is invalid. Class 1 must represent Disease.")

    missing_inputs = [column for column in FEATURE_COLUMNS if column not in NUMERIC_INPUTS and column not in CATEGORICAL_INPUTS]
    if missing_inputs:
        raise ValueError(f"Input widgets are missing configuration for: {missing_inputs}")


def show_model_summary(artifact: dict, metrics: dict) -> None:
    st.sidebar.header("Model Summary")
    sidebar_card("Selected Model", artifact["model_name"])
    sidebar_card("Test Accuracy", f"{artifact['test_accuracy']:.2%}")

    dataset = metrics.get("dataset") or artifact.get("data_summary", {})
    if dataset:
        st.sidebar.divider()
        st.sidebar.subheader("Dataset Summary")
        sidebar_card("Training Samples", str(dataset.get("rows_after_duplicates_removed", "NA")))
        removed = dataset.get("duplicates_removed", 0)
        sidebar_card("Duplicates Removed", str(removed))
        sidebar_card("Features", str(dataset.get("feature_count", len(FEATURE_COLUMNS))))


def prediction_message(label: str) -> str:
    messages = {
        "Low": "The model indicates a low risk of heart disease based on the provided information.",
        "Moderate": "The model indicates a moderate risk of heart disease based on the provided information.",
        "High": "The model indicates a high risk of heart disease based on the provided information.",
    }
    return messages.get(label, "The model has completed the prediction based on the provided information.")


def predict_patient(artifact: dict, patient: pd.DataFrame) -> dict:
    patient = coerce_input_types(patient)
    validation_errors = validate_patient(patient)
    if validation_errors:
        raise ValueError(" ".join(validation_errors))

    pipeline = artifact["pipeline"]
    prediction = int(pipeline.predict(patient)[0])
    probability = float(pipeline.predict_proba(patient)[0][1])
    label, _ = get_risk_band(probability)
    return {
        "patient": patient,
        "prediction": prediction,
        "probability": probability,
        "risk_label": label,
    }


def show_result(artifact: dict, result: dict) -> None:
    prediction = result["prediction"]
    probability = result["probability"]
    label = result["risk_label"]

    st.subheader("Prediction")
    result_text = "Elevated Risk" if prediction == 1 else "Lower Risk"
    probability_text = f"{probability:.1%}"
    band_text = risk_label(label)
    st.markdown(
        dedent(f"""
        <div class="card-grid">
            {html_card("Prediction", result_text, band_text)}
            {html_card("Estimated Probability", probability_text, band_text)}
            {html_card("Model Used", artifact["model_name"], f"Test accuracy {artifact['test_accuracy']:.2%}")}
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    callout_class = label.lower() if label in {"Low", "Moderate", "High"} else ""
    st.markdown(
        f'<div class="risk-callout {callout_class}">{prediction_message(label)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(risk_meter(probability), unsafe_allow_html=True)

    st.caption("This project is for learning only. It is not a medical diagnosis tool.")


def show_report(metrics: dict) -> None:
    if not metrics:
        st.info("Train the model to generate the report section.")
        return

    model_rows = []
    for name, values in metrics.get("models", {}).items():
        model_rows.append(
            {
                "Model": name,
                "Accuracy": format_percent(values.get("test_accuracy")),
                "Precision": format_percent(values.get("test_precision")),
                "Recall": format_percent(values.get("test_recall")),
                "F1 score": format_score(values.get("test_f1")),
                "ROC-AUC": format_score(values.get("test_roc_auc")),
                "CV accuracy": format_percent(values.get("cv_best_accuracy")),
            }
        )

    if model_rows:
        st.subheader("Model Comparison")
        st.markdown(
            '<div class="section-note">The table below compares the performance of all trained machine learning models on the test dataset. Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Cross-Validation Accuracy are included to provide a complete evaluation of each model.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)

    best_model = metrics.get("best_model")
    best_values = metrics.get("models", {}).get(best_model, {}) if best_model else {}
    dataset = metrics.get("dataset") or {}
    if best_model or dataset:
        st.subheader("Training Summary")
        st.markdown(
            dedent(f"""
            <div class="card-grid">
                {html_card("Best Model", best_model or "-", "Chosen from explainable models first, then by ROC-AUC, accuracy, and F1 score.")}
                {html_card("Test ROC-AUC", format_score(best_values.get("test_roc_auc")), "Measures how well the model distinguishes between patients with and without heart disease.")}
                {html_card("Training Samples", str(dataset.get("rows_after_duplicates_removed", "-")), "Total samples used for training after data preprocessing and duplicate removal.")}
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    importance = metrics.get("best_model_feature_importance") or []
    if importance:
        st.subheader("Feature Importance")
        st.markdown(
            '<div class="section-note">The chart below shows the most influential features identified by the Gradient Boosting model. Features with higher importance contribute more to the final prediction.</div>',
            unsafe_allow_html=True,
        )
        importance_df = pd.DataFrame(importance).head(8)
        st.bar_chart(importance_df.set_index("feature"))

    if CONFUSION_MATRIX_PATH.exists():
        st.subheader("Confusion Matrices")
        st.markdown(
            '<div class="section-note">The confusion matrices below summarize the prediction results of each trained model on the test dataset, showing correct and incorrect classifications for both classes.</div>',
            unsafe_allow_html=True,
        )
        st.image(str(CONFUSION_MATRIX_PATH), width="stretch")

    if dataset:
        st.divider()
        st.subheader("Dataset Information")
        st.markdown(
            '<div class="section-note"><strong>Source:</strong><br>Heart Disease Dataset (Kaggle)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            dedent(f"""
            <div class="card-grid">
                {html_card("Original Samples", str(dataset.get("rows_before_duplicates_removed", "-")), "Rows before duplicate removal")}
                {html_card("Training Samples", str(dataset.get("rows_after_duplicates_removed", "-")), "Rows used after preprocessing")}
                {html_card("Input Features", str(dataset.get("feature_count", len(FEATURE_COLUMNS))), "Clinical inputs used by the model")}
            </div>
            <div class="card-grid">
                {html_card("Target Classes", str(len(CLASS_LABELS)), "No Disease and Disease")}
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Disclaimer")
    st.markdown(
        '<div class="section-note">This application is developed for learning and demonstration purposes only. Predictions generated by the model should not be considered medical advice or used for clinical decision-making.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Built with")
    st.markdown("Python • Pandas • NumPy • Scikit-learn • Streamlit")
    st.caption("Created by Yugant Verma")


def main() -> None:
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="heart", layout="wide")
    apply_page_style()

    st.markdown(
        dedent("""
        <div class="hero-panel">
            <h1>Heart Disease Risk Predictor</h1>
            <p>Enter the patient's clinical details below to estimate the likelihood of heart disease using the trained machine learning model.</p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("No trained model found. Your data is ready in `data/heart.csv`; run `python src/train.py` first.")
        st.stop()
    if not PREPROCESSOR_PATH.exists():
        st.error("The saved preprocessing pipeline is missing. Run `python src/train.py` to rebuild model artifacts.")
        st.stop()

    try:
        artifact = load_model()
        saved_preprocessor = load_preprocessor()
        validate_artifact(artifact, saved_preprocessor)
    except Exception as exc:
        traceback.print_exc()
        st.error("The saved model or preprocessing pipeline could not be loaded. Re-train the project artifacts.")
        st.stop()

    metrics = load_metrics()
    show_model_summary(artifact, metrics)

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    input_tab, report_tab = st.tabs(["Prediction", "Training Report"])

    with input_tab:
        patient = build_input()
        if st.session_state.prediction_result is not None:
            current_patient = coerce_input_types(patient)
            previous_patient = st.session_state.prediction_result["patient"]
            if not current_patient.equals(previous_patient):
                st.session_state.prediction_result = None

        st.divider()
        action_col, spacer_col = st.columns([0.28, 0.72])
        with action_col:
            if st.button("Predict Heart Disease Risk", type="primary", width="stretch"):
                try:
                    with st.spinner("Running prediction..."):
                        st.session_state.prediction_result = predict_patient(artifact, patient)
                    st.success("Prediction completed.")
                except Exception:
                    traceback.print_exc()
                    st.session_state.prediction_result = None
                    st.error("Prediction failed. Please check the inputs and try again.")
        with spacer_col:
            st.empty()

        if st.session_state.prediction_result is not None:
            show_result(artifact, st.session_state.prediction_result)

            with st.expander("View model input row"):
                st.dataframe(st.session_state.prediction_result["patient"], width="stretch", hide_index=True)

    with report_tab:
        show_report(metrics)


if __name__ == "__main__":
    main()
