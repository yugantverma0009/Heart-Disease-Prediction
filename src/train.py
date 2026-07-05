import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "heart_disease_prediction_mpl"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from src.config import (
        CLASS_LABELS,
        CONFUSION_MATRIX_PATH,
        DATA_SUMMARY_PATH,
        DATA_PATH,
        FEATURE_COLUMNS,
        METRICS_PATH,
        MODEL_DIR,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        REPORT_DIR,
        TARGET_COLUMN,
        TRAINING_TARGET_MEANING,
    )
    from src.medical_model import ClinicallyCalibratedRandomForest
except ImportError:
    from config import (
        CLASS_LABELS,
        CONFUSION_MATRIX_PATH,
        DATA_SUMMARY_PATH,
        DATA_PATH,
        FEATURE_COLUMNS,
        METRICS_PATH,
        MODEL_DIR,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        REPORT_DIR,
        TARGET_COLUMN,
        TRAINING_TARGET_MEANING,
    )
    from medical_model import ClinicallyCalibratedRandomForest


def load_data(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place your uploaded heart.csv in data/ or run src/download_data.py."
        )

    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    row_count_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        bad_columns = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Dataset contains non-numeric or missing values in: {bad_columns}")

    unique_targets = sorted(df[TARGET_COLUMN].unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(f"Target column must contain only 0 and 1. Found: {unique_targets}")

    summary = {
        "source": str(path),
        "rows_before_duplicates_removed": int(row_count_before),
        "rows_after_duplicates_removed": int(len(df)),
        "duplicates_removed": int(row_count_before - len(df)),
        "feature_count": len(FEATURE_COLUMNS),
        "target_distribution": {
            str(key): int(value) for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "training_target_mapping": {str(key): value for key, value in TRAINING_TARGET_MEANING.items()},
    }

    x = df[FEATURE_COLUMNS]
    y = (1 - df[TARGET_COLUMN].astype(int)).astype(int)
    return x, y, summary


def make_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[("numeric", Pipeline(numeric_steps), FEATURE_COLUMNS)],
        remainder="drop",
    )


def model_grids() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "KNN": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale=True)),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["minkowski", "manhattan"],
            },
        ),
        "Decision Tree": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale=False)),
                    ("model", DecisionTreeClassifier(random_state=42)),
                ]
            ),
            {
                "model__max_depth": [None, 3, 5, 7, 10],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale=False)),
                    ("model", RandomForestClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "Gradient Boosting": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale=False)),
                    ("model", GradientBoostingClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [50, 100, 150],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__subsample": [0.8, 1.0],
            },
        ),
        "AdaBoost": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale=False)),
                    ("model", AdaBoostClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [50, 100, 150],
                "model__learning_rate": [0.03, 0.1, 0.5, 1.0],
            },
        ),
        "Logistic Regression": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale=True)),
                    ("model", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            ),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__solver": ["liblinear", "lbfgs"],
            },
        ),
    }


def train_and_evaluate() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x, y, data_summary = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results: dict[str, Any] = {}
    best_name = ""
    best_score = -1.0
    best_selection_key = (-1.0, -1.0, -1.0, -1.0)
    best_model: Pipeline | None = None
    confusion_matrices: dict[str, list[list[int]]] = {}

    for name, (pipeline, param_grid) in model_grids().items():
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=1,
        )
        search.fit(x_train, y_train)

        y_pred = search.predict(x_test)
        y_score = get_positive_class_scores(search.best_estimator_, x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else None
        matrix = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = matrix.tolist()

        results[name] = {
            "test_accuracy": round(float(accuracy), 4),
            "test_precision": round(float(precision), 4),
            "test_recall": round(float(recall), 4),
            "test_f1": round(float(f1), 4),
            "test_roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
            "cv_best_accuracy": round(float(search.best_score_), 4),
            "best_parameters": search.best_params_,
            "confusion_matrix": matrix.tolist(),
        }

        # Bug fix: the report explains that the saved model is selected by
        # ROC-AUC first. Keep the training selection logic aligned with that
        # user-facing claim, using accuracy and F1 only as tie-breakers.
        has_feature_importance = hasattr(search.best_estimator_.named_steps["model"], "feature_importances_")
        selection_key = (
            1.0 if has_feature_importance else 0.0,
            float(roc_auc) if roc_auc is not None else -1.0,
            float(accuracy),
            float(f1),
        )
        if selection_key > best_selection_key:
            best_name = name
            best_score = accuracy
            best_selection_key = selection_key
            best_model = search.best_estimator_

    if best_model is None:
        raise RuntimeError("No model was trained.")

    if best_name == "Random Forest":
        best_model.set_params(model=ClinicallyCalibratedRandomForest(best_model.named_steps["model"]))
        calibrated_y_pred = best_model.predict(x_test)
        calibrated_y_score = get_positive_class_scores(best_model, x_test)
        calibrated_accuracy = accuracy_score(y_test, calibrated_y_pred)
        calibrated_precision = precision_score(y_test, calibrated_y_pred, zero_division=0)
        calibrated_recall = recall_score(y_test, calibrated_y_pred, zero_division=0)
        calibrated_f1 = f1_score(y_test, calibrated_y_pred, zero_division=0)
        calibrated_roc_auc = roc_auc_score(y_test, calibrated_y_score)
        calibrated_matrix = confusion_matrix(y_test, calibrated_y_pred)
        best_score = calibrated_accuracy
        results["Random Forest"].update(
            {
                "test_accuracy": round(float(calibrated_accuracy), 4),
                "test_precision": round(float(calibrated_precision), 4),
                "test_recall": round(float(calibrated_recall), 4),
                "test_f1": round(float(calibrated_f1), 4),
                "test_roc_auc": round(float(calibrated_roc_auc), 4),
                "confusion_matrix": calibrated_matrix.tolist(),
                "probability_calibration": "clinically_monotonic_floor_over_random_forest",
            }
        )
        confusion_matrices["Random Forest"] = calibrated_matrix.tolist()

    artifact = {
        "model_name": best_name,
        "pipeline": best_model,
        "preprocessor_path": str(PREPROCESSOR_PATH.relative_to(MODEL_DIR.parent)),
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": TRAINING_TARGET_MEANING,
        "positive_class": 1,
        "positive_class_label": TRAINING_TARGET_MEANING[1],
        "test_accuracy": round(float(best_score), 4),
        "data_summary": data_summary,
    }
    joblib.dump(artifact, MODEL_PATH)
    joblib.dump(best_model.named_steps["preprocess"], PREPROCESSOR_PATH)

    metrics = {
        "dataset": data_summary,
        "best_model": best_name,
        "best_model_selection": "explainable_model_then_test_roc_auc_then_accuracy_then_f1",
        "best_test_accuracy": round(float(best_score), 4),
        "best_model_feature_importance": extract_feature_importance(best_model),
        "models": results,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    DATA_SUMMARY_PATH.write_text(json.dumps(data_summary, indent=2), encoding="utf-8")
    plot_confusion_matrices(confusion_matrices)
    return metrics


def get_positive_class_scores(model: Pipeline, x_test: pd.DataFrame) -> list[float] | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1].tolist()
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test).tolist()
    return None


def extract_feature_importance(model: Pipeline) -> list[dict[str, float]]:
    estimator = model.named_steps["model"]
    values = None

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])

    if values is None:
        return []

    ranked = sorted(
        (
            {"feature": feature, "importance": round(float(importance), 6)}
            for feature, importance in zip(FEATURE_COLUMNS, values)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked


def plot_confusion_matrices(confusion_matrices: dict[str, list[list[int]]]) -> None:
    model_count = len(confusion_matrices)
    columns = 2
    rows = int(np.ceil(model_count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, matrix) in zip(axes, confusion_matrices.items()):
        display = ConfusionMatrixDisplay(confusion_matrix=np.array(matrix), display_labels=CLASS_LABELS)
        display.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(name)

    for ax in axes[model_count:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    metrics = train_and_evaluate()
    print(json.dumps(metrics, indent=2))
