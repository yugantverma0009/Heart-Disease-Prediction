import csv
import compileall
import importlib.util
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
APP_PATH = ROOT_DIR / "app.py"
REQUIRED_FILES = [
    APP_PATH,
    ROOT_DIR / "README.md",
    ROOT_DIR / "requirements.txt",
    SRC_DIR / "config.py",
    SRC_DIR / "train.py",
    SRC_DIR / "predict_cli.py",
    SRC_DIR / "download_data.py",
    ROOT_DIR / "scripts" / "audit_predictions.py",
]
REQUIRED_PACKAGES = ["joblib", "matplotlib", "pandas", "sklearn", "streamlit"]
EXPECTED_COLUMNS = [
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
    "target",
]


def status(label: str, ok: bool, detail: str = "") -> None:
    marker = "OK" if ok else "FAIL"
    message = f"[{marker}] {label}"
    if detail:
        message = f"{message}: {detail}"
    print(message)


def check_files() -> bool:
    missing = [path for path in REQUIRED_FILES if not path.exists()]
    status("required files", not missing, ", ".join(str(path.relative_to(ROOT_DIR)) for path in missing))
    return not missing


def check_compile() -> bool:
    ok = compileall.compile_file(str(APP_PATH), quiet=1) and compileall.compile_dir(str(SRC_DIR), quiet=1)
    status("python syntax", bool(ok))
    return bool(ok)


def check_dependencies() -> bool:
    missing = [package for package in REQUIRED_PACKAGES if importlib.util.find_spec(package) is None]
    status("python dependencies", not missing, "missing " + ", ".join(missing) if missing else "")
    return not missing


def check_artifacts() -> bool:
    sys.path.insert(0, str(SRC_DIR))
    from config import DATA_PATH, DATA_SUMMARY_PATH, METRICS_PATH, MODEL_PATH, PREPROCESSOR_PATH

    data_ok = DATA_PATH.exists()
    model_ok = MODEL_PATH.exists()
    preprocessor_ok = PREPROCESSOR_PATH.exists()
    metrics_ok = METRICS_PATH.exists()
    data_summary_ok = DATA_SUMMARY_PATH.exists()
    status("dataset", data_ok, str(DATA_PATH.relative_to(ROOT_DIR)) if not data_ok else "")
    if data_ok:
        data_ok = check_dataset_columns(DATA_PATH)
    status("trained model", model_ok, str(MODEL_PATH.relative_to(ROOT_DIR)) if not model_ok else "")
    status("preprocessing pipeline", preprocessor_ok, str(PREPROCESSOR_PATH.relative_to(ROOT_DIR)) if not preprocessor_ok else "")
    status("metrics report", metrics_ok, str(METRICS_PATH.relative_to(ROOT_DIR)) if not metrics_ok else "")
    status("data summary", data_summary_ok, str(DATA_SUMMARY_PATH.relative_to(ROOT_DIR)) if not data_summary_ok else "")

    if metrics_ok:
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            status("metrics json", "best_model" in metrics and "models" in metrics)
        except json.JSONDecodeError as exc:
            status("metrics json", False, str(exc))
            return False

    return data_ok and model_ok and preprocessor_ok and metrics_ok and data_summary_ok


def check_dataset_columns(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = next(reader, [])

    missing = [column for column in EXPECTED_COLUMNS if column not in header]
    extra = [column for column in header if column not in EXPECTED_COLUMNS]
    ok = not missing
    detail = ""
    if missing:
        detail = "missing " + ", ".join(missing)
    elif extra:
        detail = "extra columns ignored: " + ", ".join(extra)

    status("dataset columns", ok, detail)
    return ok


def main() -> int:
    files_ok = check_files()
    syntax_ok = check_compile()
    dependencies_ok = check_dependencies()
    artifacts_ok = check_artifacts()
    checks = [files_ok, syntax_ok, dependencies_ok, artifacts_ok]

    if all(checks):
        print("\nProject check passed. You can run: streamlit run app.py")
        return 0

    print("\nProject check completed with action items.")
    if not dependencies_ok:
        print("Install dependencies: pip install -r requirements.txt")
    if dependencies_ok and not artifacts_ok:
        print("Train from uploaded data: python src/train.py")
    else:
        print("After dependencies are installed, train from uploaded data: python src/train.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
