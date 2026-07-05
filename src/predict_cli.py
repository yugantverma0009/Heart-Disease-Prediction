import joblib
import pandas as pd

from config import (
    CATEGORICAL_INPUTS,
    FEATURE_COLUMNS,
    FEATURE_HELP,
    MODEL_PATH,
    NUMERIC_INPUTS,
    get_risk_band,
)


def ask_numeric(column: str) -> float:
    settings = NUMERIC_INPUTS[column]
    while True:
        value = input(
            f"{column} - {FEATURE_HELP[column]} "
            f"({settings['min']} to {settings['max']}, default {settings['default']}): "
        ).strip()
        if not value:
            return float(settings["default"])

        try:
            number = float(value)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if settings["min"] <= number <= settings["max"]:
            return number

        print(f"Please enter a value between {settings['min']} and {settings['max']}.")


def ask_category(column: str) -> int:
    settings = CATEGORICAL_INPUTS[column]
    options = list(settings["options"].items())
    prompt_lines = [f"{column} - {settings['label']}"]
    prompt_lines.extend(f"  {index}. {label}" for index, (label, _) in enumerate(options, start=1))

    while True:
        value = input("\n".join(prompt_lines) + "\nChoose option number: ").strip()
        try:
            selected_index = int(value)
        except ValueError:
            print("Please enter one of the option numbers.")
            continue

        if 1 <= selected_index <= len(options):
            return int(options[selected_index - 1][1])

        print("Please choose one of the listed options.")


def ask_value(column: str) -> float:
    if column in CATEGORICAL_INPUTS:
        return ask_category(column)
    if column in NUMERIC_INPUTS:
        return ask_numeric(column)
    raise KeyError(f"No input configuration found for {column}.")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `python src/train.py` first.")

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    print("Enter patient details.")
    print("Press Enter on numeric fields to use the default value.\n")
    values = {column: ask_value(column) for column in FEATURE_COLUMNS}
    patient = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    prediction = int(pipeline.predict(patient)[0])
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(patient)[0][1])
    band, message = get_risk_band(probability)

    print("\nPrediction result")
    print("-----------------")
    print("Heart disease risk:", "Detected" if prediction == 1 else "Not detected")
    if probability is not None:
        print(f"Estimated probability: {probability:.2%}")
    print(f"Risk band: {band}")
    print(f"Note: {message}")
    print(f"Model used: {artifact.get('model_name', 'Unknown')}")


if __name__ == "__main__":
    main()
