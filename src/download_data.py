from pathlib import Path
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi

from config import DATA_DIR, DATA_PATH, DATASET_SLUG


def download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=DATA_DIR, unzip=False)

    zip_path = DATA_DIR / f"{DATASET_SLUG.split('/')[-1]}.zip"
    with ZipFile(zip_path) as archive:
        archive.extractall(DATA_DIR)

    possible_csvs = list(DATA_DIR.glob("*.csv"))
    if not possible_csvs:
        raise FileNotFoundError("Kaggle download completed, but no CSV file was found in data/.")

    source_csv = possible_csvs[0]
    if source_csv != DATA_PATH:
        source_csv.replace(DATA_PATH)

    zip_path.unlink(missing_ok=True)
    return DATA_PATH


if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset ready at: {path}")
