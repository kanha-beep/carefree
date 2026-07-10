from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import DATASET_URL, RAW_FILE, RAW_DIR, RAW_ZIP_FILE


def _extract_csv_from_zip(zip_path: Path) -> None:
    with ZipFile(zip_path, "r") as archive:
        archive.extractall(RAW_DIR)


def download_public_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(DATASET_URL, RAW_ZIP_FILE)
        _extract_csv_from_zip(RAW_ZIP_FILE)
    except (URLError, TimeoutError, OSError):
        dataset = fetch_ucirepo(id=296)
        dataset.data.original.to_csv(RAW_FILE, index=False)
        return

    if RAW_FILE.exists():
        return

    for candidate in RAW_DIR.rglob("diabetic_data.csv"):
        candidate.replace(RAW_FILE)
        return

    raise FileNotFoundError("Downloaded dataset archive did not contain diabetic_data.csv")


def load_raw_data() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        download_public_dataset()

    return pd.read_csv(RAW_FILE)
