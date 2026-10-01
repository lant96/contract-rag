"""Download the official CUAD v1 release.

CUAD (Contract Understanding Atticus Dataset) is published by The Atticus Project
under CC BY 4.0.
"""

import urllib.request
import zipfile
from pathlib import Path

CUAD_URL = "https://github.com/The-Atticus-Project/cuad/raw/main/data.zip"
CUAD_JSON_NAME = "CUADv1.json"
TEST_JSON_NAME = "test.json"
DEFAULT_RAW_DIR = Path("data/raw")


def download_cuad(dest_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download CUADv1.json into ``dest_dir`` and return its path."""

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if it is already downloaded
    json_path = dest_dir / CUAD_JSON_NAME
    test_path = dest_dir / TEST_JSON_NAME
    if json_path.exists() and test_path.exists():
        return json_path

    # Temporary location for the ZIP file
    zip_path = dest_dir / "cuad_data.zip"

    # Download
    print(f"Downloading CUAD from {CUAD_URL} ...")
    urllib.request.urlretrieve(CUAD_URL, zip_path)

    # Extract JSON file
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(CUAD_JSON_NAME, dest_dir)
        zf.extract(TEST_JSON_NAME, dest_dir)

    # Remove the temporary ZIP
    zip_path.unlink()

    return json_path
