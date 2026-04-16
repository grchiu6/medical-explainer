"""Download the HiMed Hindi medical QA dataset from Kaggle."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def download_himed():
    """Download HiMed dataset to data/himed/ directory."""
    # kagglehub reads KAGGLE_USERNAME and KAGGLE_KEY from environment
    import kagglehub

    data_dir = Path(__file__).parent / "himed"
    data_dir.mkdir(exist_ok=True)

    print("Downloading HiMed dataset from Kaggle...")
    path = kagglehub.dataset_download("aijain/hindi-health-dataset")
    print(f"Dataset downloaded to: {path}")
    print(f"\nFiles found:")
    for f in Path(path).iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    return path


if __name__ == "__main__":
    download_himed()
