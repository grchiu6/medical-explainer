"""Download datasets used by the baseline evaluation pipeline.

Currently supports:
- IndicQA Hindi (extractive QA, downloaded automatically from HuggingFace)
- HiMed (Hindi medical MCQ, downloaded MANUALLY from Kaggle — see below)

Why two strategies? IndicQA is on HuggingFace's CDN with a stable URL,
so urllib can grab it with no auth. HiMed is on Kaggle, which requires
either Kaggle API auth (kagglehub) or manual download. Manual is simpler
and more reliable for a one-time download — no auth fragility.

Usage:
    python data/download.py            # downloads IndicQA
    # For HiMed: download manually (see HiMed instructions below)
"""

import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent


def download_indicqa_hindi():
    """Download IndicQA Hindi JSON file from HuggingFace.

    The file is saved to data/indicqa/indicqa_hi.json. Idempotent —
    if the file already exists, this is a no-op (no re-download).

    Returns:
        Path to the downloaded JSON file.
    """
    indicqa_dir = DATA_DIR / "indicqa"
    indicqa_dir.mkdir(exist_ok=True)
    output_path = indicqa_dir / "indicqa_hi.json"

    if output_path.exists():
        print(f"IndicQA already exists at {output_path}")
        return output_path

    url = (
        "https://huggingface.co/datasets/ai4bharat/IndicQA"
        "/resolve/main/data/indicqa.hi.json"
    )

    print("Downloading IndicQA Hindi dataset from HuggingFace...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def check_himed_files():
    """Check whether HiMed benchmark files have been manually downloaded.

    HiMed must be downloaded manually from Kaggle — the dataset is
    behind Kaggle's auth wall and the cleanest user experience is
    to grab the JSON files yourself rather than wire up the Kaggle API.

    Expected files (place in data/himed/):
        - HiMed-Trad_Bench.json   (Traditional medicine MCQ benchmark)
        - HiMed-West_Bench.json   (Western/modern medicine MCQ benchmark)

    Source:
        https://www.kaggle.com/datasets/amitkumarjaiswal/himed-hindi-medical-dataset

    Returns:
        Dict mapping each expected filename to True (exists) or False.
    """
    himed_dir = DATA_DIR / "himed"
    himed_dir.mkdir(exist_ok=True)

    expected = ["HiMed-Trad_Bench.json", "HiMed-West_Bench.json"]
    status = {name: (himed_dir / name).exists() for name in expected}

    print(f"HiMed files in {himed_dir}:")
    for name, exists in status.items():
        marker = "[OK]" if exists else "[MISSING]"
        print(f"  {marker} {name}")

    if not all(status.values()):
        print(
            "\nTo download missing files:\n"
            "  1. Visit https://www.kaggle.com/datasets/amitkumarjaiswal/himed-hindi-medical-dataset\n"
            "  2. Download the JSON file(s) listed above\n"
            f"  3. Place them in: {himed_dir}"
        )

    return status


if __name__ == "__main__":
    download_indicqa_hindi()
    print()
    check_himed_files()
