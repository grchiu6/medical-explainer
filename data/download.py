"""Download the IndicQA Hindi dataset from HuggingFace.

IndicQA is a reading comprehension QA dataset in SQuAD format.
We download the Hindi subset (indicqa.hi.json) which contains 1,547
question-answer pairs across 261 articles.

Source: https://huggingface.co/datasets/ai4bharat/IndicQA
"""

import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def download_indicqa_hindi():
    """Download IndicQA Hindi JSON file from HuggingFace.

    The file is saved to data/indicqa_hi.json in the project directory.
    """
    data_dir = Path(__file__).parent
    output_path = data_dir / "indicqa_hi.json"

    if output_path.exists():
        print(f"Dataset already exists at {output_path}")
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


if __name__ == "__main__":
    download_indicqa_hindi()
