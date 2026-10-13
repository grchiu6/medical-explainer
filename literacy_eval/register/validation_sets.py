"""Outside validation sets: text the classifier never saw in training, to check it learned REGISTER.

WHY THESE EXIST:
The held-out test split comes from the same books and magazine as training, so a classifier that
separates the classes by source or topic still scores well on it (the word-counting baseline already
does, at 0.98). These sets come from other sources:

  readmepp_hi.jsonl    ReadMe++ Hindi (Naous et al., EMNLP 2024, MIT license): 3,170 sentences, each
                       rated 1-6 (CEFR A1..C2) for difficulty by human annotators, from many domains.
                       Check: classifier scores should RANK the sentences like the humans did
                       (Spearman correlation > 0). Our stand-in for a Hindi annotator.
  miracl_medical.jsonl Hindi Wikipedia medical passages (MIRACL `hi` corpus, 2022 dump). Encyclopedic
                       text for educated adults. Check: should score 10th/professional, rarely 5th.

Both go through the SAME normalize() as the training corpus and the LLM outputs.
(COIL-D Health v2, lay public-health messages, is gated on Hugging Face: add it once access is granted.)

Writes data/register_eval/{readmepp_hi,miracl_medical}.jsonl. Upload them to ~/scratch/register_eval/
on PACE and run scripts/register_score.sh.

    python -m literacy_eval.register.validation_sets --readmepp-dir ../readme/dataset/hi
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import zipfile
from pathlib import Path

import pandas as pd

from literacy_eval.register.chunk import MAX_WORDS, MIN_WORDS, n_words
from literacy_eval.register.clean_sources import health_hits
from literacy_eval.register.normalize import normalize
from literacy_eval.register.sources import RAW_DIR, _fetch

_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = _ROOT / "data" / "register_eval"
MIRACL_URL = ("https://huggingface.co/datasets/miracl/miracl-corpus/resolve/refs%2Fconvert%2Fparquet/"
              "hi/train/0000.parquet")
MIRACL_PARQUET = RAW_DIR / "miracl_hi.parquet"
MIRACL_SAMPLE = 600
SEED = 42

# ---------- a minimal .xlsx reader (openpyxl can't be installed in this venv, and its XML library is broken) ----------
_SI = re.compile(r"<si>(.*?)</si>", re.S)
_T = re.compile(r"<t[^>]*>(.*?)</t>", re.S)
_ROW = re.compile(r"<row[^>]*>(.*?)</row>", re.S)
_CELL = re.compile(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', re.S)
_V = re.compile(r"<v>(.*?)</v>", re.S)


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path: Path) -> list[dict]:
    """Rows of the first sheet as dicts keyed by the header row. Cells are placed by their address
    (e.g. "C5"), because Excel omits empty cells and reading in order would shift columns."""
    with zipfile.ZipFile(path) as z:
        shared = [html.unescape("".join(_T.findall(si))) for si in _SI.findall(z.read("xl/sharedStrings.xml").decode())]
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
    rows = []
    for row_xml in _ROW.findall(sheet):
        values: dict[int, str] = {}
        for letters, attrs, inner in _CELL.findall(row_xml):
            inner = inner or ""
            if 't="s"' in attrs:
                v = _V.search(inner)
                values[_col_index(letters)] = shared[int(v.group(1))] if v else ""
            elif 't="inlineStr"' in attrs:
                values[_col_index(letters)] = html.unescape("".join(_T.findall(inner)))
            else:
                v = _V.search(inner)
                values[_col_index(letters)] = html.unescape(v.group(1)) if v else ""
        rows.append(values)
    header = [rows[0].get(i, "") for i in range(max(rows[0]) + 1)]
    return [{name: r.get(i, "") for i, name in enumerate(header)} for r in rows[1:]]


def build_readmepp(readmepp_dir: Path) -> list[dict]:
    out = []
    for split in ("train", "val", "test"):
        for i, row in enumerate(read_xlsx(readmepp_dir / f"readme_hi_{split}.xlsx")):
            text = normalize(row.get("Sentence", ""))
            if n_words(text) < 3 or not row.get("Rating"):
                continue
            out.append({"id": f"readmepp/{split}/{i}", "text": text, "rating": float(row["Rating"]),
                        "domain": row.get("Domain", ""), "sub_domain": row.get("Sub-domain", ""), "split": split})
    return out


# ---------- MIRACL Hindi Wikipedia: medical articles ----------
_DISEASE_CUE = re.compile(r"रोग|बीमारी|संक्रमण|विकार|सिंड्रोम|लक्षण|चिकित्सा|उपचार")
_NOT_MEDICAL_ARTICLE = re.compile(r"\(जन्म|का जन्म|जन्मे|पुरस्कार|ज्योतिष|कुंडली|अस्पताल|महाविद्यालय|विश्वविद्यालय|फ़िल्म|फिल्म")


def medical_article_ids(df: pd.DataFrame) -> set[str]:
    """Articles whose FIRST paragraph is about a disease or treatment (not a doctor, hospital or film)."""
    lead = df[df["docid"].str.endswith("#0")]
    keep = lead[lead["text"].map(health_hits).ge(4)
                & lead["text"].str.contains(_DISEASE_CUE)
                & ~lead["text"].str.contains(_NOT_MEDICAL_ARTICLE)]
    return set(keep["docid"].str.split("#").str[0])


def build_miracl_medical() -> list[dict]:
    _fetch(MIRACL_URL, MIRACL_PARQUET)
    df = pd.read_parquet(MIRACL_PARQUET)
    ids = medical_article_ids(df)
    df = df[df["docid"].str.split("#").str[0].isin(ids)]
    rows, seen = [], set()
    for r in df.itertuples():
        text = normalize(r.text)
        if not MIN_WORDS <= n_words(text) <= MAX_WORDS or text in seen:
            continue
        seen.add(text)
        rows.append({"id": f"miracl/{r.docid}", "text": text, "title": r.title, "article": r.docid.split("#")[0]})
    random.Random(SEED).shuffle(rows)
    return rows[:MIRACL_SAMPLE]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readmepp-dir", type=Path, default=_ROOT.parent / "readme" / "dataset" / "hi")
    args = parser.parse_args()
    write_jsonl(build_readmepp(args.readmepp_dir), EVAL_DIR / "readmepp_hi.jsonl")
    write_jsonl(build_miracl_medical(), EVAL_DIR / "miracl_medical.jsonl")


if __name__ == "__main__":
    main()
