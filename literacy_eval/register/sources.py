"""Stage 1 of the register corpus: download the raw source PDFs and the Hindi OCR model.

WHAT THIS FILE DOES:
Downloads, into data/register_raw/ (gitignored, because the text is copyrighted
and can be downloaded again):
  - NCERT books, one zip per book with one PDF per chapter:
      ehev1  Class 5  हमारा अद्भुत संसार (new edition)     -> 5th grade
      ehap1  Class 5  आस-पास (old edition)                -> 5th grade
      jhsc1  Class 10 विज्ञान                             -> 10th grade
      ihsc1  Class 9  विज्ञान (अन्वेषण)                    -> "grade9" (optional extra for 10th grade;
                                                              build_corpus --variant class9_10 merges it)
  - every ICMR Patrika issue linked from the ICMR archive page -> professional
  - Tesseract's best Hindi model (tessdata_best/hin.traineddata)
Then writes data/register_raw/manifest.csv: one row per PDF with
source, label, book_or_issue, file, md5, pages.

WHY DE-DUPLICATE BY MD5:
The ICMR archive page links some issues more than once under different file
names. Two copies of an issue would put the same text in train and test.

    python -m literacy_eval.register.sources
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _ROOT / "data" / "register_raw"
MANIFEST_CSV = RAW_DIR / "manifest.csv"
TESSDATA_DIR = RAW_DIR / "tessdata"

NCERT_BOOKS = {          # code -> label
    "ehev1": "grade5",
    "ehap1": "grade5",
    "jhsc1": "grade10",
    "ihsc1": "grade9",
}
NCERT_ZIP_URL = "https://ncert.nic.in/textbook/pdf/{code}dd.zip"
ICMR_ARCHIVE_URL = "https://www.icmr.gov.in/hindi-publication-icmr-patrika"
TESSDATA_URLS = {
    "hin.traineddata": "https://github.com/tesseract-ocr/tessdata_best/raw/main/hin.traineddata",
    "eng.traineddata": "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata",
}

_PDF_HREF = re.compile(r'href="([^"]+\.pdf)"', re.I)
_USER_AGENT = {"User-Agent": "Mozilla/5.0 (research corpus builder)"}


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(urllib.parse.quote(url, safe=":/?=&%"), headers=_USER_AGENT)
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    dest.write_bytes(data)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def pdf_pages(path: Path) -> int:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(match.group(1)) if match else 0


def icmr_pdf_urls(html: str) -> list[str]:
    """PDF links on the archive page that look like Patrika issues, in page order, without repeats."""
    seen, urls = set(), []
    for href in _PDF_HREF.findall(html):
        if "patrika" not in href.lower() or href in seen:
            continue
        seen.add(href)
        urls.append(urllib.parse.urljoin(ICMR_ARCHIVE_URL, href))
    return urls


def download_ncert() -> list[dict]:
    rows = []
    for code, label in NCERT_BOOKS.items():
        zip_path = RAW_DIR / "ncert" / f"{code}.zip"
        _fetch(NCERT_ZIP_URL.format(code=code), zip_path)
        book_dir = RAW_DIR / "ncert" / code
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(book_dir)
        for pdf in sorted(book_dir.glob("*.pdf")):
            rows.append({"source": "ncert", "label": label, "group": f"{code}/{pdf.stem}",
                         "file": str(pdf.relative_to(RAW_DIR))})
    return rows


def download_icmr() -> list[dict]:
    req = urllib.request.Request(ICMR_ARCHIVE_URL, headers=_USER_AGENT)
    html = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
    rows, seen_md5 = [], set()
    for url in icmr_pdf_urls(html):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", urllib.parse.unquote(url.rsplit("/", 1)[-1]))
        dest = RAW_DIR / "icmr" / name
        try:
            _fetch(url, dest)
        except Exception as exc:  # one dead link must not stop the whole download
            print(f"  skip {url}: {exc}")
            continue
        digest = md5(dest)
        if digest in seen_md5:
            dest.unlink()
            continue
        seen_md5.add(digest)
        rows.append({"source": "icmr", "label": "professional", "group": f"icmr/{dest.stem}",
                     "file": str(dest.relative_to(RAW_DIR))})
    return rows


def download_tessdata() -> None:
    for name, url in TESSDATA_URLS.items():
        _fetch(url, TESSDATA_DIR / name)
    # Output formats ("txt", "tsv") are config files that Tesseract looks up in <TESSDATA_PREFIX>/configs.
    # Without them it silently writes only .txt, so copy them from the system install.
    system_configs = Path(subprocess.run(["brew", "--prefix"], capture_output=True, text=True).stdout.strip()
                          or "/usr/local") / "share" / "tessdata" / "configs"
    for alt in (system_configs, Path("/usr/share/tesseract-ocr/5/tessdata/configs"), Path("/usr/share/tessdata/configs")):
        if alt.is_dir():
            shutil.copytree(alt, TESSDATA_DIR / "configs", dirs_exist_ok=True)
            return
    raise FileNotFoundError("Tesseract configs/ folder not found; install tesseract first")


def build_manifest(rows: list[dict]) -> pd.DataFrame:
    manifest = pd.DataFrame(rows)
    manifest["md5"] = [md5(RAW_DIR / f) for f in manifest["file"]]
    manifest = manifest.drop_duplicates("md5").reset_index(drop=True)
    manifest["pages"] = [pdf_pages(RAW_DIR / f) for f in manifest["file"]]
    return manifest


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    download_tessdata()
    rows = download_ncert()
    print(f"NCERT: {len(rows)} chapter PDFs")
    icmr = download_icmr()
    print(f"ICMR: {len(icmr)} unique issues")
    manifest = build_manifest(rows + icmr)
    manifest.to_csv(MANIFEST_CSV, index=False)
    print(manifest.groupby(["source", "label"])["pages"].agg(["count", "sum"]))
    print(f"wrote {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
