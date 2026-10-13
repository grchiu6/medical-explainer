"""Stage 2 of the register corpus: OCR every page of every source PDF the same way.

WHAT THIS FILE DOES:
For each PDF in data/register_raw/manifest.csv and each page, renders the page
at 300 dpi (grayscale) and runs Tesseract with the tessdata_best Hindi+English
model. Writes data/register_raw/ocr/<file stem>/p<NNN>.txt (text) and .tsv
(word boxes with confidences, used by clean_sources to drop low-confidence lines).

WHY OCR EVEN WHEN A PDF HAS A TEXT LAYER:
ICMR PDFs have no usable text (scans, or a legacy font that extracts as
"|É¶xÉÉå BÉEä"), so they must be OCR'd. If NCERT used its own text layer, only
the professional class would carry OCR errors, and the classifier could learn
"OCR noise = professional". Same OCR for every source = same noise everywhere.

RESUMABLE: pages whose .txt already exists are skipped, so a killed run can
simply be restarted.

    python -m literacy_eval.register.ocr --workers 10
    python -m literacy_eval.register.ocr --source icmr --limit-files 3   # quick test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from literacy_eval.register.sources import MANIFEST_CSV, RAW_DIR, TESSDATA_DIR

OCR_DIR = RAW_DIR / "ocr"
DPI = 300
LANGS = "hin+eng"
PSM = "1"   # automatic page segmentation with orientation detection; handles ICMR's two columns


def page_out_dir(pdf_file: str) -> Path:
    return OCR_DIR / Path(pdf_file).stem


def ocr_page(pdf_path: str, page: int, out_base: str) -> str:
    """Render one page and OCR it to <out_base>.txt and <out_base>.tsv. Returns out_base."""
    env = {**os.environ, "TESSDATA_PREFIX": str(TESSDATA_DIR), "OMP_THREAD_LIMIT": "1"}
    with tempfile.TemporaryDirectory() as tmp:
        img_base = Path(tmp) / "page"
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(DPI), "-gray",
                        "-singlefile", "-png", pdf_path, str(img_base)], check=True, capture_output=True)
        subprocess.run(["tesseract", f"{img_base}.png", out_base, "-l", LANGS, "--psm", PSM, "txt", "tsv"],
                       check=True, capture_output=True, env=env)
    return out_base


def pending_jobs(manifest: pd.DataFrame) -> list[tuple[str, int, str]]:
    jobs = []
    for row in manifest.itertuples():
        out_dir = page_out_dir(row.file)
        out_dir.mkdir(parents=True, exist_ok=True)
        for page in range(1, int(row.pages) + 1):
            base = out_dir / f"p{page:03d}"
            if not base.with_suffix(".txt").exists():
                jobs.append((str(RAW_DIR / row.file), page, str(base)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--source", choices=["ncert", "icmr"], help="only OCR this source")
    parser.add_argument("--limit-files", type=int, help="only the first N PDFs (for testing)")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_CSV)
    if args.source:
        manifest = manifest[manifest["source"] == args.source]
    if args.limit_files:
        manifest = manifest.head(args.limit_files)
    jobs = pending_jobs(manifest)
    print(f"{len(jobs)} pages to OCR with {args.workers} workers")
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ocr_page, *job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except subprocess.CalledProcessError as exc:
                failed += 1
                print(f"  failed {futures[fut][:2]}: {exc.stderr[-200:] if exc.stderr else exc}")
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} pages done", flush=True)
    print(f"done; {failed} pages failed")


if __name__ == "__main__":
    main()
