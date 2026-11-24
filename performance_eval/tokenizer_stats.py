"""Phase A tokenizer fertility (plan A7): tokens per whitespace word for the HiMed and Nanda tokenizers.

fertility = len(tokenizer.encode(text, add_special_tokens=False)) / number of whitespace-separated words,
per item on question_with_options, in Hindi and in English (google translation). Descriptive only.

  python -m performance_eval.tokenizer_stats \\
      --hi-scored performance_eval/outputs/milu_hi_med/himed_c0_baseline_scored.csv \\
      --en-raw performance_eval/outputs/milu_en_med/himed_c0_en_google_raw.csv \\
      --exclude-file performance_eval/data/excluded_items.csv \\
      --exclude-file performance_eval/data/bad_translation_items.csv \\
      --out performance_eval/analysis_outputs/phase_a/tokenizer_fertility.md
Downloads only the two tokenizers (no weights). HF_TOKEN is read from .env if a repo needs it.
"""

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

from performance_eval.exclusions import load_exclusions

TOKENIZERS = {"HiMed": ("FreedomIntelligence/HiMed-8B", False),
              "Nanda": ("MBZUAI/Llama-3-Nanda-10B-Chat", True)}  # name -> (repo, trust_remote_code)


def fertility(tokenizer, text: str) -> float:
    words = text.split()
    if not words:
        raise ValueError("text has no words")
    return len(tokenizer.encode(text, add_special_tokens=False)) / len(words)


def quartile_table(values: list[float], correct: list[bool]) -> list[dict]:
    """Split items into 4 equal-count groups by value rank (ties ordered by input position) and report
    accuracy per group. lo/hi are the smallest/largest value in the group."""
    if len(values) != len(correct):
        raise ValueError(f"{len(values)} values but {len(correct)} correctness flags")
    if len(values) < 4:
        raise ValueError("need at least 4 items for quartiles")
    order = sorted(range(len(values)), key=lambda i: values[i])
    n = len(order)
    out = []
    for q in range(4):
        idx = order[q * n // 4:(q + 1) * n // 4]
        out.append({"quartile": q + 1, "lo": values[idx[0]], "hi": values[idx[-1]], "n": len(idx),
                    "accuracy": sum(bool(correct[i]) for i in idx) / len(idx)})
    return out


def _read(path: Path, excluded: set[str]) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f) if r["item_id"] not in excluded}


def _quartile_md(title: str, rows: list[dict], fmt: str) -> list[str]:
    lines = [f"### {title}", "", "| quartile | range | n | HiMed Hindi accuracy |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| Q{r['quartile']} | {r['lo']:{fmt}}–{r['hi']:{fmt}} | {r['n']} | {r['accuracy'] * 100:.1f}% |")
    return lines + [""]


def build_report(texts: dict[str, dict[str, str]], correct: dict[str, bool], tokenizers: dict) -> str:
    """texts: {"hi": {item_id: text}, "en": {...}}; correct: HiMed Hindi correctness per item_id."""
    ids = sorted(correct)
    fert = {(name, lang): [fertility(tok, texts[lang][i]) for i in ids]
            for name, tok in tokenizers.items() for lang in ("hi", "en")}
    lines = ["# Tokenizer fertility (HiMed vs Nanda)", "",
             f"Items: {len(ids)} (388-item set after both exclusion files). Fertility = tokens / whitespace "
             "words on question_with_options, add_special_tokens=False. English = google translation. "
             "Descriptive only, no significance test.", "",
             "## Mean fertility (median in parentheses)", "",
             "| tokenizer | Hindi | English | Hindi / English |", "|---|---|---|---|"]
    for name in tokenizers:
        h, e = fert[(name, "hi")], fert[(name, "en")]
        lines.append(f"| {name} | {statistics.mean(h):.2f} ({statistics.median(h):.2f}) | "
                     f"{statistics.mean(e):.2f} ({statistics.median(e):.2f}) | "
                     f"{statistics.mean(h) / statistics.mean(e):.2f} |")
    flags = [correct[i] for i in ids]
    lines += ["", "## HiMed Hindi accuracy by quartile", ""]
    lines += _quartile_md("By HiMed-tokenizer Hindi fertility", quartile_table(fert[("HiMed", "hi")], flags), ".2f")
    lines += _quartile_md("By Hindi word count (length confound check)",
                          quartile_table([len(texts["hi"][i].split()) for i in ids], flags), ".0f")
    lines += ["If accuracy falls with fertility but not with word count, fragmentation (not just length) "
              "tracks errors; if both fall together, length is a confound.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hi-scored", type=Path, required=True, help="HiMed Hindi c0_baseline scored CSV")
    ap.add_argument("--en-raw", type=Path, required=True, help="any English (google) raw/scored CSV of the same items")
    ap.add_argument("--exclude-file", action="append", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    excluded: set[str] = set()
    for f in args.exclude_file:
        excluded |= set(load_exclusions(f))
    hi, en = _read(args.hi_scored, excluded), _read(args.en_raw, excluded)
    if set(hi) != set(en):
        raise SystemExit(f"item sets differ between --hi-scored and --en-raw ({len(set(hi) ^ set(en))} items)")

    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    tokenizers = {name: AutoTokenizer.from_pretrained(repo, trust_remote_code=trc, token=token)
                  for name, (repo, trc) in TOKENIZERS.items()}
    report = build_report({"hi": {i: r["question_with_options"] for i, r in hi.items()},
                           "en": {i: r["question_with_options"] for i, r in en.items()}},
                          {i: r["is_correct"] in ("1", "True") for i, r in hi.items()}, tokenizers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    sys.exit(main())
