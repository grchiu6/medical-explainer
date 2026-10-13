"""Stage 5 of the register corpus: clean -> normalize -> chunk, and write the corpus plus a report.

WHAT THIS FILE DOES:
  1. For every kept PDF in data/register_raw/manifest.csv, cleans its OCR'd
     pages (clean_sources) into prose paragraphs.
  2. Groups paragraphs into split units ("groups"): a chapter for NCERT Class 5,
     an issue for ICMR, and a 4-page block of a chapter for NCERT Class 10,
     which has only 5 biology chapters (chapter-level splits would put a
     single chapter in test).
  3. Per group: normalize() each paragraph, split into sentences, drop
     sentences already seen anywhere (de-duplication), pack into passages whose
     lengths follow the usable LLM-output lengths.
  4. Assigns whole groups to train/val/test (~70/15/15 of passages, per class),
     then balances classes within each split.
  5. Writes {train,val,test}.jsonl and corpus_report.md.

TWO VARIANTS (the 10th-grade class is the smallest, so it sets every class's size):
  class10     10th grade = NCERT Class 10 biology only   -> data/register_corpus/
  class9_10   10th grade = Class 10 + Class 9 biology    -> data/register_corpus_9_10/
              (more data per class, but the class spans two grades)
Train on both and compare.

    python -m literacy_eval.register.build_corpus --variant class10
    python -m literacy_eval.register.build_corpus --variant class9_10
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from literacy_eval.analysis.clean import CLEAN_CSV
from literacy_eval.register import chunk
from literacy_eval.register.clean_sources import DropCounter, clean_file, keep_file
from literacy_eval.register.normalize import normalize
from literacy_eval.register.sources import MANIFEST_CSV

_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = _ROOT / "data" / "register_corpus"
VARIANT_DIRS = {"class10": CORPUS_DIR, "class9_10": _ROOT / "data" / "register_corpus_9_10"}
LABELS = ["grade5", "grade10", "professional"]
GRADE10_BLOCK_PAGES = 4
SEED = 42


def md_table(df: pd.DataFrame) -> str:
    """A DataFrame as a GitHub markdown table (pandas' to_markdown needs the tabulate package)."""
    df = df.reset_index()
    rows = [" | ".join(map(str, df.columns)), " | ".join("---" for _ in df.columns)]
    rows += [" | ".join("" if pd.isna(v) else str(v) for v in r) for r in df.itertuples(index=False)]
    return "\n".join(f"| {r} |" for r in rows)


def label_for(label: str, variant: str) -> str | None:
    """The class a manifest row belongs to in this variant, or None if the variant doesn't use it."""
    if label == "grade9":
        return "grade10" if variant == "class9_10" else None
    return label


def group_of(row: pd.Series, page: int, label: str) -> str:
    """Split unit. Both Class 9 and Class 10 books have few chapters, so they are split by 4-page block."""
    if label == "grade10":
        return f"{row['group']}/b{(page - 1) // GRADE10_BLOCK_PAGES}"
    return row["group"]


def output_length_targets() -> list[int]:
    clean = pd.read_csv(CLEAN_CSV)
    usable = clean[clean["usable_rules"]]
    return chunk.length_targets([chunk.n_words(normalize(a)) for a in usable["answer"].fillna("")])


def collect_paragraphs(manifest: pd.DataFrame, drops: DropCounter,
                       variant: str = "class10") -> dict[tuple[str, str, str], list[str]]:
    """(label, source, group) -> normalized paragraphs in reading order."""
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for _, row in manifest.iterrows():
        label = label_for(row["label"], variant)
        if label is None or not keep_file(row["source"], row["file"]):
            drops["file:excluded"] += 1
            continue
        for para in clean_file(row["source"], row["file"], int(row["pages"]), drops):
            text = normalize(para["text"])
            if text:
                groups[(label, row["source"], group_of(row, para["page"], label))].append(text)
    return groups


def build_passages(groups: dict, targets: list[int], rng: random.Random, stats: Counter) -> list[dict]:
    seen: set[str] = set()
    passages = []
    for (label, source, group), paragraphs in groups.items():   # manifest order: NCERT first, then ICMR
        sentences = [s for p in paragraphs for s in chunk.split_sentences(p)]
        stats[f"{label}:sentences"] += len(sentences)
        unique = chunk.dedupe_sentences(sentences, seen)
        stats[f"{label}:duplicate_sentences"] += len(sentences) - len(unique)
        for text in chunk.pack_passages(unique, targets, rng):
            passages.append({"text": text, "label": label, "source": source, "group": group})
    return passages


def assign_splits(passages: list[dict], rng: random.Random) -> None:
    for label in LABELS:
        sizes = Counter(p["group"] for p in passages if p["label"] == label)
        assignment = chunk.split_groups(dict(sorted(sizes.items())), rng)
        for p in passages:
            if p["label"] == label:
                p["split"] = assignment[p["group"]]


def balance_splits(passages: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    out = {}
    for split in chunk.SPLIT_SHARES:
        by_class = {label: [p for p in passages if p["label"] == label and p["split"] == split] for label in LABELS}
        balanced = chunk.balance(by_class, rng)
        rows = [p for label in LABELS for p in balanced[label]]
        rng.shuffle(rows)
        out[split] = rows
    return out


def write_report(path: Path, drops: Counter, stats: Counter, passages: list[dict],
                 splits: dict[str, list[dict]], targets: list[int], rng: random.Random, variant: str) -> None:
    pre = pd.DataFrame(passages)
    pre["words"] = pre["text"].map(chunk.n_words)
    pre["number_share"] = pre["text"].map(lambda t: t.split().count("0") / max(1, chunk.n_words(t)))
    pre["sentence_words"] = pre["text"].map(
        lambda t: sum(map(chunk.n_words, chunk.split_sentences(t))) / max(1, len(chunk.split_sentences(t))))
    final = pd.concat([pd.DataFrame(rows).assign(split=s) for s, rows in splits.items()])
    final["words"] = final["text"].map(chunk.n_words)

    lines = ["# Register corpus report", "", f"Variant: **{variant}**. Generated by "
             "`python -m literacy_eval.register.build_corpus`.", ""]
    lines += ["## Final corpus (balanced)", "", md_table(final.pivot_table(index="label", columns="split", values="text",
              aggfunc="count").reindex(LABELS)), ""]
    lines += ["## Before balancing", "", md_table(pre.groupby("label").agg(
        passages=("text", "count"), groups=("group", "nunique"), median_words=("words", "median"),
        mean_sentence_words=("sentence_words", "mean"), number_share=("number_share", "mean"),
    ).reindex(LABELS).round(3)), ""]
    lines += ["Passage-length targets (from usable LLM outputs): "
              f"median {pd.Series(targets).median():.0f}, n={len(targets)}.", ""]
    lines += ["## Passage length by class (final corpus; should look alike)", "",
              md_table(final.groupby("label")["words"].describe(percentiles=[.1, .25, .5, .75, .9]).reindex(LABELS)
                       .round(0)), "",
              "`mean_sentence_words` is expected to differ: sentence length is a real register signal.", ""]
    lines += ["## Sentence counts and duplicates", "", "| key | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(stats.items())] + [""]
    lines += ["## Cleaning drops (per rule)", "", "| rule | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(drops.items())] + [""]
    lines += ["## Sample passages (train, 10 per class)", ""]
    train = [p for p in splits["train"]]
    for label in LABELS:
        pool = [p for p in train if p["label"] == label]
        lines += [f"### {label}", ""]
        for p in rng.sample(pool, min(10, len(pool))):
            lines += [f"- `{p['group']}`: {p['text']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANT_DIRS), default="class10")
    variant = parser.parse_args().variant
    corpus_dir = VARIANT_DIRS[variant]
    rng = random.Random(SEED)
    manifest = pd.read_csv(MANIFEST_CSV)
    targets = output_length_targets()
    drops, stats = DropCounter(), Counter()

    groups = collect_paragraphs(manifest, drops, variant)
    passages = build_passages(groups, targets, rng, stats)
    assign_splits(passages, rng)
    splits = balance_splits(passages, rng)

    corpus_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (corpus_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = corpus_dir / "corpus_report.md"
    write_report(report, drops, stats, passages, splits, targets, rng, variant)
    print(f"[{variant}] passages before balancing:")
    print(pd.DataFrame(passages).groupby("label").size().to_string())
    print({s: len(r) for s, r in splits.items()})
    print(f"wrote {corpus_dir}/{{train,val,test}}.jsonl and {report}")


if __name__ == "__main__":
    main()
