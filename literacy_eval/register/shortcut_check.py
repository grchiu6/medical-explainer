"""Stage 6 of the register corpus: look for shortcuts BEFORE spending GPU time on BERT.

WHAT THIS FILE DOES:
Trains three deliberately simple classifiers on train.jsonl and reports
accuracy on test.jsonl (balanced, 3 classes, so chance = 33%):

  passage_length   Gaussian naive Bayes on passage word count only.
                   Lengths were matched across classes, so this should sit
                   near CHANCE. If it doesn't, length matching failed.
  sentence_length  Gaussian naive Bayes on mean words per sentence. This is a
                   real register signal: the simple baseline BERT must beat.
  word_nb          Multinomial naive Bayes on words. High accuracy is
                   expected (vocabulary differs by register), so the useful
                   output is the TOP WORDS per class: anything that is not
                   language (a masthead, a page label, a recurring name) is a
                   leftover artifact to clean before training BERT.

Only numpy/pandas are used (scikit-learn can't be installed in this venv).

    python -m literacy_eval.register.shortcut_check [--corpus-dir data/register_corpus_9_10]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from literacy_eval.register.build_corpus import CORPUS_DIR, LABELS
from literacy_eval.register.chunk import n_words, split_sentences

TOP_WORDS = 25


def load(corpus_dir: Path, split: str) -> list[dict]:
    with (corpus_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mean_sentence_words(text: str) -> float:
    sentences = split_sentences(text)
    return sum(map(n_words, sentences)) / len(sentences) if sentences else 0.0


class GaussianNB1D:
    """Naive Bayes on one numeric feature: one normal distribution per class."""

    def fit(self, x: list[float], y: list[str]) -> "GaussianNB1D":
        self.params = {}
        for label in sorted(set(y)):
            values = np.array([v for v, l in zip(x, y) if l == label], dtype=float)
            self.params[label] = (values.mean(), values.std() + 1e-6)
        return self

    def predict(self, x: list[float]) -> list[str]:
        def loglik(v, mu, sd):
            return -math.log(sd) - 0.5 * ((v - mu) / sd) ** 2
        return [max(self.params, key=lambda l: loglik(v, *self.params[l])) for v in x]


class WordNB:
    """Multinomial naive Bayes on whitespace tokens, with add-one smoothing."""

    def fit(self, texts: list[str], y: list[str]) -> "WordNB":
        self.counts = {label: Counter() for label in sorted(set(y))}
        for text, label in zip(texts, y):
            self.counts[label].update(text.split())
        self.vocab = set().union(*self.counts.values())
        self.totals = {label: sum(c.values()) for label, c in self.counts.items()}
        return self

    def log_prob(self, word: str, label: str) -> float:
        return math.log((self.counts[label][word] + 1) / (self.totals[label] + len(self.vocab)))

    def predict(self, texts: list[str]) -> list[str]:
        return [max(self.counts, key=lambda l: sum(self.log_prob(w, l) for w in t.split() if w in self.vocab))
                for t in texts]

    def top_words(self, label: str, k: int, min_count: int = 5) -> list[tuple[str, float, int]]:
        """Words most specific to `label`: log P(w|label) minus the mean log P(w|other classes)."""
        others = [l for l in self.counts if l != label]
        scored = []
        for word, count in self.counts[label].items():
            if count < min_count:
                continue
            ratio = self.log_prob(word, label) - np.mean([self.log_prob(word, o) for o in others])
            scored.append((word, round(ratio, 2), count))
        return sorted(scored, key=lambda s: -s[1])[:k]


def accuracy(pred: list[str], gold: list[str]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def confusion(pred: list[str], gold: list[str]) -> list[str]:
    lines = ["| gold \\ predicted | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
    for g in LABELS:
        row = [sum(1 for p, gg in zip(pred, gold) if gg == g and p == q) for q in LABELS]
        lines.append(f"| {g} | " + " | ".join(map(str, row)) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    corpus_dir = parser.parse_args().corpus_dir
    train, test = load(corpus_dir, "train"), load(corpus_dir, "test")
    ytr, yte = [r["label"] for r in train], [r["label"] for r in test]
    lines = ["# Shortcut check", "", f"train = {len(train)} passages, test = {len(test)} passages, "
             "3 balanced classes (chance = 0.33).", ""]

    results = {}
    for name, feature in [("passage_length", lambda t: n_words(t)), ("sentence_length", mean_sentence_words)]:
        model = GaussianNB1D().fit([feature(r["text"]) for r in train], ytr)
        pred = model.predict([feature(r["text"]) for r in test])
        results[name] = (accuracy(pred, yte), pred)
    nb = WordNB().fit([r["text"] for r in train], ytr)
    nb_pred = nb.predict([r["text"] for r in test])
    results["word_nb"] = (accuracy(nb_pred, yte), nb_pred)

    lines += ["| classifier | test accuracy | what it means |", "|---|---|---|",
              f"| passage_length | {results['passage_length'][0]:.3f} | should be near 0.33 (lengths matched) |",
              f"| sentence_length | {results['sentence_length'][0]:.3f} | simple real-register baseline for BERT to beat |",
              f"| word_nb | {results['word_nb'][0]:.3f} | vocabulary baseline; check top words below for artifacts |", ""]
    for name, (_, pred) in results.items():
        lines += [f"### Confusion: {name}", ""] + confusion(pred, yte) + [""]
    lines += ["## Most class-specific words (word_nb)", "",
              "Look for anything that is not language: masthead text, page labels, names, OCR junk.", ""]
    for label in LABELS:
        words = ", ".join(f"{w} ({c})" for w, _, c in nb.top_words(label, TOP_WORDS))
        lines += [f"**{label}:** {words}", ""]
    report = corpus_dir / "shortcut_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    for name, (acc, _) in results.items():
        print(f"{name:16s} {acc:.3f}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
