import pytest

from performance_eval import tokenizer_stats as ts


class CharTokenizer:
    """Fake tokenizer: one token per non-space character."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [c for c in text if not c.isspace()]


def test_fertility():
    assert ts.fertility(CharTokenizer(), "ab cd") == 2.0
    assert ts.fertility(CharTokenizer(), "abc\nd") == 2.0


def test_fertility_empty_raises():
    with pytest.raises(ValueError):
        ts.fertility(CharTokenizer(), "   ")


def test_quartile_table():
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    correct = [True, True, True, False, False, False, False, True]
    rows = ts.quartile_table(values, correct)
    assert [r["quartile"] for r in rows] == [1, 2, 3, 4]
    assert sum(r["n"] for r in rows) == 8
    assert [r["n"] for r in rows] == [2, 2, 2, 2]
    assert [r["accuracy"] for r in rows] == [1.0, 0.5, 0.0, 0.5]
    assert rows[0]["lo"] == 1 and rows[0]["hi"] == 2 and rows[3]["hi"] == 8


def test_quartile_table_with_ties_and_validates():
    rows = ts.quartile_table([1, 1, 1, 1, 2, 2, 2, 2], [True] * 8)
    assert sum(r["n"] for r in rows) == 8
    with pytest.raises(ValueError):
        ts.quartile_table([1, 2], [True])


def test_build_report_has_means_and_quartiles():
    texts = {"hi": {f"i{k}": "क ख ग" * (k + 1) for k in range(8)}, "en": {f"i{k}": "a b" for k in range(8)}}
    correct = {f"i{k}": k < 4 for k in range(8)}
    md = ts.build_report(texts, correct, {"HiMed": CharTokenizer(), "Nanda": CharTokenizer()})
    assert "| HiMed |" in md and "| Nanda |" in md
    assert md.count("| Q1 |") == 2
