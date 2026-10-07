"""Tests for literacy_eval/analysis/aggregate.py on a small synthetic table."""

import numpy as np
import pandas as pd

from literacy_eval.analysis import aggregate as A


def _synthetic_features(n_items=20):
    rows = []
    for i in range(n_items):
        for j, level in enumerate(["low", "mid", "high"]):
            rows.append({
                "model": "himed", "level": level, "item_id": f"i{i}",
                "failure_type": "refusal" if (level == "low" and i < 5) else "ok",
                "usable_rules": not (level == "low" and i < 5),
                "usable": not (level == "low" and i < 5), "judged": True,
                "asl": 10 + 5 * j + (i % 3), "mapped_option": "A" if i % 2 else "NONE",
                **{f: 1.0 + j for f in A.ADAPTATION_FEATURES if f != "asl"},
                "short_sentence_share": 0.9 - 0.3 * j,     # falls from low to high
                "latin_rate": 0.0 if level == "low" else 0.05,
                "truncated": level == "high",
            })
    return pd.DataFrame(rows)


def test_usability_table_counts_failures():
    table = A.usability_table(_synthetic_features()).set_index("level")
    assert table.loc["low", "refusal"] == 5
    assert table.loc["low", "usable"] == 15 and table.loc["high", "usable"] == 20


def test_adaptation_uses_only_items_usable_at_all_levels():
    table = A.adaptation_table(_synthetic_features(), ["himed"]).set_index("feature")
    assert table.loc["asl", "n_items"] == 15                  # 5 low refusals dropped
    assert table.loc["asl", "page_p"] < 0.001
    assert table.loc["asl", "share_expected_direction"] == 1.0
    assert np.isnan(table.loc["n_words", "page_p_holm"])      # not in the Holm family


def test_accuracy_counts_unusable_rows_as_wrong():
    feats = _synthetic_features()
    feats.loc[feats["failure_type"] == "refusal", "mapped_option"] = np.nan
    reference = {f"i{i}": {"correct_answer": "A"} for i in range(20)}
    table = A.accuracy_table(feats, reference, suspect=set())
    low = table[(table["level"] == "low") & (table["items"] == "all_items")].iloc[0]
    assert low["n"] == 20                      # denominator is all items, not just judged ones
    assert low["correct"] == 8                 # odd i >= 5: 5,7,...,19


def test_consistency_table_preserved_share():
    cons = pd.DataFrame({"model": ["himed"] * 4, "pair": ["low-high"] * 4,
                         "verdict": ["SAME", "COMPATIBLE", "CONTRADICTS", "NO_ANSWER"]})
    row = A.consistency_table(cons).iloc[0]
    assert row["preserved_pct"] == 50.0 and row["contradicts_pct"] == 25.0


def test_adaptation_with_too_few_usable_items_reports_nan_not_error():
    feats = _synthetic_features(n_items=6)
    table = A.adaptation_table(feats, ["himed"]).set_index("feature")
    assert table.loc["asl", "n_items"] == 1
    assert np.isnan(table.loc["asl", "page_p"]) and np.isnan(table.loc["asl", "wilcoxon_p"])


def test_load_inputs_keeps_unjudged_rows_usable(tmp_path, monkeypatch):
    from literacy_eval.analysis import judge as J
    feats = pd.DataFrame({"model": ["himed"] * 3, "level": ["low", "mid", "high"],
                          "item_id": ["i1"] * 3, "usable_rules": [True, True, True],
                          "failure_type": ["ok"] * 3})
    feats_csv = tmp_path / "features.csv"
    feats.to_csv(feats_csv, index=False)
    judged = pd.DataFrame({"job_id": ["himed|low|i1", "himed|mid|i1"], "model": ["himed"] * 2,
                           "coherent": ["False", "True"], "mapped_option": ["NONE", "A"],
                           "harm": ["none", "none"], "answer_en": ["", ""], "harm_reason": ["", ""],
                           "judge_error": ["", ""]})
    judged.to_csv(tmp_path / "judge_response.csv", index=False)
    for task in J.TASK_FILES:
        monkeypatch.setitem(J.TASK_FILES, task, tmp_path / f"judge_{task}.csv")
    out = A.load_inputs(feats_csv)["features"].set_index("level")
    assert out.loc["low", "usable"] == False        # judged incoherent
    assert out.loc["mid", "usable"] == True         # judged coherent
    assert out.loc["high", "usable"] == True        # not judged yet: rules decide
    assert A.usability_table(out.reset_index())["gate"].tolist() == ["rules+judge", "rules+judge", "rules only (upper bound)"]


def test_adaptation_respects_expected_direction():
    table = A.adaptation_table(_synthetic_features(), ["himed"]).set_index("feature")
    assert table.loc["short_sentence_share", "expected"] == "down"
    assert table.loc["short_sentence_share", "share_expected_direction"] == 1.0
    assert table.loc["short_sentence_share", "share_monotonic"] == 1.0
    assert table.loc["short_sentence_share", "page_p"] < 0.001       # tested as decreasing
    assert table.loc["short_sentence_share", "rank_biserial"] > 0     # positive = expected way
    assert table.loc["gloss_rate", "expected"] == "none"
    assert np.isnan(table.loc["gloss_rate", "page_p_holm"])          # no ordered hypothesis


def test_compliance_table_reports_instruction_checks():
    table = A.compliance_table(_synthetic_features()).set_index("level")
    assert table.loc["low", "no_latin_pct"] == 100.0
    assert table.loc["high", "no_latin_pct"] == 0.0
    assert table.loc["low", "mean_short_sentence_share"] == 0.9


def test_usability_table_counts_truncation_separately():
    table = A.usability_table(_synthetic_features()).set_index("level")
    assert table.loc["high", "truncated"] == 20
    assert table.loc["high", "usable"] == 20          # truncation is not a failure
