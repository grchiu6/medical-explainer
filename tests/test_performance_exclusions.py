import csv

import pytest

from performance_eval.analyze import main as analyze_main
from performance_eval.exclusions import apply_exclusions, load_exclusions
from tests.test_performance_analyze import _write_raw_ids


def _write_excl(path, rows, header=("item_id", "reason")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_apply_exclusions_drops_listed_ids():
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert [it["id"] for it in apply_exclusions(items, {"b": "gov"})] == ["a", "c"]


def test_apply_exclusions_unknown_id_raises():
    with pytest.raises(ValueError, match="zzz"):
        apply_exclusions([{"id": "a"}], {"zzz": "gov"})


def test_load_exclusions_reads_reasons(tmp_path):
    p = tmp_path / "ex.csv"
    _write_excl(p, [["a", "gov"], ["b", "non_medical"]])
    assert load_exclusions(p) == {"a": "gov", "b": "non_medical"}


def test_load_exclusions_requires_columns(tmp_path):
    p = tmp_path / "ex.csv"
    _write_excl(p, [["a"]], header=("item_id",))
    with pytest.raises(ValueError, match="reason"):
        load_exclusions(p)


def test_analyze_exclude_file_filters_scored_rows(tmp_path):
    _write_raw_ids(tmp_path / "m1_c0_raw.csv", ["a", "b", "c"])
    _write_raw_ids(tmp_path / "m1_c2_raw.csv", ["a", "b", "c"])
    ex = tmp_path / "ex.csv"
    _write_excl(ex, [["b", "gov"]])
    analyze_main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
                  "--exclude-file", str(ex), "--ladder", "c0:first_letter", "c2:after_marker"])
    table = (tmp_path / "results_m1.md").read_text(encoding="utf-8")
    row = next(line for line in table.splitlines() if line.startswith("| m1 | c0"))
    assert row.split("|")[3].strip() == "2"


from performance_eval import run as run_mod  # noqa: E402
from performance_eval.data_milu import split_test_fewshot  # noqa: E402
from tests.test_performance_run import _fake_items, fakes  # noqa: E402,F401


def test_run_exclude_file_drops_items_after_split_and_records_sidecar(tmp_path, fakes):
    import json

    test, _ = split_test_fewshot(_fake_items(), fewshot_n=10, seed=42)
    ex = tmp_path / "ex.csv"
    _write_excl(ex, [[test[0]["id"], "gov"]])
    run_mod.main(["--model", "himed", "--condition", "c", "--exclude-file", str(ex),
                  "--output-dir", str(tmp_path)])
    rows = list(csv.DictReader(open(tmp_path / "himed_c_raw.csv", encoding="utf-8")))
    assert [r["item_id"] for r in rows] == [t["id"] for t in test[1:]]
    cfg = json.loads((tmp_path / "himed_c_raw.csv.config.json").read_text(encoding="utf-8"))
    assert cfg["exclude_file"] == "ex.csv" and len(cfg["exclude_sha256"]) == 64


def test_run_without_exclude_file_sidecar_is_none(tmp_path, fakes):
    import json

    run_mod.main(["--model", "himed", "--condition", "c", "--output-dir", str(tmp_path)])
    cfg = json.loads((tmp_path / "himed_c_raw.csv.config.json").read_text(encoding="utf-8"))
    assert cfg["exclude_file"] is None and cfg["exclude_sha256"] is None


def test_run_exclude_file_allows_passages_for_excluded_items(tmp_path, fakes, monkeypatch):
    test, _ = split_test_fewshot(_fake_items(), fewshot_n=10, seed=42)
    ex = tmp_path / "ex.csv"
    _write_excl(ex, [[test[0]["id"], "gov"]])
    seen = {}

    def fake_load_passages(path, item_ids, allow_extra=False):
        seen["allow_extra"] = allow_extra
        return {i: [] for i in item_ids}

    monkeypatch.setattr(run_mod, "load_passages", fake_load_passages)
    monkeypatch.setattr(run_mod, "rag_sidecar_fields", lambda p: {})
    run_mod.main(["--model", "himed", "--condition", "c", "--exclude-file", str(ex),
                  "--passages-file", str(tmp_path / "p.jsonl"), "--output-dir", str(tmp_path)])
    assert seen["allow_extra"] is True
