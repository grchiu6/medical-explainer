import pytest

from performance_eval.rag.corpus import (
    build_corpus,
    load_statpearls,
    read_corpus,
    textbook_row_to_passage,
)


def test_statpearls_reader_reads_medrag_jsonl_fields(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"id":"x_0","title":"T","content":"C","contents":"T. C"}\n', encoding="utf-8")
    rows = load_statpearls(tmp_path)
    assert rows == [{"passage_id": "statpearls:x_0", "source": "statpearls", "title": "T", "content": "C", "contents": "T. C"}]


def test_statpearls_reads_files_in_sorted_order(tmp_path):
    (tmp_path / "b.jsonl").write_text('{"id":"b_0","title":"B","content":"b","contents":"B. b"}\n', encoding="utf-8")
    (tmp_path / "a.jsonl").write_text('{"id":"a_0","title":"A","content":"a","contents":"A. a"}\n', encoding="utf-8")
    assert [r["passage_id"] for r in load_statpearls(tmp_path)] == ["statpearls:a_0", "statpearls:b_0"]


def test_statpearls_missing_or_empty_dir_raises(tmp_path):
    with pytest.raises(SystemExit):
        load_statpearls(tmp_path / "nope")
    with pytest.raises(SystemExit):
        load_statpearls(tmp_path)  # exists, no *.jsonl


def test_textbook_row_to_passage_maps_fields():
    row = {"id": "Anatomy_Gray_0", "title": "Anatomy", "content": "C", "contents": "Anatomy. C"}
    assert textbook_row_to_passage(row) == {
        "passage_id": "textbooks:Anatomy_Gray_0",
        "source": "textbooks",
        "title": "Anatomy",
        "content": "C",
        "contents": "Anatomy. C",
    }


def test_build_corpus_prefixes_ids_and_rejects_duplicates(tmp_path):
    tb = [{"passage_id": "textbooks:1", "source": "textbooks", "title": "a", "content": "b", "contents": "a. b"}]
    sp = [{"passage_id": "statpearls:1", "source": "statpearls", "title": "c", "content": "d", "contents": "c. d"}]
    assert build_corpus(tb, sp, tmp_path / "c.jsonl") == 2
    with pytest.raises(ValueError):
        build_corpus(tb, tb, tmp_path / "d.jsonl")


def test_build_then_read_round_trips(tmp_path):
    tb = [{"passage_id": "textbooks:1", "source": "textbooks", "title": "हिन्दी", "content": "b", "contents": "हिन्दी. b"}]
    sp = [{"passage_id": "statpearls:1", "source": "statpearls", "title": "c", "content": "d", "contents": "c. d"}]
    out = tmp_path / "sub" / "c.jsonl"
    build_corpus(tb, sp, out)
    assert read_corpus(out) == tb + sp


def test_build_corpus_duplicate_detection_uses_counter_and_names_dupes(tmp_path):
    def row(i):
        return {"passage_id": f"textbooks:{i}", "source": "textbooks", "title": "t",
                "content": "c", "contents": "c"}
    rows = [row(i) for i in range(5)] + [row(2), row(2), row(4)]
    with pytest.raises(ValueError, match="textbooks:2") as e:
        build_corpus(rows, [], tmp_path / "c.jsonl")
    assert "textbooks:4" in str(e.value)
