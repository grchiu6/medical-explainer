import csv

import pytest

from performance_eval.rag import tagging
from performance_eval.rag.tagging import (
    TAGS, agreement, build_tag_prompt, parse_tag, run_tagging, write_handcheck)

ITEM = {"id": "milu_hi_x", "question": "q", "options": {"A": "a", "B": "b"},
        "question_with_options": "बुखार का इलाज क्या है?\nA. पैरासिटामोल\nB. कुछ नहीं",
        "answer_letter": "B", "n_options": 2}


def test_prompt_contains_question_and_no_leaks():
    p = build_tag_prompt(ITEM)
    assert ITEM["question_with_options"] in p
    low = p.lower()
    for bad in ("correct", "answer key", "rag", "answer_letter", "reference_answer", "model_answer"):
        assert bad not in low
    for t in TAGS:
        assert t in p


def test_parse_tag_cases():
    assert parse_tag("Clinical") == "clinical"
    assert parse_tag("The tag is AYUSH.") == "ayush"
    assert parse_tag("public health") == "public_health"
    assert parse_tag("Public_Health") == "public_health"
    assert parse_tag("## Thinking\nmaybe clinical\n## Response\nother") == "other"
    assert parse_tag("clinical then ayush") == "clinical"
    assert parse_tag("ayush, clinical") == "ayush"
    assert parse_tag("nonclinical") is None
    assert parse_tag("garbage") is None
    assert parse_tag("") is None
    assert parse_tag("## Thinking\nclinical clinical") is None  # no Response


def test_handcheck_columns_and_blindness(tmp_path):
    items = [{**ITEM, "id": f"id{i:03d}", "answer_letter": "C"} for i in range(80)]
    out = tmp_path / "s.csv"
    write_handcheck(out, items)
    with open(out, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    assert header == ["item_id", "question_with_options", "tag"]
    assert len(rows) == 50 and all(row[2] == "" for row in rows)
    assert len({row[0] for row in rows}) == 50
    for bad in ("answer_letter", "reference_answer", "model_answer"):
        assert all(bad not in c for row in rows for c in row) and bad not in header
    # deterministic regardless of input order
    out2 = tmp_path / "s2.csv"
    write_handcheck(out2, list(reversed(items)))
    assert out.read_bytes() == out2.read_bytes()


def test_agreement_perfect():
    m = {"a": "clinical", "b": "ayush", "c": "other"}
    r = agreement(m, dict(m))
    assert r == {"n": 3, "percent_agree": 100.0, "cohens_kappa": 1.0}


def test_agreement_mixed_hand_computed():
    m = {str(i): t for i, t in enumerate(["clinical"] * 5 + ["other"] * 5)}
    h = {str(i): t for i, t in enumerate(["clinical"] * 4 + ["other"] * 1 + ["other"] * 4 + ["clinical"] * 1)}
    # p_o = 8/10; p_e = .5*.5 + .5*.5 = .5; kappa = .6
    r = agreement(m, h)
    assert r["n"] == 10 and r["percent_agree"] == 80.0
    assert r["cohens_kappa"] == pytest.approx(0.6)


def test_agreement_degenerate_and_overlap():
    assert agreement({"a": "other", "b": "other"}, {"a": "other", "b": "other", "z": "x"})["cohens_kappa"] == 1.0
    assert agreement({"a": "other", "b": "other"}, {"a": "other", "b": "clinical"})["cohens_kappa"] == 0.0
    assert agreement({"a": "other"}, {"b": "other"})["n"] == 0


def test_agreement_untagged_is_category():
    r = agreement({"a": "untagged", "b": "other"}, {"a": "other", "b": "other"})
    assert r["percent_agree"] == 50.0


def test_run_tagging_resume_and_untagged(tmp_path):
    items = [{**ITEM, "id": f"i{i}"} for i in range(3)]
    outs = iter(["clinical", "blah", "ayush"])
    prompts = []

    def fake(prompt, gen_kwargs=None):
        prompts.append(prompt)
        return next(outs)

    out = tmp_path / "question_types.csv"
    run_tagging(items[:2], fake, out)
    run_tagging(items, fake, out)
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["item_id"], r["tag"]) for r in rows] == [("i0", "clinical"), ("i1", "untagged"), ("i2", "ayush")]
    raw = tmp_path / "question_types_raw.csv"
    with open(raw, encoding="utf-8", newline="") as f:
        rr = list(csv.DictReader(f))
    assert [r["model_output"] for r in rr] == ["clinical", "blah", "ayush"]
    assert len(prompts) == 3


def test_read_hand_tags_validation(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("item_id,question_with_options,tag\na,q,Clinical\nb,q,\n", encoding="utf-8-sig")
    with pytest.raises(SystemExit, match="b"):
        tagging.read_tags(p, strict=True)
    p.write_text("item_id,question_with_options,tag\na,q,bogus\n", encoding="utf-8-sig")
    with pytest.raises(SystemExit, match="bogus"):
        tagging.read_tags(p, strict=True)
