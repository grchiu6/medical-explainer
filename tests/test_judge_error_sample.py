import pandas as pd

from literacy_eval.judge import error_sample as E


def _llm(n_per_model=4):
    rows = []
    for task in ("complexity", "reference_agreement"):
        for model in ("a", "b", "c"):
            for i in range(n_per_model):
                rows.append({"task": task, "model": model, "level": "low", "item_id": f"{model}{i}",
                             "score": 1 if i < 3 else 5})
    return pd.DataFrame(rows)


def test_draw_is_stratified_reproducible_and_weighted():
    llm = _llm()
    s1, s2 = E.draw(llm, per_stratum=3, seed=1), E.draw(llm, per_stratum=3, seed=1)
    assert s1.equals(s2)
    g = s1.groupby(["task", "score"]).size()
    assert g[("complexity", 1)] == 3 and g[("complexity", 5)] == 3      # 9 and 3 available
    one = s1[(s1.task == "complexity") & (s1.score == 1)]
    assert sorted(one.model) == ["a", "b", "c"]                         # round-robin over models
    assert set(one.weight) == {3.0} and set(one.stratum_n) == {9}


def test_review_sheet_formulas_point_at_their_hindi_cell():
    s = pd.DataFrame({"job_id": ["j"], "task": ["complexity"], "model": ["a"], "level": ["low"], "item_id": ["x"],
                      "failure_type": ["ok"], "stratum_n": [1], "weight": [1.0], "score": [2], "feedback": ["f"],
                      "question_hi": ["प्रश्न"], "answer_hi": ["उत्तर"], "reference_hi": [""]})
    sheet = E.review_sheet(s)
    col = E._col_letter(list(sheet.columns).index("answer_hi"))
    assert sheet.loc[0, "answer_en"] == f'=GOOGLETRANSLATE({col}2, "hi", "en")'
    assert sheet.loc[0, "reference_en"] == ""                           # nothing to translate
    assert all(c in sheet for c in E.REVIEW_COLUMNS)


def test_analyze_reweights_to_the_population():
    sheet = pd.DataFrame({"task": ["t", "t"], "model": ["a", "b"], "score": ["1", "5"], "weight": ["9", "1"],
                          "score_verdict": ["too_low", "correct"], "feedback_faithful": ["y", "y"],
                          "judge_error": ["J2", "J0"], "answer_error": ["A1", "A0"], "human_score": ["2", "5"]})
    out = E.analyze(sheet)
    assert out["verdict_by_task"].loc["t", "too_low"] == 90.0
    assert out["score_agreement_by_task"].loc["t", "exact"] == 50.0
