from models.himed import _merge_gen_kwargs, _DEFAULT_GEN_KWARGS

def test_merge_gen_kwargs_none_returns_defaults():
    assert _merge_gen_kwargs(None) == _DEFAULT_GEN_KWARGS

def test_merge_gen_kwargs_overrides_only_given_keys():
    merged = _merge_gen_kwargs({"do_sample": False})
    assert merged["do_sample"] is False
    assert merged["temperature"] == _DEFAULT_GEN_KWARGS["temperature"]


class _FakeTok:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return messages


def test_format_default_system_unchanged(monkeypatch):
    import models.himed as h
    monkeypatch.setattr(h, "_tokenizer", _FakeTok())
    msgs = h._format_for_himed("P")
    assert msgs[0]["content"] == "आप एक सहायक चिकित्सा शिक्षक हैं। दिए गए निर्देशों का पालन करते हुए हिंदी में उत्तर दें।"
    assert h._format_for_himed("P", system_prompt=None) == msgs


def test_format_custom_system(monkeypatch):
    import models.himed as h
    monkeypatch.setattr(h, "_tokenizer", _FakeTok())
    msgs = h._format_for_himed("P", system_prompt="S")
    assert msgs == [{"role": "system", "content": "S"}, {"role": "user", "content": "P"}]
