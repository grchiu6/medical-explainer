from models.nanda import _merge_gen_kwargs, _DEFAULT_GEN_KWARGS

def test_merge_gen_kwargs_none_returns_defaults():
    assert _merge_gen_kwargs(None) == _DEFAULT_GEN_KWARGS

def test_merge_gen_kwargs_overrides_only_given_keys():
    merged = _merge_gen_kwargs({"do_sample": False})
    assert merged["do_sample"] is False
    assert merged["temperature"] == _DEFAULT_GEN_KWARGS["temperature"]
