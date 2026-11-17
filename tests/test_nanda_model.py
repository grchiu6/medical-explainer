import torch

import models.nanda as nanda
from models.nanda import _merge_gen_kwargs, _DEFAULT_GEN_KWARGS

def test_merge_gen_kwargs_none_returns_defaults():
    assert _merge_gen_kwargs(None) == _DEFAULT_GEN_KWARGS

def test_merge_gen_kwargs_overrides_only_given_keys():
    merged = _merge_gen_kwargs({"do_sample": False})
    assert merged["do_sample"] is False
    assert merged["temperature"] == _DEFAULT_GEN_KWARGS["temperature"]


# --- Prompt-template regression (2026-11-16) --------------------------------
# The old hand-built template had a doubled <|begin_of_text|> and no "\n\n"
# after each <|end_header_id|>. Under greedy decoding Nanda's first token was
# <|eot_id|> on every MILU item (489/489 empty answers). The official chat
# template produced real answers on the same items (PACE diagnostic job 7123800).

class _FakeEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.template_call = None
        self.encode_call = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = (messages, kwargs)
        return "<TEMPLATED PROMPT>"

    def __call__(self, text, **kwargs):
        self.encode_call = (text, kwargs)
        return _FakeEncoding(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens=True):
        return " C "


class _FakeModel:
    def parameters(self):
        yield torch.zeros(1)

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 34]])


def test_format_uses_official_chat_template(monkeypatch):
    fake = _FakeTokenizer()
    monkeypatch.setattr(nanda, "_tokenizer", fake)

    out = nanda._format_for_nanda("PROMPT")

    assert out == "<TEMPLATED PROMPT>"
    messages, kwargs = fake.template_call
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "PROMPT"
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is False


def test_get_answer_does_not_add_a_second_bos(monkeypatch):
    fake = _FakeTokenizer()
    monkeypatch.setattr(nanda, "_load_model", lambda: None)
    monkeypatch.setattr(nanda, "_tokenizer", fake)
    monkeypatch.setattr(nanda, "_model", _FakeModel())

    answer = nanda.get_answer("PROMPT", gen_kwargs={"do_sample": False})

    text, kwargs = fake.encode_call
    assert text == "<TEMPLATED PROMPT>"
    # The chat template already contains <|begin_of_text|>; the tokenizer must not add another.
    assert kwargs.get("add_special_tokens") is False
    assert answer == "C"
