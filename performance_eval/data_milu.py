"""MILU Hindi "Health and Medicine" loader plus frozen test/few-shot splits.

MILU (ai4bharat/MILU, gated) config "Hindi", split "test" has columns
question, option1..option4, target ("optionN"), language, domain, subject.
There is no id column, so ids are a content hash (see _make_id).
"""

import hashlib
import random

from datasets import load_dataset

_SUBJECT = "Health and Medicine"
_LETTERS = "ABCDEFGHIJ"


def _clean(text: str) -> str:
    """Collapse whitespace (incl. newlines) so an option never starts a new line."""
    return " ".join(str(text).split())


def _make_id(question: str, option_texts: list[str]) -> str:
    """Stable id: sha1 of question + options. A collision means two rows have
    identical question and option text, i.e. a true duplicate item."""
    payload = "\n".join([question] + option_texts)
    return "milu_hi_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _format_qwo(question: str, options: dict[str, str]) -> str:
    return question + "\n" + "\n".join(f"{l}. {options[l]}" for l in sorted(options))


def load_milu_hindi_medicine(hf_token: str | None = None) -> list[dict]:
    ds = load_dataset("ai4bharat/MILU", "Hindi", split="test", token=hf_token)
    items, seen = [], set()
    for row in ds:
        if row.get("subject") != _SUBJECT:
            continue
        keys = sorted(k for k in row if k.startswith("option") and row[k] is not None)
        raw = [row[k] for k in keys]
        options = {_LETTERS[i]: _clean(t) for i, t in enumerate(raw)}
        target = row["target"]
        if target not in keys:
            raise ValueError(f"target {target!r} not among option fields {keys}")
        question = row["question"]
        item_id = _make_id(question, [options[l] for l in options])
        if item_id in seen:
            raise ValueError(f"Duplicate item id {item_id}")
        seen.add(item_id)
        items.append({
            "id": item_id,
            "question": question,
            "options": options,
            "question_with_options": _format_qwo(question, options),
            "answer_letter": _LETTERS[keys.index(target)],
            "n_options": len(options),
        })
    return items


def split_test_fewshot(
    items: list[dict], fewshot_n: int = 10, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Return (test_items, fewshot_pool), disjoint and deterministic."""
    rng = random.Random(seed)
    pool_idx = set(rng.sample(range(len(items)), fewshot_n))
    pool = [it for i, it in enumerate(items) if i in pool_idx]
    test = [it for i, it in enumerate(items) if i not in pool_idx]
    return test, pool


def pick_exemplars(pool: list[dict], k: int = 3, seed: int = 42) -> list[dict]:
    """Deterministically pick k items, preferring distinct answer letters."""
    rng = random.Random(seed)
    order = list(pool)
    rng.shuffle(order)
    chosen, letters = [], set()
    for it in order:
        if len(chosen) == k:
            break
        if it["answer_letter"] not in letters:
            chosen.append(it)
            letters.add(it["answer_letter"])
    for it in order:
        if len(chosen) == k:
            break
        if not any(it is c for c in chosen):
            chosen.append(it)
    return chosen


def shuffle_item_options(item: dict, seed: int) -> dict:
    """New item with options permuted by seed and answer_letter remapped."""
    rng = random.Random(f"{seed}:{item['id']}")
    letters = list(item["options"])
    texts = [item["options"][l] for l in letters]
    correct_text = item["options"][item["answer_letter"]]
    rng.shuffle(texts)
    options = dict(zip(letters, texts))
    answer = next(l for l, t in options.items() if t == correct_text)
    out = dict(item)
    out["options"] = options
    out["answer_letter"] = answer
    out["question_with_options"] = _format_qwo(item["question"], options)
    return out


def load_english_items(translations_path, hindi_items: list[dict], drop_ids=frozenset()) -> list[dict]:
    """English versions of hindi_items from a translations JSONL (item_id, question, options).

    Keeps id, answer_letter and n_options, so English and Hindi runs are paired by item_id.
    Items in drop_ids are skipped; any other item without a translation is an error."""
    return load_mixed_items(translations_path, hindi_items, "en", "en", drop_ids=drop_ids)


_ITEM_LANGS = ("hi", "en")


def load_mixed_items(translations_path, hindi_items: list[dict], question_lang: str, options_lang: str,
                     drop_ids=frozenset()) -> list[dict]:
    """Items whose stem and options each come from Hindi (the item itself) or English (the translation).

    Same validation as the English loader: missing / hard-flagged / option-key-mismatched translations
    raise ValueError even when only one part is English, so every language setup covers the same items.
    English parts pass through _clean; Hindi parts are the item's own question/options. With
    ("hi", "hi") the result equals the original question_with_options byte-for-byte."""
    import json
    from pathlib import Path

    if question_lang not in _ITEM_LANGS or options_lang not in _ITEM_LANGS:
        raise ValueError(f"question_lang/options_lang must be in {_ITEM_LANGS}, got {question_lang!r}/{options_lang!r}")
    recs = {}
    for line in Path(translations_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["item_id"]] = r
    kept = [it for it in hindi_items if it["id"] not in drop_ids]
    missing = [it["id"] for it in kept if it["id"] not in recs]
    if missing:
        raise ValueError(f"{len(missing)} item(s) have no translation, e.g. {missing[:5]}")
    hard = {"option_count_mismatch", "empty_option", "bad_json"}
    bad = [it["id"] for it in kept
           if hard & set(recs[it["id"]].get("flags", []))
           or sorted(recs[it["id"]].get("options", {})) != sorted(it["options"])]
    if bad:
        raise ValueError(f"{len(bad)} translation(s) are structurally broken, e.g. {bad[:5]}; drop them with "
                         "--exclude-file performance_eval/data/bad_translation_items.csv")
    out = []
    for it in kept:
        r = recs[it["id"]]
        question = r["question"] if question_lang == "en" else it["question"]
        options = ({l: _clean(r["options"][l]) for l in sorted(it["options"])} if options_lang == "en"
                   else dict(it["options"]))
        out.append({**it, "question": question, "options": options,
                    "question_with_options": _format_qwo(question, options)})
    return out
