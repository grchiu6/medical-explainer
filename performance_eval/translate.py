"""Hindi -> English translation of MILU test items (no human validator; see plan D5).

Primary: Google Cloud Translation v2 (not one of the compared models, so it cannot
favour one of them). Secondary: Gemini 3.8 Flash. Conclusions must hold under both.
Only question + option texts are sent; gold answers never leave this machine.

CLI:
  python -m performance_eval.translate translate --engine google|gemini
  python -m performance_eval.translate report      # flags, agreement, bad-item list
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data" / "milu_en"  # tracked: MT output drifts over time
DATA_DIR = Path(__file__).parent / "data"
GOOGLE_URL = "https://translation.googleapis.com/language/translate/v2"
# Structural problems make an item unusable; the rest are reported only.
HARD_FLAGS = ("option_count_mismatch", "empty_option", "bad_json")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_NUMBERED = re.compile(r"(?:^|\n)\s*\(?\d\)?[.)]", re.MULTILINE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _devanagari_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(bool(_DEVANAGARI.match(c)) for c in letters) / len(letters) if letters else 0.0


def check_translation(hindi_item: dict, tr: dict) -> list[str]:
    """Flags for one translation {question, options{letter: text}} against its Hindi item."""
    flags = []
    if sorted(tr.get("options", {})) != sorted(hindi_item["options"]):
        flags.append("option_count_mismatch")
    if any(not str(v).strip() for v in tr.get("options", {}).values()) or not tr.get("question", "").strip():
        flags.append("empty_option")
    fields = [tr.get("question", ""), *map(str, tr.get("options", {}).values())]
    if any(_devanagari_share(f) > 0.2 for f in fields):
        flags.append("devanagari_left")
    if _NUMBERED.search(hindi_item["question"]) and not _NUMBERED.search(tr.get("question", "")):
        flags.append("numbering_lost")
    return flags


def translate_google(items: list[dict], api_key: str, post=None) -> list[dict]:
    """One request per item: q = [question, *options in letter order]."""
    if post is None:
        import requests

        post = requests.post
    out = []
    for it in items:
        letters = sorted(it["options"])
        q = [it["question"], *(it["options"][l] for l in letters)]
        resp = post(GOOGLE_URL, params={"key": api_key},
                    data={"q": q, "source": "hi", "target": "en", "format": "text"}, timeout=60)
        resp.raise_for_status()
        texts = [t["translatedText"] for t in resp.json()["data"]["translations"]]
        tr = {"question": texts[0], "options": dict(zip(letters, texts[1:]))}
        out.append({"item_id": it["id"], **tr, "flags": check_translation(it, tr), "raw": texts})
    return out


_PROMPT = """Translate this Hindi multiple-choice medical exam question into English.
Rules:
- Translate faithfully. Do not answer, explain, correct, or add information.
- Keep every option, in the same letters. Keep numbered statements (1., 2., ...) as separate numbered lines.
- Keep drug, disease and medicine names; transliterate AYUSH/traditional terms that have no English equivalent.
Return only JSON: {{"question": "...", "options": {{{option_keys}}}}}

Question:
{question}

Options:
{options}
"""


def build_translate_prompt(item: dict) -> str:
    letters = sorted(item["options"])
    return _PROMPT.format(
        option_keys=", ".join(f'"{l}": "..."' for l in letters),
        question=item["question"],
        options="\n".join(f"{l}. {item['options'][l]}" for l in letters),
    )


def parse_translation(text: str, n_options: int) -> tuple[dict | None, list[str]]:
    try:
        obj = json.loads(_FENCE.sub("", (text or "").strip()))
        tr = {"question": str(obj["question"]), "options": {str(k): str(v) for k, v in obj["options"].items()}}
    except (ValueError, KeyError, TypeError, AttributeError):
        return None, ["bad_json"]
    return tr, []


def translate_gemini(items: list[dict], get_answer) -> list[dict]:
    out = []
    for it in items:
        raw = get_answer(build_translate_prompt(it), gen_kwargs={})
        tr, flags = parse_translation(raw, it["n_options"])
        if tr is None:
            tr = {"question": "", "options": {}}
        else:
            flags = check_translation(it, tr)
        out.append({"item_id": it["id"], **tr, "flags": flags, "raw": raw})
    return out


def _ngrams(text: str, n: int) -> Counter:
    s = " ".join(text.lower().split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(a: str, b: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popovic 2015), averaged over n = 1..max_n."""
    scores = []
    for n in range(1, max_n + 1):
        ca, cb = _ngrams(a, n), _ngrams(b, n)
        if not ca or not cb:
            continue
        overlap = sum((ca & cb).values())
        p, r = overlap / sum(cb.values()), overlap / sum(ca.values())
        scores.append(0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r))
    return sum(scores) / len(scores) if scores else 0.0


def _flat(tr: dict) -> str:
    return tr.get("question", "") + "\n" + "\n".join(tr.get("options", {})[l] for l in sorted(tr.get("options", {})))


def agreement(primary: dict[str, dict], secondary: dict[str, dict]) -> dict[str, float]:
    return {iid: chrf(_flat(primary[iid]), _flat(secondary[iid])) for iid in primary if iid in secondary}


def write_bad_translations(records: list[dict], path: Path, secondary: list[dict] | None = None) -> None:
    """Union of structurally broken items across engines, so every English run drops the same ids."""
    reasons: dict[str, list[str]] = {}
    for engine, recs in (("google", records), ("gemini", secondary or [])):
        for r in recs:
            hard = [fl for fl in r["flags"] if fl in HARD_FLAGS]
            if hard:
                reasons.setdefault(r["item_id"], []).append(f"{engine}:{'+'.join(hard)}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["item_id", "reason"])
        for iid, rs in reasons.items():
            w.writerow([iid, ";".join(rs)])


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, recs: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _test_items(exclude_file: Path) -> list[dict]:
    from dotenv import load_dotenv

    from performance_eval.data_milu import load_milu_hindi_medicine, split_test_fewshot
    from performance_eval.exclusions import apply_exclusions, load_exclusions

    load_dotenv(".env")
    items = load_milu_hindi_medicine(hf_token=os.environ.get("HF_TOKEN"))
    test, _ = split_test_fewshot(items, fewshot_n=10, seed=42)
    return apply_exclusions(test, load_exclusions(exclude_file))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("translate")
    t.add_argument("--engine", choices=["google", "gemini"], required=True)
    t.add_argument("--exclude-file", type=Path, default=DATA_DIR / "excluded_items.csv")
    t.add_argument("--out-dir", type=Path, default=OUT_DIR)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "translate":
        out = args.out_dir / f"translations_{args.engine}.jsonl"
        done = {r["item_id"] for r in read_jsonl(out)}
        todo = [it for it in _test_items(args.exclude_file) if it["id"] not in done]
        print(f"{len(done)} done, {len(todo)} remaining")
        for i, it in enumerate(todo, 1):  # one item per write so a crash loses nothing
            if args.engine == "google":
                recs = translate_google([it], os.environ["GOOGLE_TRANSLATE_API_KEY"])
            else:
                from models import gemini_flash

                usage = out.with_suffix(".usage.json")
                gemini_flash.load_usage(usage)
                try:
                    recs = translate_gemini([it], gemini_flash.get_answer)
                finally:
                    gemini_flash.save_usage(usage)
            _append_jsonl(out, recs)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
        return

    prim = {r["item_id"]: r for r in read_jsonl(args.out_dir / "translations_google.jsonl")}
    sec = {r["item_id"]: r for r in read_jsonl(args.out_dir / "translations_gemini.jsonl")}
    write_bad_translations(list(prim.values()), DATA_DIR / "bad_translation_items.csv", list(sec.values()))
    agree = agreement(prim, sec)
    lines = ["# Translation report", "", f"Primary (Google): {len(prim)} items; secondary (Gemini): {len(sec)} items.", ""]
    for name, recs in (("google", prim), ("gemini", sec)):
        c = Counter(fl for r in recs.values() for fl in r["flags"])
        lines.append(f"- {name} flags: {dict(c) or 'none'}")
    if agree:
        vals = sorted(agree.values())
        lines += ["", f"chrF agreement Google vs Gemini: median {vals[len(vals) // 2]:.3f}, min {vals[0]:.3f}", "",
                  "Lowest-agreement 10% (read these first):", "", "| item_id | chrF | Google | Gemini |", "|---|---|---|---|"]
        for iid, s in sorted(agree.items(), key=lambda kv: kv[1])[: max(1, len(agree) // 10)]:
            g, m = (_flat(x[iid]).replace("\n", " / ").replace("|", "/") for x in (prim, sec))
            lines.append(f"| {iid} | {s:.3f} | {g} | {m} |")
    (args.out_dir / "translation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "translation_report.md")


if __name__ == "__main__":
    main()
