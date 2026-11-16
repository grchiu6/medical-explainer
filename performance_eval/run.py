"""Run a model on MILU Hindi medical MCQs under one prompting condition.

Writes ``{output-dir}/{model}_{condition}_raw.csv`` with columns
item_id, question_with_options, reference_answer, model_answer. Resume is keyed
on item_id (prompt text differs per condition, item_id does not). A sidecar
``{csv}.config.json`` records the run configuration; resuming under a different
configuration is refused.
"""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from performance_eval.data_milu import (
    load_milu_hindi_medicine,
    pick_exemplars,
    shuffle_item_options,
    split_test_fewshot,
)
from performance_eval.prompts import build_prompt

FIELDNAMES = ["item_id", "question_with_options", "reference_answer", "model_answer"]
# Keys allowed to differ on resume: a full run may continue a limited run's rows.
_CONFIG_EXEMPT_KEYS = {"limit"}


def _load_model_fn(name: str) -> Callable[..., str]:
    """Import models.<name> lazily (it imports torch) and return its get_answer."""
    return importlib.import_module(f"models.{name}").get_answer


def _set_seed(seed: int) -> None:
    import transformers  # lazy: unit tests don't need it

    transformers.set_seed(seed)


def config_sidecar_path(output_path: Path) -> Path:
    return Path(f"{output_path}.config.json")


def check_or_write_config(output_path: Path, config: dict) -> None:
    """Write the config sidecar for a fresh CSV; on resume, refuse a config mismatch.

    Raises SystemExit if the CSV has content but the sidecar is missing, or if
    any non-exempt key differs.
    """
    output_path = Path(output_path)
    sidecar = config_sidecar_path(output_path)
    config = json.loads(json.dumps(config))  # normalize (tuples -> lists, etc.)
    fix = "use a new --condition label or delete the CSV and its .config.json"
    if output_path.exists() and output_path.stat().st_size > 0:
        if not sidecar.is_file():
            raise SystemExit(f"{output_path} has rows but {sidecar.name} is missing, so the "
                             f"run configuration cannot be checked; {fix}")
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        differing = sorted(k for k in set(saved) | set(config)
                           if k not in _CONFIG_EXEMPT_KEYS and saved.get(k) != config.get(k))
        if differing:
            details = "; ".join(f"{k}: saved={saved.get(k)!r} now={config.get(k)!r}" for k in differing)
            raise SystemExit(f"Refusing to resume {output_path}: config differs ({details}); {fix}")
        return
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_gen_kwargs(
    sample_decoding: bool,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
) -> dict:
    """Greedy by default; None values unset the model's lingering sampling defaults and
    repetition_penalty=1.0 disables the model's penalty (HF also applies it to prompt
    tokens). With sample_decoding, start from {} so the model's defaults apply."""
    kwargs: dict = {} if sample_decoding else {
        "do_sample": False, "temperature": None, "top_p": None, "repetition_penalty": 1.0}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    return kwargs


def validate_args(
    sample_decoding: bool, temperature: float | None = None, top_p: float | None = None, **_
) -> None:
    """temperature/top_p are meaningless under greedy decoding."""
    if not sample_decoding and (temperature is not None or top_p is not None):
        raise ValueError("--temperature/--top-p require --sample-decoding")


def run_condition(
    items: list[dict],
    fewshot_pool: list[dict],
    get_answer_fn: Callable[..., str],
    use_cot: bool,
    use_fewshot: bool,
    gen_kwargs: dict,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    has_content = output_path.exists() and output_path.stat().st_size > 0
    if has_content:
        with open(output_path, encoding="utf-8", newline="") as f:
            done = {row["item_id"] for row in csv.DictReader(f)}
    remaining = [it for it in items if it["id"] not in done]
    print(f"{len(done)} done, {len(remaining)} remaining")

    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not has_content:
            writer.writeheader()
        for i, item in enumerate(remaining, 1):
            prompt = build_prompt(
                item["question_with_options"],
                item["n_options"],
                use_cot=use_cot,
                exemplars=fewshot_pool if use_fewshot else None,
            )
            answer = get_answer_fn(prompt, gen_kwargs=gen_kwargs)
            writer.writerow({
                "item_id": item["id"],
                "question_with_options": item["question_with_options"],
                "reference_answer": item["answer_letter"],
                "model_answer": answer,
            })
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(remaining)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a model on a MILU MCQ condition")
    parser.add_argument("--model", required=True, choices=["himed", "nanda"])
    parser.add_argument("--condition", required=True, help="label used in the output filename, e.g. c0_baseline")
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--use-fewshot", action="store_true")
    parser.add_argument("--fewshot-n", type=int, default=10, help="size of the held-out few-shot pool")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="override the model's default max_new_tokens (needed for CoT)")
    parser.add_argument("--sample-decoding", action="store_true",
                        help="C4 only: use the model's default sampling instead of greedy")
    parser.add_argument("--temperature", type=float, default=None, help="requires --sample-decoding")
    parser.add_argument("--top-p", type=float, default=None, help="requires --sample-decoding")
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="data seed (split/exemplars/shuffle) — keep fixed across all conditions")
    parser.add_argument("--gen-seed", type=int, default=42,
                        help="generation seed (transformers.set_seed only); vary it for a noise floor")
    parser.add_argument("--shuffle-options", action="store_true",
                        help="shuffle option order of test items (position-bias check); exemplars stay native")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N test items")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "milu_hi_med")
    args = parser.parse_args(argv)

    try:
        validate_args(args.sample_decoding, temperature=args.temperature, top_p=args.top_p)
    except ValueError as e:
        parser.error(str(e))

    gen_kwargs = build_gen_kwargs(
        args.sample_decoding,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    load_dotenv()
    items = load_milu_hindi_medicine(hf_token=os.environ.get("HF_TOKEN"))
    test, pool = split_test_fewshot(items, fewshot_n=args.fewshot_n, seed=args.seed)
    exemplars = pick_exemplars(pool, seed=args.seed)
    if args.shuffle_options:
        test = [shuffle_item_options(it, seed=args.seed) for it in test]
    if args.limit is not None:
        test = test[: args.limit]

    output_path = args.output_dir / f"{args.model}_{args.condition}_raw.csv"
    check_or_write_config(output_path, {
        "model": args.model,
        "condition": args.condition,
        "use_cot": args.use_cot,
        "use_fewshot": args.use_fewshot,
        "fewshot_n": args.fewshot_n,
        "gen_kwargs": gen_kwargs,
        "seed": args.seed,
        "gen_seed": args.gen_seed,
        "shuffle_options": args.shuffle_options,
        "limit": args.limit,
        "exemplar_ids": [ex["id"] for ex in exemplars] if args.use_fewshot else [],
    })

    _set_seed(args.gen_seed)
    get_answer_fn = _load_model_fn(args.model)
    run_condition(test, exemplars, get_answer_fn, args.use_cot, args.use_fewshot, gen_kwargs, output_path)


if __name__ == "__main__":
    main()
