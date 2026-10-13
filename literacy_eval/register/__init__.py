"""Register classifier: build a 5th-grade / 10th-grade / professional Hindi corpus and score LLM outputs with it.

Stages (each a `python -m literacy_eval.register.<stage>` CLI):
    sources -> ocr -> build_corpus (clean_sources + normalize + chunk) -> prepare_outputs

See literacy_eval/docs/2026-10-12-register-classifier-plan.md for the design.
"""
