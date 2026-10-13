"""Score new text with an already-trained register classifier (no retraining).

WHAT THIS FILE DOES:
Loads a trained checkpoint (best.pt from train.py) and writes <input stem>_scores.jsonl for every
input jsonl (each row needs a "text" field). Every other field is copied through, plus
p_grade5 / p_grade10 / p_professional, expected_level (0 = 5th grade ... 2 = professional) and predicted.

Used for the outside validation sets (validation_sets.py), so the same trained models can be checked
on new data without training again.

    python -m literacy_eval.register.score --encoder indicbert \\
        --checkpoint <run dir>/best.pt --inputs a.jsonl b.jsonl --out-dir <run dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from literacy_eval.register.train import ENCODERS, OrdinalClassifier, score_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=320)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[args.encoder])
    model = OrdinalClassifier(ENCODERS[args.encoder]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        score_file(model, path, args.out_dir / f"{path.stem}_scores.jsonl", tokenizer, args, device)


if __name__ == "__main__":
    main()
