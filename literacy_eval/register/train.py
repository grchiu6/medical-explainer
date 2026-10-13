"""Stage 8 (PACE GPU): fine-tune a Hindi encoder as an ORDINAL register classifier, then score.

WHAT THIS FILE DOES:
  1. Fine-tunes an encoder (MuRIL or IndicBERTv2-MLM) on train.jsonl with an
     ordinal head, choosing the best epoch by macro-F1 on val.jsonl.
  2. Reports accuracy, macro-F1 and the confusion matrix on test.jsonl.
  3. Scores every window in llm_windows.jsonl (the 900 LLM outputs, usable
     ones only) and, if given, any extra jsonl with a "text" field
     (e.g. ReadMe++ Hindi for validation).

THE ORDINAL HEAD (why not plain 3-way softmax):
The classes are ordered: 5th < 10th < professional. The head produces ONE
score s per passage and learns two thresholds b1 < b2. Then
    P(level > 5th)  = sigmoid(s - b1)
    P(level > 10th) = sigmoid(s - b2)
and the loss is binary cross-entropy on those two questions (CORAL, Cao et
al. 2020). A 10th-grade passage predicted "professional" is penalised less
than one predicted "5th grade". The output also gives a continuous
    expected_level = P(>5th) + P(>10th)    in [0, 2]
so we can test whether low < mid < high for the same question, not just
exact class matches.

    python -m literacy_eval.register.train --encoder muril --corpus-dir <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ENCODERS = {
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "tiny": "hf-internal-testing/tiny-random-bert",   # smoke tests only
}
LABELS = ["grade5", "grade10", "professional"]   # ordered
LABEL_ID = {label: i for i, label in enumerate(LABELS)}
INIT_GAP = 3.0   # initial distance between the two ordinal thresholds (see OrdinalClassifier)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class OrdinalClassifier(nn.Module):
    """Encoder [CLS] -> one score -> two ordered thresholds (CORAL)."""

    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(0.1)
        self.score = nn.Linear(self.encoder.config.hidden_size, 1)
        # b1 = first threshold; gap is kept positive so b2 > b1 always.
        # The gap must start WIDE: with gap g, the best P(10th grade) any passage can get is
        # sigmoid(g/2) - sigmoid(-g/2), which is only 0.245 for g = 1 (never the winning class) and
        # needs g > 1.39 to exceed 1/3. Start at -1.5 and +1.5 (gap 3, best possible 0.635).
        self.b1 = nn.Parameter(torch.tensor(-INIT_GAP / 2))
        self.log_gap = nn.Parameter(torch.tensor(math.log(INIT_GAP)))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        s = self.score(self.dropout(hidden)).squeeze(-1)
        thresholds = torch.stack([self.b1, self.b1 + torch.exp(self.log_gap)])
        return s.unsqueeze(-1) - thresholds          # logits of P(y > k), k = 0, 1


def build_optimizer(model: OrdinalClassifier, lr: float, head_lr: float) -> torch.optim.Optimizer:
    """AdamW with two learning rates: `lr` (small) for the pretrained encoder, `head_lr` (much larger) for the
    new score layer and the two thresholds. Those start from scratch and must be able to move far; at the
    encoder's 2e-5 they barely moved in the first training run (the thresholds stayed put, so the middle
    class could never be predicted). Thresholds get no weight decay (it would pull them together)."""
    head = [model.score.weight, model.score.bias]
    thresholds = [model.b1, model.log_gap]
    return torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr, "weight_decay": 0.01},
        {"params": head, "lr": head_lr, "weight_decay": 0.01},
        {"params": thresholds, "lr": head_lr, "weight_decay": 0.0},
    ])


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    """Label k -> [k > 0, k > 1] as floats."""
    return torch.stack([(y > 0), (y > 1)], dim=-1).float()


def decode(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predicted class, class probabilities, expected level) from ordinal logits."""
    p_gt = torch.sigmoid(logits).cpu().numpy()                     # P(>0), P(>1)
    p_gt[:, 1] = np.minimum(p_gt[:, 1], p_gt[:, 0])                 # keep probabilities consistent
    probs = np.stack([1 - p_gt[:, 0], p_gt[:, 0] - p_gt[:, 1], p_gt[:, 1]], axis=1)
    return probs.argmax(1), probs, p_gt.sum(1)


def batches(rows: list[dict], tokenizer, batch_size: int, max_len: int, shuffle: bool, rng: random.Random):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [rows[j] for j in order[i:i + batch_size]]
        enc = tokenizer([r["text"] for r in chunk], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        labels = torch.tensor([LABEL_ID.get(r.get("label"), -1) for r in chunk])
        yield enc, labels, chunk


@torch.no_grad()
def predict(model, rows, tokenizer, args, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, probs, levels = [], [], []
    for enc, _, _ in batches(rows, tokenizer, args.batch_size * 2, args.max_len, False, random.Random(0)):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float()
        p, pr, lv = decode(logits)
        preds.append(p), probs.append(pr), levels.append(lv)
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(levels)


def metrics(pred: np.ndarray, gold: list[int]) -> dict:
    gold_arr = np.array(gold)
    f1s = []
    for k in range(len(LABELS)):
        tp = int(((pred == k) & (gold_arr == k)).sum())
        fp = int(((pred == k) & (gold_arr != k)).sum())
        fn = int(((pred != k) & (gold_arr == k)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    confusion = [[int(((gold_arr == g) & (pred == p)).sum()) for p in range(len(LABELS))] for g in range(len(LABELS))]
    return {"accuracy": float((pred == gold_arr).mean()), "macro_f1": float(np.mean(f1s)),
            "f1_per_class": dict(zip(LABELS, f1s)), "confusion_gold_x_pred": confusion}


def score_file(model, path: Path, out_path: Path, tokenizer, args, device) -> None:
    rows = load_jsonl(path)
    _, probs, levels = predict(model, rows, tokenizer, args, device)
    with out_path.open("w", encoding="utf-8") as f:
        for row, pr, lv in zip(rows, probs, levels):
            out = {k: v for k, v in row.items() if k != "text"}
            out.update({f"p_{label}": round(float(p), 4) for label, p in zip(LABELS, pr)})
            out["expected_level"] = round(float(lv), 4)
            out["predicted"] = LABELS[int(np.argmax(pr))]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"scored {len(rows)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True, help="has train/val/test.jsonl, llm_windows.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--extra-score", type=Path, nargs="*", default=[], help="more jsonl files to score")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate for the pretrained encoder")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="learning rate for the new score layer + thresholds")
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir / f"{args.encoder}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = (load_jsonl(args.corpus_dir / f"{s}.jsonl") for s in ("train", "val", "test"))
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  device {device}")
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[args.encoder])
    model = OrdinalClassifier(ENCODERS[args.encoder]).to(device)
    optimizer = build_optimizer(model, args.lr, args.head_lr)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = random.Random(args.seed)

    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        for enc, labels, _ in batches(train, tokenizer, args.batch_size, args.max_len, True, rng):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = loss_fn(logits.float(), ordinal_targets(labels.to(device)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(), scheduler.step(), optimizer.zero_grad()
            total += loss.item()
        val_pred, _, _ = predict(model, val, tokenizer, args, device)
        val_m = metrics(val_pred, [LABEL_ID[r["label"]] for r in val])
        history.append({"epoch": epoch, "train_loss": total, **{f"val_{k}": v for k, v in val_m.items()
                                                                 if k in ("accuracy", "macro_f1")}})
        print(f"epoch {epoch}: loss {total:.2f}  val acc {val_m['accuracy']:.3f}  "
              f"val macro-F1 {val_m['macro_f1']:.3f}  ({time.time() - start:.0f}s)", flush=True)
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_pred, _, _ = predict(model, test, tokenizer, args, device)
    result = {"encoder": ENCODERS[args.encoder], "args": {k: str(v) for k, v in vars(args).items()},
              "best_val_macro_f1": best_f1, "history": history,
              "test": metrics(test_pred, [LABEL_ID[r["label"]] for r in test])}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["test"], indent=2))

    score_file(model, args.corpus_dir / "test.jsonl", out_dir / "test_scores.jsonl", tokenizer, args, device)
    score_file(model, args.corpus_dir / "llm_windows.jsonl", out_dir / "llm_window_scores.jsonl", tokenizer, args, device)
    for extra in args.extra_score:
        score_file(model, extra, out_dir / f"{extra.stem}_scores.jsonl", tokenizer, args, device)


if __name__ == "__main__":
    main()
