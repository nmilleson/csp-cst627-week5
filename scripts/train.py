"""Train one configuration and save the best-on-validation checkpoint.

Configs with a "teacher" key are trained by knowledge distillation: the loss mixes the
teacher's temperature-softened predictions (KL divergence) with the hard labels (cross-entropy).

Also records the numbers the scaling-law estimate needs: parameter count N and the
number of real (non-padding) training tokens D actually processed.

Usage:
    python scripts/train.py --config baseline
    python scripts/train.py --config linear_attention
    python scripts/train.py --config distilled_student     # needs checkpoints/baseline.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT_DIR, PAD_ID, RESULTS_DIR, SEED, TextDataset, collate, load_labels, set_seed  # noqa: E402
from configs import get_config  # noqa: E402
from model import build_model, count_params, load_for_inference  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, help="override the config's epoch count (e.g. quick smoke test)")
    args = ap.parse_args()

    cfg = get_config(args.config)
    if "source" in cfg:
        print(f"[{cfg['name']}] post-training transform of '{cfg['source']}' - nothing to train; "
              f"run benchmark.py --config {cfg['name']}")
        return
    if args.epochs:
        cfg["epochs"] = args.epochs
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels()

    train_ds, val_ds = TextDataset("train"), TextDataset("val")
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    val_dl = DataLoader(val_ds, batch_size=64, collate_fn=collate)

    model = build_model(cfg, len(labels)).to(device)
    params = count_params(model)
    print(f"[{cfg['name']}] {params['params_total']:,} params ({params['params_non_embedding']:,} non-embedding) on {device}")

    teacher = None
    if "teacher" in cfg:
        teacher, _, _, _ = load_for_inference(cfg["teacher"], device)
        alpha, temp = cfg["kd_alpha"], cfg["kd_temperature"]
        print(f"  distilling from '{cfg['teacher']}' (alpha={alpha}, T={temp})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(train_dl)
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"{cfg['name']}.pt"
    best_acc, tokens_seen, history = -1.0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            tokens_seen += (x != PAD_ID).sum().item()
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    kd = F.kl_div(F.log_softmax(logits.float() / temp, -1), F.softmax(t_logits.float() / temp, -1),
                                  reduction="batchmean") * temp**2
                    loss = alpha * kd + (1 - alpha) * loss
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss ({loss.item()}) at epoch {epoch} - training diverged")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        val_acc = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), "val_acc": val_acc})
        print(f"  epoch {epoch:2d}  loss {running / len(train_dl):.4f}  val_acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"config": cfg, "labels": labels, "state_dict": model.state_dict()}, ckpt_path)
    train_time = time.perf_counter() - t0

    meta = {
        "config": cfg["name"],
        "seed": args.seed,
        **params,
        "train_tokens": tokens_seen,
        "train_examples_per_epoch": len(train_ds),
        "epochs": cfg["epochs"],
        "teacher": cfg.get("teacher", ""),
        "best_val_acc": best_acc,
        "train_time_s": round(train_time, 1),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "history": history,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"train_{cfg['name']}.json").write_text(json.dumps(meta, indent=2))
    print(f"best val_acc {best_acc:.4f}  |  {tokens_seen:,} training tokens  |  {train_time:.0f}s  ->  {ckpt_path}")


if __name__ == "__main__":
    main()
