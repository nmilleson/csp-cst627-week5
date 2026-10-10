"""Benchmark one trained configuration on the held-out test set.

Protocol (identical for every configuration):
  * fixed seeds; request order shuffled per trial with seed + trial
  * batch size 1: one scanned document or one spoken note per request, as on the tablet
  * per-request latency = host tensor -> device -> forward -> argmax back on host,
    bracketed by torch.cuda.synchronize() so asynchronous kernels are fully counted
  * WARMUP untimed requests before every trial (CUDA context, kernel selection, caches)
  * TRIALS repeated passes over the whole test set; accuracy comes from the same pass
  * peak memory = torch.cuda.max_memory_allocated() during the trial (weights + activations)
  * checkpoint_mb = serialized size of the weights as loaded (4-bit configs ship packed weights)

Every trial is appended as one row to results/runs.csv (the workbook's raw-data sheet),
and a per-config summary is written to results/summary_<config>.json.

Usage:
    python scripts/benchmark.py --config baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, RUN_FIELDS, RUNS_CSV, SEED, TextDataset, set_seed  # noqa: E402
from model import load_for_inference, state_dict_mb  # noqa: E402



def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_trial(model, items, order, device, warmup: int):
    # Warm-up on a fixed slice of real requests (mix of long docs and short notes).
    for i in order[:warmup]:
        model(torch.tensor([items[i][0]]).to(device)).argmax(-1).cpu()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    lat_ms = np.empty(len(order))
    correct = np.empty(len(order), dtype=bool)
    sources = []
    for j, i in enumerate(order):
        ids, label, src = items[i]
        x = torch.tensor([ids])
        sync(device)
        t0 = time.perf_counter()
        pred = model(x.to(device)).argmax(-1).cpu()
        sync(device)
        lat_ms[j] = (time.perf_counter() - t0) * 1000
        correct[j] = pred.item() == label
        sources.append(src)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    return lat_ms, correct, np.array(sources), peak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-requests", type=int, default=0, help="cap requests per trial (0 = full test set)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() if device.type == "cuda" else 0
    model, cfg, labels, params = load_for_inference(args.config, device)
    weights_mb = (torch.cuda.memory_allocated() - mem_before) / 2**20 if device.type == "cuda" else float("nan")
    ckpt_mb = state_dict_mb(model)
    precision = cfg.get("precision", "fp32")

    items = TextDataset("test").items
    n = min(args.max_requests or len(items), len(items))
    run_id = uuid.uuid4().hex[:8]
    env = {
        "device": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor() or "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "python_version": platform.python_version(),
    }
    print(f"[{args.config}] run {run_id} on {env['device']} | {n} requests x {args.trials} trials, "
          f"{args.warmup} warm-up each")

    RESULTS_DIR.mkdir(exist_ok=True)
    new_file = not RUNS_CSV.exists()
    rows = []
    for trial in range(args.trials):
        order = np.random.default_rng(args.seed + trial).permutation(len(items))[:n]
        lat, correct, src, peak = run_trial(model, items, order, device, args.warmup)
        docs, notes = src == "synthetic_docs", src == "slurp_notes"
        row = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": args.config,
            "variant_type": cfg.get("variant_type", ""),
            "description": cfg.get("description", ""),
            "trial": trial,
            "seed": args.seed,
            "n_requests": n,
            "accuracy": round(correct.mean(), 5),
            "acc_synthetic_docs": round(correct[docs].mean(), 5) if docs.any() else "",
            "acc_slurp_notes": round(correct[notes].mean(), 5) if notes.any() else "",
            "p50_ms": round(float(np.percentile(lat, 50)), 4),
            "p95_ms": round(float(np.percentile(lat, 95)), 4),
            "mean_ms": round(float(lat.mean()), 4),
            "p50_docs_ms": round(float(np.percentile(lat[docs], 50)), 4) if docs.any() else "",
            "p50_notes_ms": round(float(np.percentile(lat[notes], 50)), 4) if notes.any() else "",
            "peak_mem_mb": round(peak, 2),
            "weights_mem_mb": round(weights_mb, 2),
            **params,
            "checkpoint_mb": round(ckpt_mb, 2),
            "precision": precision,
            **env,
        }
        rows.append(row)
        print(f"  trial {trial}: acc {row['accuracy']:.4f} (docs {row['acc_synthetic_docs']}, "
              f"notes {row['acc_slurp_notes']})  p50 {row['p50_ms']:.2f} ms  p95 {row['p95_ms']:.2f} ms  "
              f"peak {row['peak_mem_mb']:.1f} MB")

    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)

    def stat(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return {"median": round(float(np.median(vals)), 4), "std": round(float(vals.std(ddof=1)), 4)
                if len(vals) > 1 else 0.0}

    summary = {
        "run_id": run_id,
        "config": args.config,
        "variant_type": cfg.get("variant_type", ""),
        "description": cfg.get("description", ""),
        "trials": args.trials,
        "warmup": args.warmup,
        "n_requests": n,
        **{k: stat(k) for k in ["accuracy", "acc_synthetic_docs", "acc_slurp_notes", "p50_ms", "p95_ms",
                                "mean_ms", "peak_mem_mb"]},
        "weights_mem_mb": round(weights_mb, 2),
        "checkpoint_mb": round(ckpt_mb, 2),
        "precision": precision,
        **params,
        **env,
    }
    (RESULTS_DIR / f"summary_{args.config}.json").write_text(json.dumps(summary, indent=2))
    print(f"median over {args.trials} trials: acc {summary['accuracy']['median']:.4f}  "
          f"p50 {summary['p50_ms']['median']:.2f} ms  p95 {summary['p95_ms']['median']:.2f} ms  "
          f"peak {summary['peak_mem_mb']['median']:.1f} MB  ->  {RUNS_CSV}")


if __name__ == "__main__":
    main()
