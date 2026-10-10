"""Merge synthetic documents + SLURP driver notes into one 16-class dataset.

Writes:
    data/train.jsonl, data/val.jsonl, data/test.jsonl   (text, label, source, meta)
    data/labels.json                                     (ordered label list)
    data/dataset_card.json                               (counts, lengths, provenance)

The tablet routes both input kinds through one model, so labels share one space:
"doc:*" for scanned documents and "note:*" for spoken notes. The benchmark reports
accuracy overall and per source.

Usage:
    python scripts/build_eval_set.py
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, MAX_LEN, RAW_DIR, SEED, read_jsonl, write_jsonl  # noqa: E402

SOURCES = ["synthetic_docs", "driver_notes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    splits = {}
    for split in ["train", "val", "test"]:
        rows = []
        for src in SOURCES:
            path = RAW_DIR / src / f"{split}.jsonl"
            if not path.exists():
                sys.exit(f"missing {path} - run generate_documents.py and fetch_driver_notes.py first")
            rows.extend(read_jsonl(path))
        rng.shuffle(rows)
        splits[split] = rows

    labels = sorted({r["label"] for rows in splits.values() for r in rows})
    missing = {s: sorted(set(labels) - {r["label"] for r in rows}) for s, rows in splits.items()}
    assert not any(missing.values()), f"labels missing from a split: {missing}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    card = {"labels": labels, "max_len_bytes": MAX_LEN, "splits": {}}
    for split, rows in splits.items():
        write_jsonl(DATA_DIR / f"{split}.jsonl", rows)
        by_src = {}
        for src in sorted({r["source"] for r in rows}):
            lens = [len(r["text"].encode("utf-8")) for r in rows if r["source"] == src]
            by_src[src] = {
                "n": len(lens),
                "bytes_mean": round(statistics.mean(lens), 1),
                "bytes_p95": sorted(lens)[int(0.95 * (len(lens) - 1))],
                "bytes_max": max(lens),
                "truncated": sum(n + 1 > MAX_LEN for n in lens),
            }
        card["splits"][split] = {
            "n": len(rows),
            "by_source": by_src,
            "by_label": dict(sorted(Counter(r["label"] for r in rows).items())),
        }
    card["provenance"] = {
        "synthetic_docs": "scripts/generate_documents.py (procedural templates + OCR noise; "
                          "test split uses unseen company/city names)",
        "slurp_notes": "SLURP text annotations, github.com/pswietojanski/slurp, CC BY 4.0",
    }
    (DATA_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2))

    print(f"{len(labels)} labels")
    for split, info in card["splits"].items():
        srcs = ", ".join(f"{k}={v['n']} (mean {v['bytes_mean']}B, p95 {v['bytes_p95']}B, "
                         f"{v['truncated']} truncated)" for k, v in info["by_source"].items())
        print(f"{split:5s}: {info['n']:5d}  {srcs}")


if __name__ == "__main__":
    main()
