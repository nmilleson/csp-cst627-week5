"""Shared helpers: paths, seeding, byte-level tokenizer, dataset loading."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"
WORKBOOK = RESULTS_DIR / "brightline_benchmark.xlsx"

SEED = 627

# Columns of results/runs.csv: one row per benchmark trial (the workbook's Runs sheet).
RUN_FIELDS = [
    "run_id", "timestamp_utc", "config", "variant_type", "description", "trial", "seed", "n_requests",
    "accuracy", "acc_synthetic_docs", "acc_slurp_notes", "p50_ms", "p95_ms", "mean_ms", "p50_docs_ms",
    "p50_notes_ms", "peak_mem_mb", "weights_mem_mb", "params_total", "params_non_embedding",
    "checkpoint_mb", "precision", "device", "torch_version", "cuda_version", "python_version",
]

# Byte-level tokenizer: ids 0..255 are raw bytes, then special tokens.
# Bytes keep the model robust to OCR noise (no out-of-vocabulary words) and
# give the long sequences where attention cost actually matters.
PAD_ID = 256
CLS_ID = 257
VOCAB_SIZE = 258
MAX_LEN = 1024


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def encode(text: str, max_len: int = MAX_LEN) -> list[int]:
    """Text -> [CLS] + utf-8 bytes, truncated to max_len."""
    return [CLS_ID] + list(text.encode("utf-8"))[: max_len - 1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels() -> list[str]:
    return json.loads((DATA_DIR / "labels.json").read_text())


def load_split(split: str) -> list[dict]:
    return read_jsonl(DATA_DIR / f"{split}.jsonl")


class TextDataset:
    """Encodes a split once; items are (token_ids, label_id, source)."""

    def __init__(self, split: str, max_len: int = MAX_LEN):
        labels = load_labels()
        idx = {name: i for i, name in enumerate(labels)}
        rows = load_split(split)
        self.items = [(encode(r["text"], max_len), idx[r["label"]], r["source"]) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    """Pad to the longest sequence in the batch (dynamic padding)."""
    import torch

    ids, labels, sources = zip(*batch)
    width = max(len(x) for x in ids)
    x = torch.full((len(ids), width), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(ids):
        x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return x, torch.tensor(labels, dtype=torch.long), list(sources)
