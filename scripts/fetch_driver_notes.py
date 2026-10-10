"""Download the public SLURP spoken-language dataset and keep in-cab driver-note intents.

SLURP (Bastianelli et al., EMNLP 2020) is ~72k audio recordings of people speaking short
commands to a voice assistant, each with a transcript and an intent label. We only use the
text annotations (~18 MB from GitHub, not the ~6 GB of audio): the tablet already runs
on-device speech-to-text, so the model we're choosing sees the transcript.

We keep the 8 intents a driver would plausibly speak in the cab (examples are illustrative):

    calendar_set      "remind me about the 3pm delivery window"
    calendar_query    "what's my next appointment"
    email_sendemail   "send an email to dispatch that i'm running late"
    email_query       "do i have any new emails"
    weather_query     "is it going to snow tonight"
    transport_query   "how do i get to the interstate"
    datetime_query    "what time is it in denver"
    alarm_set         "wake me up in thirty minutes"

SLURP's own train / devel / test splits become our train / val / test.

Source: https://github.com/pswietojanski/slurp (CC BY 4.0)

Usage:
    python scripts/fetch_driver_notes.py
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR, write_jsonl  # noqa: E402

BASE_URL = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/{split}.jsonl"
SPLITS = {"train": "train", "val": "devel", "test": "test"}
INTENTS = ["calendar_set", "calendar_query", "email_sendemail", "email_query",
           "weather_query", "transport_query", "datetime_query", "alarm_set"]


def download(split: str, cache: Path) -> Path:
    path = cache / f"{split}.jsonl"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = BASE_URL.format(split=split)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    cache = RAW_DIR / "slurp_source"
    out = RAW_DIR / "driver_notes"
    for ours, theirs in SPLITS.items():
        rows, seen = [], set()
        with open(download(theirs, cache), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec["sentence"].strip()
                if rec["intent"] not in INTENTS or text in seen:
                    continue
                seen.add(text)
                rows.append({
                    "text": text,
                    "label": f"note:{rec['intent']}",
                    "source": "slurp_notes",
                    "meta": {"slurp_id": rec["slurp_id"], "n_recordings": len(rec.get("recordings", []))},
                })
        write_jsonl(out / f"{ours}.jsonl", rows)
        counts = {i: sum(r["label"] == f"note:{i}" for r in rows) for i in INTENTS}
        print(f"{ours:5s}: {len(rows):5d} notes  {counts}")


if __name__ == "__main__":
    main()
