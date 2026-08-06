"""Generate mixed-format sample volumes so the pipeline is exercised across
CSV, JSON and plain-text inputs ("one dataset, every format" testing §3.4).

Builds on ``generate_all`` (same people + storylines + answer keys) but also
re-emits the loaded sources as:
    bank_export.json    -> JSON array of records     (load_json)
    cdr_export.jsonl     -> NDJSON stream            (load_ndjson)
    social_export.txt    -> tab-delimited table       (load_text)
leaving the original CSVs in place, so a single folder feeds every loader.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .emit import generate_all

ROOT = Path(__file__).resolve().parents[3]


def _to_json(df: pd.DataFrame, path: Path) -> None:
    df.to_json(path, orient="records", lines=False)


def _to_jsonl(df: pd.DataFrame, path: Path) -> None:
    df.to_json(path, orient="records", lines=True)


def generate_all_mixed(out_dir, formats=("csv", "json", "jsonl", "text"), **kw) -> Dict[str, Path]:
    files = generate_all(out_dir, **kw)

    for name in ("bank", "cdr"):
        csv_path = files.get(name)
        if not (csv_path and csv_path.exists()):
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        if "json" in formats and name == "bank":
            p = out_dir / "sources" / "bank_export.json"
            _to_json(df, p)
            files["bank_json"] = p
        if "jsonl" in formats and name == "cdr":
            p = out_dir / "sources" / "cdr_export.jsonl"
            _to_jsonl(df, p)
            files["cdr_jsonl"] = p
    return files


def make_social_text(out_dir: Path, n: int = 320, seed: int = 7) -> Path:
    """A representative plain-text social file (independent of scenario ids)."""
    import random
    from datetime import datetime, timedelta, timezone
    rng = random.Random(seed)
    ts_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    path = out_dir / "sources" / "social_export.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("timestamp\thandle\tmentions\tgeo\n")
        for _ in range(n):
            ts_start += timedelta(seconds=rng.randint(30, 3600 * 30))
            handle = f"u{rng.randint(1000, 9999)}"
            mentions = f"u{rng.randint(1000, 9999)}"
            geo = f"loc{rng.randint(1, 40)}"
            fh.write(f"{ts_start.isoformat()}\t{handle}\t{mentions}\t{geo}\n")
    return path


if __name__ == "__main__":
    out = ROOT / "data"
    f = generate_all_mixed(out, seed=42, surprise_seed=99)
    social = make_social_text(out)
    print("wrote:", sorted(set(f.values())))
    print("text:", social)