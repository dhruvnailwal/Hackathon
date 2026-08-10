"""Generate a dataset suite: multiple datasets per cross-source correlation mode.

Modes (see aml.data_generator.emit.CORRELATION_MODES):
    full             every person appears in all three sources (original scenario)
    none             three disjoint populations — no correlation between sources
    pair_bank_cdr    bank+cdr share a population; social is independent
    pair_bank_social bank+social share a population; cdr is independent
    pair_cdr_social  cdr+social share a population; bank is independent

Each dataset is a self-contained folder (data/datasets/<mode>/set_<k>/) with
sources/*.csv|json|jsonl plus answer_key.json / answer_key_surprise.json, in
the exact layout the pipeline/eval harness expect. A manifest.json records
every dataset with per-source row counts and cross-source entity overlap so
the correlation structure can be verified mechanically.

Usage:
    python scripts/generate_dataset_suite.py [--out data/datasets] [--n-seeds 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml.data_generator.emit import CORRELATION_MODES, name_column  # noqa: E402
from aml.data_generator.emit_multi import generate_all_mixed  # noqa: E402


def source_entity_sets(dataset: Path) -> dict:
    """Distinct names per source file (the cross-source correlation proxy)."""
    out = {}
    for src in ("bank", "cdr", "social"):
        path = dataset / "sources" / f"{src}_export.csv"
        if not path.exists():
            out[src] = set()
            continue
        df = pd.read_csv(path, low_memory=False)
        col = name_column(df, src)
        out[src] = set(df[col].astype(str).dropna().unique())
    return out


def overlap(a: set, b: set) -> float:
    denom = len(a | b)
    return len(a & b) / denom if denom else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "datasets"))
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="datasets per correlation mode")
    ap.add_argument("--n-background", type=int, default=90)
    ap.add_argument("--n-per-storyline", type=int, default=3)
    ap.add_argument("--n-surprise", type=int, default=12)
    args = ap.parse_args()

    out_root = Path(args.out)
    manifest = {"modes": list(CORRELATION_MODES), "datasets": []}

    for mode in CORRELATION_MODES:
        for k in range(args.n_seeds):
            seed = 1000 + k * 100 + CORRELATION_MODES.index(mode)
            dataset = out_root / mode / f"set_{k + 1}"
            generate_all_mixed(
                dataset, seed=seed, surprise_seed=seed + 55,
                n_background=args.n_background,
                n_per_storyline=args.n_per_storyline,
                n_surprise=args.n_surprise, correlation=mode)
            sets = source_entity_sets(dataset)
            manifest["datasets"].append({
                "mode": mode,
                "name": dataset.relative_to(out_root).as_posix(),
                "seed": seed,
                "surprise_seed": seed + 55,
                "n_entities": sum(len(s) for s in sets.values()),
                "rows": {
                    src: int(pd.read_csv(dataset / "sources" / f"{src}_export.csv",
                                         low_memory=False).shape[0])
                    for src in ("bank", "cdr", "social")
                    if (dataset / "sources" / f"{src}_export.csv").exists()
                },
                "overlap": {
                    "bank_cdr": round(overlap(sets["bank"], sets["cdr"]), 3),
                    "bank_social": round(overlap(sets["bank"], sets["social"]), 3),
                    "cdr_social": round(overlap(sets["cdr"], sets["social"]), 3),
                },
            })
            print(f"{mode:16s} set_{k + 1}  rows="
                  f"{manifest['datasets'][-1]['rows']}  overlap="
                  f"{manifest['datasets'][-1]['overlap']}")

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# Dataset suite (cross-source correlation variants)",
          "",
          f"- modes: {', '.join(CORRELATION_MODES)}",
          f"- datasets: {len(manifest['datasets'])}",
          "",
          "| mode | set | rows (b/c/s) | overlap b-c / b-s / c-s |",
          "|---|---|---|---|"]
    for d in manifest["datasets"]:
        rows = d["rows"]
        ov = d["overlap"]
        md.append(f"| {d['mode']} | {d['name']} | "
                  f"{rows.get('bank', 0)}/{rows.get('cdr', 0)}/{rows.get('social', 0)} | "
                  f"{ov['bank_cdr']} / {ov['bank_social']} / {ov['cdr_social']} |")
    (out_root / "suite.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"manifest -> {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
