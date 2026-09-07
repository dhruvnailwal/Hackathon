"""Validate the pipeline against real-world PaySim transaction data.

PaySim (Lopez-Rojas et al. 2016) is a mobile-money transaction simulator
calibrated against a real African mobile-money service's private logs; it
ships a real ``isFraud`` label per transaction. Source:
https://www.kaggle.com/datasets/ealaxi/paysim1 (CC BY-SA 4.0). Kaggle gates
downloads behind login, so this script does NOT fetch it automatically —
download it yourself and place the extracted CSV at
``data/real_datasets/paysim/paysim_full.csv`` first.

This complements ``real_dataset_validate.py`` (72.8k-row HF AML sample) and
``fetch_samples.py`` (input-robustness smoke test): this one is the first
run against a widely-cited, real (not internet-toy) labeled fraud dataset,
at a scale (6.36M rows) too large to run whole, so a bounded, reproducible
sample is built first (see ``_build_sample``):

  - every one of the 8,213 real fraud-labeled rows (never subsampled)
  - full transaction history for the ~1-2k most active accounts, so some
    entities actually clear ``min_events_per_entity`` the way a real
    high-volume account would (PaySim's per-account volume is otherwise
    ~1 txn/account — a single-shot fraud dataset, not a behavioral one;
    that mismatch with our entity-centric design is itself the finding)
  - a random background pad up to the row budget

The ``isFraud``/``isFlaggedFraud`` label columns are stripped from the file
handed to the pipeline (no leakage) and kept only in a separate, write-only
``ground_truth.json`` used solely by this script's own evaluation step —
the same discipline as ``data/answer_key_surprise.json``.

Usage:
    python scripts/real_dataset_paysim.py            # uses cached sample if present
    python scripts/real_dataset_paysim.py --rebuild   # rebuild the sample from paysim_full.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from aml.config import PipelineConfig  # noqa: E402
from aml.loaders import load_any  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402
from aml.schema_detect import detect_dataframe  # noqa: E402
from eval import recall_precision  # noqa: E402

DATA_DIR = ROOT / "data" / "real_datasets" / "paysim"
FULL_CSV = DATA_DIR / "paysim_full.csv"
SAMPLE_CSV = DATA_DIR / "paysim_sample.csv"
TRUTH_JSON = DATA_DIR / "ground_truth.json"
OUT = ROOT / "results" / "real_datasets"

TARGET_TOTAL_ROWS = 120_000
ACTIVE_ROW_BUDGET = 70_000
SEED = 42


def _build_sample() -> None:
    if not FULL_CSV.exists():
        raise SystemExit(
            f"Missing {FULL_CSV}. Download the PaySim CSV from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 (requires a "
            "free Kaggle login), unzip it, and place it at that path."
        )
    print("Building reproducible sample from the full 6.36M-row file ...")
    df = pd.read_csv(FULL_CSV)

    both = pd.concat([df["nameOrig"], df["nameDest"]]).value_counts()
    active_sorted = both[both >= 30].sort_values(ascending=False)

    is_fraud = df["isFraud"] == 1
    fraud_rows = df[is_fraud]

    chosen, running = [], 0
    for acct, cnt in active_sorted.items():
        if running + cnt > ACTIVE_ROW_BUDGET:
            continue
        chosen.append(acct)
        running += cnt
    chosen = set(chosen)

    touches_chosen = (df["nameOrig"].isin(chosen) | df["nameDest"].isin(chosen)) & ~is_fraud
    active_rows = df[touches_chosen]

    rest = df[~is_fraud & ~touches_chosen]
    pad_budget = TARGET_TOTAL_ROWS - len(fraud_rows) - len(active_rows)
    pad = rest.sample(n=min(max(pad_budget, 0), len(rest)), random_state=SEED)

    sample = pd.concat([fraud_rows, active_rows, pad]).sort_values("step").reset_index(drop=True)

    # PaySim ships a relative hour-tick ('step'), not calendar time — derive
    # a real timestamp so Stage A's own date-parse probe has to do its job,
    # rather than handing it a pre-solved column.
    start = pd.Timestamp("2025-01-01")
    sample["txn_datetime"] = start + pd.to_timedelta(sample["step"], unit="h")

    fraud_only = sample[sample["isFraud"] == 1]
    truth = {
        "fraud_row_count": int(sample["isFraud"].sum()),
        "total_rows": int(len(sample)),
        "n_active_accounts_ge30": int(len(chosen)),
        "fraud_actor_ids": sorted(set(fraud_only["nameOrig"]) | set(fraud_only["nameDest"])),
    }
    TRUTH_JSON.write_text(json.dumps(truth, indent=2))

    pipeline_cols = ["txn_datetime", "type", "amount", "nameOrig", "oldbalanceOrg",
                      "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]
    sample[pipeline_cols].to_csv(SAMPLE_CSV, index=False)
    print(f"  sample rows={len(sample):,}  fraud rows={truth['fraud_row_count']:,}  "
          f"ground-truth accounts={len(truth['fraud_actor_ids']):,}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the sample from paysim_full.csv")
    args = ap.parse_args()

    if args.rebuild or not SAMPLE_CSV.exists() or not TRUTH_JSON.exists():
        _build_sample()
    else:
        print(f"Using cached sample: {SAMPLE_CSV}")

    truth = json.loads(TRUTH_JSON.read_text())
    fraud_actor_ids = set(truth["fraud_actor_ids"])

    print("\n== Schema auto-detection (blind — no manual column mapping) ==")
    ls = load_any(SAMPLE_CSV)
    det = detect_dataframe(ls.df, file=ls.file)
    resolved = {k: v.slot for k, v in det.columns.items() if v.slot}
    unresolved = [k for k, v in det.columns.items() if not v.slot]
    print(f"  format={ls.format} shape={ls.df.shape} source_type={det.source_type}")
    for col, slot in resolved.items():
        print(f"    {col:16s} -> {slot}")
    if unresolved:
        print(f"  unresolved columns (expected to be ignored): {unresolved}")

    print("\n== Full pipeline run ==")
    cfg = PipelineConfig.from_yaml(str(ROOT / "config" / "config.yaml"))
    pipe = Pipeline(min_events_per_entity=cfg.min_events_per_entity,
                    tol_minutes=cfg.tol_minutes,
                    structuring_threshold=cfg.structuring_threshold)
    res = pipe.run([str(SAMPLE_CSV)], with_models=cfg.with_models)

    print("Model verdicts:")
    for v in res.sufficiency.values():
        if v.model == "schema_detection":
            continue
        print(f"   {v.model:20s} {v.status:<9s} {v.reason[:90]}")

    # map ground-truth raw account ids -> resolved entity_ids (same mapping
    # the pipeline itself produced; not a separate hand-rolled linkage)
    id_map = res.entity_map.entity_id_by_identifier
    truth_entities = {id_map[a] for a in fraud_actor_ids if a in id_map}
    print(f"\nGround-truth fraud accounts: {len(fraud_actor_ids):,}; "
          f"of those, {len(truth_entities):,} resolved to a scored entity_id.")

    metrics = {}
    if res.rankings and truth_entities:
        metrics = recall_precision(res.rankings, truth_entities, {e: "fraud" for e in truth_entities})
        print("\nMetrics vs. real PaySim fraud labels:")
        for k, v in metrics.items():
            if k != "type_recall@10":
                print(f"   {k:14s} {v:.3f}")
    else:
        print("\nNo rankings produced or no ground-truth entities resolved — "
              "see verdicts above for why (likely BLOCKED on volume).")

    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": {"name": "paysim_sample.csv", "source": "kaggle:ealaxi/paysim1",
                    "rows": int(len(ls.df)), "fraud_rows": truth["fraud_row_count"]},
        "detection": {"source_type": det.source_type, "resolved_slots": resolved,
                       "unresolved_columns": unresolved, "warnings": det.warnings},
        "pipeline": {
            "entities": int(res.unified["entity_id"].nunique()) if not res.unified.empty else 0,
            "n_rankings": len(res.rankings),
            "warnings": res.warnings,
            "verdicts": {k: {"status": v.status, "reason": v.reason}
                         for k, v in res.sufficiency.items() if v.model != "schema_detection"},
        },
        "ground_truth": {"fraud_accounts": len(fraud_actor_ids),
                          "resolved_to_entity": len(truth_entities)},
        "metrics": metrics,
    }
    (OUT / "paysim_verdict.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nWrote {OUT / 'paysim_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
