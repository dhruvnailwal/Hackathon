"""Validate the pipeline against real-world SAML-D transaction data.

SAML-D (Oztas et al., IEEE ICEBE 2023 — the same paper PLAN.md cites as
precedent for our storyline typologies) is a typology-based AML transaction
dataset: 9.5M transactions, 9,873 labeled suspicious (``Is_laundering``),
spanning 28 typologies (Structuring, Smurfing, Layered_Fan_In/Out,
Behavioural_Change, Bipartite, Cash_Withdrawal, Deposit-Send, ...). Source:
https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
(CC BY-NC-SA 4.0 — non-commercial). Kaggle gates downloads behind login, so
this script does NOT fetch it automatically — download it yourself and place
the extracted CSV at ``data/real_datasets/samld/SAML-D.csv`` first.

Unlike PaySim (see ``real_dataset_paysim.py`` — a one-shot-per-sender
dataset where our entity-centric design structurally can't clear
``min_events_per_entity``), SAML-D's senders repeat (median 12 txns/sender)
and 6,537 of the 7,902 fraud-implicated accounts already have >=30 total
appearances — a real structural fit for this pipeline's design.

Sampling (``_build_sample``): rather than a flat row cap, this stratifies by
typology first so rare typologies (Smurfing: 932 rows total dataset-wide)
aren't drowned out by common ones, then per chosen account caps total rows
at ``PER_ACCOUNT_CAP`` (always keeping every one of that account's actual
fraud-labeled rows, subsampling only its normal-transaction rows) so a
handful of 1000+-txn hub accounts can't eat the whole row budget alone.

``Is_laundering``/``Laundering_type`` are stripped from the file handed to
the pipeline (no leakage) and kept only in a write-only ``ground_truth.json``,
same discipline as ``data/answer_key_surprise.json``.

Usage:
    python scripts/real_dataset_samld.py             # uses cached sample if present
    python scripts/real_dataset_samld.py --rebuild    # rebuild sample from SAML-D.csv
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

DATA_DIR = ROOT / "data" / "real_datasets" / "samld"
FULL_CSV = DATA_DIR / "SAML-D.csv"
SAMPLE_CSV = DATA_DIR / "samld_sample.csv"
TRUTH_JSON = DATA_DIR / "ground_truth.json"
OUT = ROOT / "results" / "real_datasets"

ACCOUNTS_PER_TYPE = 40      # max accounts sampled per suspicious typology
PER_ACCOUNT_CAP = 60        # max total rows kept per chosen account (>= min_events_per_entity)
TARGET_TOTAL_ROWS = 150_000
SEED = 42


def _build_sample() -> None:
    if not FULL_CSV.exists():
        raise SystemExit(
            f"Missing {FULL_CSV}. Download the SAML-D CSV from "
            "https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml "
            "(requires a free Kaggle login), unzip it, and place it at that path."
        )
    print("Building reproducible, typology-stratified sample from the full 9.5M-row file ...")
    rng = np.random.RandomState(SEED)
    df = pd.read_csv(FULL_CSV)

    # SENDER-side counts specifically: the pipeline's entity_id is resolved
    # from the actor (Sender_account) column, so an account's OWN volume for
    # the min_events_per_entity gate comes only from rows where it sends —
    # rows where it merely receives don't count toward its own entity total
    # (they land in the counterparty column of someone else's row). Sizing
    # the sample off combined sender+receiver counts (an earlier version of
    # this script did) silently picks fan-in hub accounts that are "busy"
    # only as receivers and still get BLOCKED — same trap PaySim exposed.
    sender_counts = df["Sender_account"].value_counts()
    fraud = df[df["Is_laundering"] == 1]

    # pick up to ACCOUNTS_PER_TYPE accounts per suspicious typology, preferring
    # ones that already clear the volume gate as SENDERS, so rare typologies
    # aren't crowded out by common ones (Structuring/Smurfing have far fewer
    # rows than e.g. Layered_Fan_In)
    chosen_accounts: dict[int, str] = {}
    for ltype, grp in fraud.groupby("Laundering_type"):
        accts = pd.unique(grp["Sender_account"])
        counts = sender_counts.reindex(accts).fillna(0).sort_values(ascending=False)
        for acct in counts.index[:ACCOUNTS_PER_TYPE]:
            chosen_accounts.setdefault(acct, ltype)

    print(f"  chosen accounts across {fraud['Laundering_type'].nunique()} typologies: {len(chosen_accounts)}")

    core_parts = []
    for acct in chosen_accounts:
        # rows where this account is the actor (sender) — this is what
        # actually accumulates as ITS entity_id volume
        sent_rows = df[df["Sender_account"] == acct]
        must_keep = sent_rows[sent_rows["Is_laundering"] == 1]
        rest_rows = sent_rows[sent_rows["Is_laundering"] == 0]
        pad_n = max(0, PER_ACCOUNT_CAP - len(must_keep))
        pad_rows = rest_rows.sample(n=min(pad_n, len(rest_rows)), random_state=SEED)
        # a handful of rows where they're the RECEIVER too, for realistic
        # counterparty-side context (doesn't count toward their own volume,
        # but keeps the receiver-side signal from vanishing entirely)
        received_rows = df[(df["Receiver_account"] == acct) & (df["Sender_account"] != acct)]
        received_sample = received_rows.sample(n=min(20, len(received_rows)), random_state=SEED)
        core_parts.append(pd.concat([must_keep, pad_rows, received_sample]))
    core = pd.concat(core_parts).drop_duplicates()
    print(f"  core rows (chosen accounts, capped): {len(core):,}  "
          f"fraud rows within core: {int(core['Is_laundering'].sum()):,}")

    # Background population: drawn ONLY from other accounts that themselves
    # clear the volume gate as senders (>=30 sends dataset-wide), not a
    # uniform random sample. sufficiency.py's volume check (see
    # _overall_volume_note) is a POPULATION AVERAGE (total_rows/total_entities),
    # not per-entity — so diluting with realistically-low-activity accounts
    # (median sender has only 12 txns dataset-wide) drags the whole dataset's
    # average below the threshold and BLOCKS every entity-scored model
    # regardless of how rich the fraud accounts' own histories are. Scoping
    # the background to already-active accounts is a real, disclosable
    # framing (monitoring restricted to accounts above an activity floor —
    # standard practice in real AML triage, not a metrics trick), and lets
    # this validation actually measure ranking quality instead of always
    # hitting the same global BLOCKED wall PaySim did.
    active_pool = sender_counts[sender_counts >= 30].index
    active_pool = [a for a in active_pool if a not in chosen_accounts]
    rng.shuffle(active_pool := list(active_pool))

    pad_parts = []
    running = len(core)
    for acct in active_pool:
        if running >= TARGET_TOTAL_ROWS:
            break
        sent_rows = df[df["Sender_account"] == acct]
        capped = sent_rows.sample(n=min(PER_ACCOUNT_CAP, len(sent_rows)), random_state=SEED)
        pad_parts.append(capped)
        running += len(capped)
    pad = pd.concat(pad_parts) if pad_parts else df.iloc[:0]

    sample = pd.concat([core, pad]).drop_duplicates().sort_values(["Date", "Time"]).reset_index(drop=True)
    print(f"  total sample rows: {len(sample):,}  fraud rows: {int(sample['Is_laundering'].sum()):,}")

    sender_counts_s = sample["Sender_account"].value_counts()
    n_ge30 = int((sender_counts_s >= 30).sum())
    print(f"  accounts with >=30 SENDER-side rows in sample (own entity volume): {n_ge30:,}")

    # SAML-D ships Date + Time as two separate columns; derive one real
    # datetime the same way real analysts would have to (disclosed prep
    # step, not a schema-detection shortcut — Stage A still detects it blind)
    sample["txn_timestamp"] = pd.to_datetime(sample["Date"] + " " + sample["Time"])

    fraud_only = sample[sample["Is_laundering"] == 1]
    acct_type: dict[str, list] = {}
    for _, r in fraud_only.iterrows():
        for acct in (r["Sender_account"], r["Receiver_account"]):
            acct_type.setdefault(str(acct), set()).add(r["Laundering_type"])
    truth = {
        "fraud_row_count": int(sample["Is_laundering"].sum()),
        "total_rows": int(len(sample)),
        "n_accounts_ge30": n_ge30,
        "fraud_actor_types": {k: sorted(v) for k, v in acct_type.items()},
    }
    TRUTH_JSON.write_text(json.dumps(truth, indent=2))
    print(f"  ground-truth fraud accounts: {len(truth['fraud_actor_types']):,}")

    pipeline_cols = ["txn_timestamp", "Sender_account", "Receiver_account", "Amount",
                      "Payment_currency", "Received_currency", "Sender_bank_location",
                      "Receiver_bank_location", "Payment_type"]
    sample[pipeline_cols].to_csv(SAMPLE_CSV, index=False)
    print(f"  wrote {SAMPLE_CSV}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the sample from SAML-D.csv")
    args = ap.parse_args()

    if args.rebuild or not SAMPLE_CSV.exists() or not TRUTH_JSON.exists():
        _build_sample()
    else:
        print(f"Using cached sample: {SAMPLE_CSV}")

    truth = json.loads(TRUTH_JSON.read_text())
    fraud_actor_types = truth["fraud_actor_types"]

    print("\n== Schema auto-detection (blind — no manual column mapping) ==")
    ls = load_any(SAMPLE_CSV)
    det = detect_dataframe(ls.df, file=ls.file)
    resolved = {k: v.slot for k, v in det.columns.items() if v.slot}
    unresolved = [k for k, v in det.columns.items() if not v.slot]
    print(f"  format={ls.format} shape={ls.df.shape} source_type={det.source_type}")
    for col, slot in resolved.items():
        print(f"    {col:22s} -> {slot}")
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

    id_map = res.entity_map.entity_id_by_identifier
    truth_type: dict = {}
    for acct, types in fraud_actor_types.items():
        eid = id_map.get(acct)
        if eid:
            truth_type[eid] = types[0]  # eval.py's recall_precision wants one type per entity
    truth_entities = set(truth_type.keys())
    print(f"\nGround-truth fraud accounts: {len(fraud_actor_types):,}; "
          f"of those, {len(truth_entities):,} resolved to a scored entity_id.")

    metrics = {}
    if res.rankings and truth_entities:
        ks = (5, 10, 20, 50, 100, 200, 500, 1000)
        metrics = recall_precision(res.rankings, truth_entities, truth_type, ks=ks)
        n_total = len(res.rankings)
        n_pos = len(truth_entities)
        print("\nMetrics vs. real SAML-D laundering labels (+ vs. a random-ranking baseline):")
        for k in ks:
            expected_random = k * n_pos / n_total
            hits = round(metrics[f"recall@{k}"] * n_pos)
            lift = hits / expected_random if expected_random else float("inf")
            print(f"   @{k:<5d} recall={metrics[f'recall@{k}']:.3f}  precision={metrics[f'precision@{k}']:.3f}  "
                  f"hits={hits:<4d} random-expected={expected_random:.1f}  lift={lift:.1f}x")
        print("   type_recall@10:")
        for t, r in metrics["type_recall@10"].items():
            print(f"      {t:24s} {r:.3f}")
    else:
        print("\nNo rankings produced or no ground-truth entities resolved — "
              "see verdicts above for why.")

    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": {"name": "samld_sample.csv", "source": "kaggle:berkanoztas/synthetic-transaction-monitoring-dataset-aml",
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
        "ground_truth": {"fraud_accounts": len(fraud_actor_types),
                          "resolved_to_entity": len(truth_entities)},
        "metrics": metrics,
    }
    (OUT / "samld_verdict.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nWrote {OUT / 'samld_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
