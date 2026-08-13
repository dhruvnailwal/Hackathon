"""Dirty-data audit harness: classify every §3 scenario as (a)/(b)/(c).

  (a) runs correctly / degraded-but-honest
  (b) crashes
  (c) silently produces a wrong/misleading result  <-- must fix

Usage:  python scripts/dirty_audit.py [--verbose]
Exit code 1 if any scenario is (b) or (c).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml.pipeline import Pipeline  # noqa: E402
from aml.schema_detect import detect_dataframe  # noqa: E402
from aml.loaders import load_any  # noqa: E402

SAMPLES = ROOT / "data" / "dirty_samples"


def check(name: str, expect: dict, verbose: bool = False) -> tuple:
    """Run the pipeline over the fixture and compare against expectations.

    ``expect`` keys:
      source: expected classification; slots: expected resolved slots;
      verdicts: {model: expected status}; n_entities: min entities;
      loaded_warning: warning substring the loader must emit (or None to ignore)
    """
    path = str(SAMPLES / name)
    problems = []
    try:
        loaded = load_any(path)
        det = detect_dataframe(loaded.df, file=path)

        if "loaded_warning" in expect:
            if expect["loaded_warning"] not in loaded.warning:
                problems.append(
                    f"loader warning {expect['loaded_warning']!r} not emitted "
                    f"(got {loaded.warning!r})")

        if expect.get("source") and det.source_type != expect["source"]:
            problems.append(f"source={det.source_type} (expected {expect['source']})")
        missing = (expect.get("slots") or set()) - det.resolved_slots
        if missing:
            problems.append(f"missing slots {sorted(missing)}; got {sorted(det.resolved_slots)}")

        if expect.get("amount_parse_min"):
            parsed = det.normalized["amount"].dropna()
            if parsed.size < expect["amount_parse_min"]:
                problems.append(f"only {parsed.size} amounts parsed (expected >= {expect['amount_parse_min']})")

        # full pipeline run (sufficiency + models + fusion)
        res = Pipeline(min_events_per_entity=30).run([path])
        for model, status in (expect.get("verdicts") or {}).items():
            got = res.sufficiency.get(model)
            if got is None:
                problems.append(f"verdict {model} missing")
            elif got.status != status:
                problems.append(f"{model}={got.status} (expected {status})")
        if expect.get("n_entities_min"):
            if res.unified["entity_id"].dropna().nunique() < expect["n_entities_min"]:
                problems.append(
                    f"only {res.unified['entity_id'].dropna().nunique()} entities "
                    f"(expected >= {expect['n_entities_min']})")
        if expect.get("undated_rows_max") is not None:
            undated = int(res.unified["timestamp"].isna().sum())
            if undated > expect["undated_rows_max"]:
                problems.append(f"{undated} undated rows (expected <= {expect['undated_rows_max']})")
    except Exception as e:  # noqa: BLE001 - harness must classify crash as (b)
        status = "(b) CRASH"
        problems.append(f"{type(e).__name__}: {str(e)[:120]}")
    else:
        status = "(c) SILENT-WRONG" if problems else "(a) ok"

    if verbose or problems or status.startswith("(b)"):
        for p in problems:
            print(f"   ! {p}")
    return status, problems


def audit(verbose: bool = False) -> list:
    cases = [
        # (name, expectations)
        ("renamed_mixed_case.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty", "location"},
            "verdicts": {"benford": "SUPPORTED", "structuring": "SUPPORTED",
                         "statml": "SUPPORTED", "network": "DEGRADED",
                         "time_correlation": "DEGRADED"},
        }),
        ("headers_es.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty", "actor_id"},
            "verdicts": {"benford": "SUPPORTED", "network": "DEGRADED",
                         "time_correlation": "DEGRADED"},
        }),
        ("unix_epoch.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty"},
            "verdicts": {"time_correlation": "DEGRADED", "statml": "SUPPORTED",
                         "benford": "SUPPORTED", "network": "DEGRADED"},
        }),
        ("mixed_dates.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty"},
            "undated_rows_max": 30,  # 12.5% junk strings unparseable -> NaN, honest
        }),
        ("currency_mess.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty"},
            "amount_parse_min": 55,  # "N/A" and "—" rows legitimately dropped
        }),
        ("missing_values.csv", {"source": "bank", "slots": {"timestamp", "amount"}}),
        ("dup_columns.csv", {"source": "bank", "slots": {"timestamp", "amount", "counterparty"}}),
        ("no_header.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty"},
            "n_entities_min": 4,
        }),
        ("latin1.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty"},
            "loaded_warning": "latin-1",
        }),
        ("semicolon.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty"},
            "loaded_warning": "delimiter",  # 'sniffed delimiter';'
        }),
        ("sparse.csv", {
            "source": "bank",
            "verdicts": {"time_correlation": "BLOCKED", "network": "BLOCKED",
                         "statml": "BLOCKED", "benford": "BLOCKED",
                         "structuring": "BLOCKED", "behavioral": "BLOCKED"},
        }),
        ("bank_only.csv", {
            "source": "bank", "slots": {"timestamp", "amount", "counterparty"},
            "verdicts": {"time_correlation": "DEGRADED", "network": "DEGRADED",
                         "benford": "SUPPORTED", "statml": "SUPPORTED"},
        }),
        ("cdr_only.csv", {
            "source": "cdr", "slots": {"timestamp", "counterparty", "duration", "location"},
            "verdicts": {"benford": "BLOCKED", "structuring": "BLOCKED",
                         "network": "DEGRADED", "statml": "DEGRADED"},
        }),
        ("social_only.csv", {
            "source": "social", "slots": {"timestamp", "counterparty"},
            "verdicts": {"network": "BLOCKED", "benford": "BLOCKED",
                         "structuring": "BLOCKED"},
        }),
    ]
    results = []
    print(f"{'file':24s} status")
    print("-" * 40)
    for name, expect in cases:
        status, problems = check(name, expect, verbose)
        results.append((name, status, problems))
        print(f"{name:24s} {status}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    results = audit(verbose=args.verbose)
    bad = [r for r in results if r[1] != "(a) ok"]
    print("-" * 40)
    print(f"{len(results) - len(bad)}/{len(results)} scenarios pass")
    for name, status, problems in bad:
        print(f"  {name:24s} {status}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())