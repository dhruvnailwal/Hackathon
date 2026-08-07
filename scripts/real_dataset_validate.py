"""Validate the pipeline against a *real, labeled* money-laundering dataset.

Pulls the public AML training CSV from the Hugging Face Space
``Rithin08/Money-Laundering-Detection`` (AMLSim-derived, ~72.8k rows with
``Is Laundering`` labels) and runs the full auto pipeline over it.

This complements the internet-sample validation (``fetch_samples.py``):
those files prove *input robustness*, this one exercises a real financial
transfer table and records how the source-agnostic pipeline degrades when a
dataset lacks the entity/time dimensions the models need.

Usage:
    python scripts/real_dataset_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml.config import PipelineConfig  # noqa: E402
from aml.loaders import load_any  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402
from aml.schema_detect import detect_dataframe  # noqa: E402

NAME = "processed_dataset.csv"
URL = "https://huggingface.co/spaces/Rithin08/Money-Laundering-Detection/resolve/main/processed_dataset.csv"
DEST = ROOT / "data" / "internet_samples" / NAME
OUT = ROOT / "results" / "real_datasets"


def fetch() -> bool:
    if DEST.exists():
        print(f"   cached {NAME} ({DEST.stat().st_size:,} bytes)")
        return True
    req = Request(URL, headers={"User-Agent": "aml-sample-fetcher"})
    with urlopen(req, timeout=120) as r:
        DEST.write_bytes(r.read())
    print(f"  ok {NAME} ({DEST.stat().st_size:,} bytes)")
    return True


def main() -> int:
    print("== 1/3  Fetching real labeled AML dataset (Hugging Face Space) ==")
    try:
        fetch()
    except Exception as exc:  # pragma: no cover - network only
        print(f"   ! download failed: {exc}")
        return 2

    print("\n== 2/3  Schema auto-detection ==")
    ls = load_any(DEST)
    det = detect_dataframe(ls.df, file=ls.file)
    resolved = {k: v.slot for k, v in det.columns.items() if v.slot}
    print(f"  format={ls.format} shape={ls.df.shape} source_type={det.source_type}")
    for col, slot in resolved.items():
        print(f"    {col:16s} -> {slot}")

    print("\n== 3/3  Full pipeline run ==")
    cfg = PipelineConfig.from_yaml(str(ROOT / "config" / "config.yaml"))
    res = Pipeline(min_events_per_entity=cfg.min_events_per_entity).run(
        [str(DEST)], with_models=cfg.with_models
    )

    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": {"name": NAME, "url": URL, "bytes": DEST.stat().st_size},
        "detection": {
            "source_type": det.source_type,
            "resolved_slots": resolved,
            "unresolved_columns": [k for k, v in det.columns.items() if not v.slot],
            "warnings": det.warnings,
        },
        "labels": {
            str(k): int(v)
            for k, v in ls.df["Is Laundering"].value_counts().items()
        },
        "pipeline": {
            "rows": res.per_source[0].n_rows,
            "entities": res.per_source[0].n_entities,
            "warnings": res.per_source[0].warnings,
            "verdicts": {
                k: {"status": v.status, "reason": v.reason}
                for k, v in res.sufficiency.items()
                if v.model != "schema_detection"
            },
            "n_rankings": len(res.rankings),
        },
    }
    (OUT / "verdict.json").write_text(
        json.dumps(record, indent=2, default=str)
    )

    print("Verdicts:")
    for v in res.sufficiency.values():
        if v.model == "schema_detection":
            continue
        print(f"   {v.model:20s} {v.status:<9s} {v.reason[:75]}")
    print(f"\nRankings produced: {len(res.rankings)}")
    assert "rows" in record["pipeline"]


if __name__ == "__main__":
    raise SystemExit(main())