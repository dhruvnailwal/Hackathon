"""One-command analysis: auto-detect schema, auto-run models, write a report.

Usage:
    python run_pipeline.py --config config/config.yaml [data/...] [file.csv ...]

Pass zero or more files/directories. Any CSV / JSON / JSONL / TXT /LOG file is
loaded, columns are auto-detected, the models the data can support are run
automatically, and a markdown + JSON report is written to the configured dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aml.config import PipelineConfig  # noqa: E402
from aml.loaders import collect_files  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402
from aml.report import generate_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("inputs", nargs="*", help="files or directories to analyze")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    cfg.report_dir = cfg.report_dir or "data/report"

    if args.inputs:
        paths = collect_files(args.inputs)
    else:
        src_dir = ROOT / (cfg.sources_dir or "data/sources")
        paths = collect_files([str(src_dir)])

    if not paths:
        print("No supported input files found.", file=sys.stderr)
        return 2

    print(f"Loaded {len(paths)} file(s):")
    for p in paths:
        print(f"   - {Path(p).name}")

    pipe = Pipeline(
        min_events_per_entity=cfg.min_events_per_entity,
        tol_minutes=cfg.tol_minutes,
        structuring_threshold=cfg.structuring_threshold,
        probe_cfg=cfg.probe_config(),
    )
    result = pipe.run(paths, with_models=cfg.with_models)

    print("\nSources detected:")
    for rs in result.per_source:
        print(f"   {rs.source:8s} {Path(rs.file).name:24s} rows={rs.n_rows:<6,} "
              f"entities={rs.n_entities}")
    print("Model verdicts:")
    for v in result.sufficiency.values():
        if v.model == "schema_detection":
            continue
        print(f"   {v.model:20s} {v.status:<9s} {v.reason[:80]}")

    written = generate_report(result, cfg, cfg.report_dir)

    print("\nRanked insights (top 10):")
    for i, r in enumerate(result.rankings[:10], 1):
        print(f"   {i:2d}. {r['entity_id']}  score={r['score']:.3f}  fired={r['models_fired']}")
    print("\nReport written:")
    for kind, path in written.items():
        print(f"   {kind:8s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())