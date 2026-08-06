"""One-command demo: generate synthetic data (CSV+JSON+TXT), run the pipeline,
write the report, and print a summary. Configuration comes from config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aml.config import PipelineConfig  # noqa: E402
from aml.data_generator.emit_multi import generate_all_mixed, make_social_text  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402
from aml.report import generate_report  # noqa: E402


def main() -> int:
    cfg = PipelineConfig.from_yaml(str(ROOT / "config" / "config.yaml"))

    print("== 1/3  Generating synthetic data (CSV + JSON + TXT) ==")
    generate_all_mixed(
        ROOT / cfg.data_dir,
        seed=cfg.seeds[0],
        surprise_seed=cfg.seeds[1] if len(cfg.seeds) > 1 else 99,
    )
    make_social_text(ROOT / cfg.data_dir)
    print("   wrote data/sources/*.csv/json/jsonl/txt + answer keys")

    out_dir = ROOT / (cfg.sources_dir or "data/sources")
    paths = sorted(p for p in out_dir.iterdir() if p.is_file()
                   and p.suffix.lower() in (".csv", ".json", ".jsonl", ".txt"))
    print(f"== 2/3  Running the pipeline on {len(paths)} files ==")

    pipe = Pipeline(
        min_events_per_entity=cfg.min_events_per_entity,
        tol_minutes=cfg.tol_minutes,
        structuring_threshold=cfg.structuring_threshold,
    )
    res = pipe.run([str(p) for p in paths], with_models=cfg.with_models)

    print("\nSources detected:")
    for rs in res.per_source:
        print(f"   {rs.source:8s} {Path(rs.file).name:24s} rows={rs.n_rows:<7,} entities={rs.n_entities}")
    print("Sufficiency verdicts:")
    for v in res.sufficiency.values():
        print(f"   {v.model:20s} {v.status:<9s} {v.reason}")
    print("\nTop-10 ranked insights:")
    for i, r in enumerate(res.rankings[:10], 1):
        print(f"   {i:2d}. {r['entity_id']}  score={r['score']:.3f}  n_events={r['n_events']}  fired={r['models_fired']}")

    print("\nExplanations (sample):")
    for eid in list(res.explanations)[:3]:
        print(f"   {eid}: {res.explanations[eid]}")

    written = generate_report(res, cfg, ROOT / (cfg.report_dir or "data/report"))
    print("\nReport written:")
    for kind, path in written.items():
        print(f"   {kind:8s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())