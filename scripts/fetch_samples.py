"""Fetch real public sample files from the internet, then run the whole
pipeline (auto schema detection -> auto models -> report) over them.

This demonstrates input robustness on *real-world* data pulled from the web
rather than only synthetic files.

Usage:
    python scripts/fetch_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml.config import PipelineConfig  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402
from aml.report import generate_report  # noqa: E402

SAMPLES = {
    # real CSV
    "tips.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
    # real JSON (array of records)
    "users.json": "https://jsonplaceholder.typicode.com/users",
    # real plain text
    "pandas_readme.txt": "https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md",
    # real compact CSV (weather): Date + numeric measurements
    "seattle_weather.csv": "https://raw.githubusercontent.com/plotly/datasets/master/2016-weather-data-seattle.csv",
}


def fetch_one(url: str, dest: Path) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "aml-sample-fetcher"})
        with urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as exc:
        print(f"   ! failed {Path(dest).name}: {exc}")
        return False


def main() -> int:
    out_dir = ROOT / "data" / "internet_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== 1/3  Downloading real sample files from the internet ==")
    files = []
    for name, url in SAMPLES.items():
        dest = out_dir / name
        if fetch_one(url, dest):
            files.append(dest)
            print(f"   ok {name} ({dest.stat().st_size:,} bytes)")

    if not files:
        print("No internet samples fetched; nothing to run.", file=sys.stderr)
        return 2

    print(f"\n== 2/3  Running the pipeline over {len(files)} real files ==")
    cfg = PipelineConfig.from_yaml(str(ROOT / "config" / "config.yaml"))
    res = Pipeline(min_events_per_entity=cfg.min_events_per_entity).run(
        [str(p) for p in files], with_models=cfg.with_models
    )

    print("\nSources detected:")
    for rs in res.per_source:
        print(f"   {rs.source:8s} {Path(rs.file).name:24s} rows={rs.n_rows:<7,} entities={rs.n_entities}")
        for w in rs.warnings:
            print(f"       warn: {w}")
    print("Model verdicts:")
    for v in res.sufficiency.values():
        if v.model in ("schema_detection",):
            continue
        print(f"   {v.model:20s} {v.status:<9s} {v.reason[:70]}")
    print("Top rankings:", [(r['entity_id'], round(r['score'], 3)) for r in res.rankings[:8]])

    written = generate_report(res, cfg, ROOT / (cfg.report_dir or "data/report"))
    print("\n== 3/3  Report ==")
    for kind, path in written.items():
        print(f"   {kind:8s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
