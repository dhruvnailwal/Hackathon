"""Hyperparameter sweep for the model families (plan §4.2).

One-at-a-time tuning: each cell replaces one model's hyperparameters in the
shipped ensemble (Borda fusion) and measures recall@5/@10 vs the answer key.
The sweep grid is deliberately small — 30 truth anomalies do not justify a
large search — and every result is recorded under results/tuning/.

Usage:
    python scripts/tune.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval import load_truth, map_truth_to_entities, to_records, recall_precision  # noqa: E402
from aml import models as _m  # noqa: E402
from aml.loaders import collect_files  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "results" / "tuning"


def sweep_grid(df, base, truth_anom, truth_type, grid, label):
    """One-at-a-time: replace ONE model with each hyperparameter cell."""
    rows = []
    for cfg in grid:
        scores = {k: v for k, v in base.items()}
        scores[label] = cfg["fit"](df)
        fusion = _m.fuse_rank_borda(scores, df)
        m = recall_precision(fusion.rank, truth_anom, truth_type)
        rows.append({
            "model": label,
            "params": {k: v for k, v in cfg.items() if k != "fit"},
            "recall@5": round(m["recall@5"], 3),
            "recall@10": round(m["recall@10"], 3),
            "precision@5": round(m["precision@5"], 3),
            "precision@10": round(m["precision@10"], 3),
        })
    best = max(rows, key=lambda r: (r["recall@10"], r["precision@10"]))
    return rows, best


def main() -> int:
    pipe = Pipeline()
    res = pipe.run(collect_files([str(ROOT / "data" / "sources")]), with_models=True)
    truth = map_truth_to_entities(
        load_truth(str(ROOT / "data" / "answer_key.json")), res.entity_map)
    truth_anom, truth_type = to_records(res.rankings, truth)
    base = dict(res.model_scores)

    grid_network = [
        {"fit": lambda df, h=36.0: _m.fit_oddball(df), "density": "abs-resid"},
    ]
    grid_statml = [
        {"fit": lambda df, c=0.03: _m.fit_statml(df, contamination=c), "contamination": c}
        for c in (0.03, 0.06, 0.1)
    ]
    grid_struct = [
        {"fit": lambda df, t=t, b=b: _m.fit_structuring(df, threshold=t, band=b),
         "threshold": t, "band": b}
        for t in (5000.0, 10000.0, 20000.0) for b in (0.85, 0.90)
    ]
    grid_time = [
        {"fit": lambda df, tm=tm: _m.fit_time_correlation(df, tolerance_minutes=tm),
         "tolerance_min": tm}
        for tm in (15, 25, 40)
    ]
    grid_behavioral = [
        {"fit": lambda df, g=g, q=q, s=s: _m.fit_behavioral(df, dormant_gap_days=g,
                                                            silence_amt_q=q, silence_days=s),
         "dormant_days": g, "silence_q": q, "silence_days": s}
        for g in (30, 45, 60) for q in (0.99, 0.995) for s in (10, 15)
    ]

    all_rows, all_best = [], []
    for label, grid in (("network", grid_network), ("statml", grid_statml),
                        ("structuring", grid_struct), ("time_correlation", grid_time),
                        ("behavioral", grid_behavioral)):
        rows, best = sweep_grid(res.unified, base, truth_anom, truth_type, grid, label)
        all_rows += rows
        all_best.append(best)
        for r in rows:
            print(f"  {label:16s} {r['params']}  rec@5={r['recall@5']} "
                  f"rec@10={r['recall@10']} prec@10={r['precision@10']}")
        print(f"  -> best {label}: {best['params']} rec@10={best['recall@10']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\nwritten -> {OUT / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
