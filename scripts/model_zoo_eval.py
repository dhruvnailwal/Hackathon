"""Model-zoo evaluation: candidate models vs. the pipeline baseline.

Runs the full pipeline over data/sources, then re-scores the unified
dataframe with the candidate models:

  Stage E (stat/ML)  : IsolationForest (baseline) vs EIF vs LOF
  Stage D (network)  : burst-concentration (baseline) vs OddBall density
  Stage F (fusion)   : top-2 weighted (baseline) vs Borda rank fusion

Every candidate scores entities identically ({entity_id: float}); recall /
precision vs. the answer key decide whether a candidate deserves to replace
the baseline in the shipped pipeline.

Usage:
    python scripts/model_zoo_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval import load_truth, map_truth_to_entities, to_records, recall_precision  # noqa: E402
from aml.loaders import collect_files  # noqa: E402
from aml.models import (  # noqa: E402
    fit_network, fit_oddball, fit_statml, fit_statml_eif, fit_statml_lof,
    fuse_model_scores, fuse_rank_borda,
)
from aml.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "results" / "model_zoo"
BASE_ENSEMBLE = ["time_correlation", "network", "statml", "benford", "structuring", "behavioral"]


def ranking_from_scores(scores) -> list:
    return sorted(scores.items(), key=lambda kv: -kv[1])


def evalscore(scores: dict, truth_anom: set, truth_type: dict) -> dict:
    rank = [{"entity_id": e, "score": s} for e, s in ranking_from_scores(scores)]
    return recall_precision(rank, truth_anom, truth_type)


def main() -> int:
    paths = collect_files([str(ROOT / "data" / "sources")])
    pipe = Pipeline()
    res = pipe.run(paths, with_models=True)
    df = res.unified

    truth = map_truth_to_entities(load_truth(str(ROOT / "data" / "answer_key.json")), res.entity_map)
    truth_anom, truth_type = to_records(res.rankings, truth)
    n_anom = len(truth_anom)
    print(f"unified rows={len(df):,} entities={df['entity_id'].nunique():,} "
          f"truth anomalies={n_anom}")

    # ---- per-model candidates ----------------------------------------------
    stats = {
        "statml_baseline": fit_statml(df),
        "statml_eif": fit_statml_eif(df),
        "statml_lof": fit_statml_lof(df),
        "network_baseline": fit_network(df),
        "network_oddball": fit_oddball(df),
    }
    for name, s in stats.items():
        m = evalscore(s, truth_anom, truth_type)
        print(f"  {name:18s} recall@5={m['recall@5']:.3f} recall@10={m['recall@10']:.3f} "
              f"prec@10={m['precision@10']:.3f}")

    # ---- ensemble variants (replace one family at a time) -------------------
    base = dict(res.model_scores)
    variants = {
        "baseline_ensemble": base,
        "statml->eif": {**base, "statml": stats["statml_eif"]},
        "statml->lof": {**base, "statml": stats["statml_lof"]},
        "network->oddball": {**base, "network": stats["network_oddball"]},
        "statml->eif+network->oddball": {**base, "statml": stats["statml_eif"],
                                         "network": stats["network_oddball"]},
    }
    report = []
    header = f"| ensemble | fusion | recall@5 | recall@10 | precision@5 | precision@10 |"
    sep = "|---|---|---|---|---|---|"
    report.append(header)
    report.append(sep)
    scores_borda, scores_top2 = {}, {}
    for vname, scores in variants.items():
        f_top2 = fuse_model_scores(scores, df)
        f_borda = fuse_rank_borda(scores, df)
        m_top2 = recall_precision(f_top2.rank, truth_anom, truth_type)
        m_borda = recall_precision(f_borda.rank, truth_anom, truth_type)
        scores_top2[vname] = m_top2
        scores_borda[vname] = m_borda
        report.append(
            f"| {vname} | top2 | {m_top2['recall@5']:.3f} | {m_top2['recall@10']:.3f} "
            f"| {m_top2['precision@5']:.3f} | {m_top2['precision@10']:.3f} |"
        )
        report.append(
            f"| {vname} | borda | {m_borda['recall@5']:.3f} | {m_borda['recall@10']:.3f} "
            f"| {m_borda['precision@5']:.3f} | {m_borda['precision@10']:.3f} |"
        )

    # ---- winner -------------------------------------------------------------
    best = max(scores_borda, key=lambda k: scores_borda[k]["recall@10"])
    best_top2 = max(scores_top2, key=lambda k: scores_top2[k]["recall@10"])
    print(f"\nBest Borda ensemble: {best} recall@10={scores_borda[best]['recall@10']:.3f}")
    print(f"Best top2  ensemble: {best_top2} recall@10={scores_top2[best_top2]['recall@10']:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps({
        "truth_anomalies": n_anom,
        "per_model": {k: evalscore(v, truth_anom, truth_type) for k, v in stats.items()},
        "fusion_top2": scores_top2,
        "fusion_borda": scores_borda,
        "winner_borda": best,
        "winner_top2": best_top2,
    }, indent=2, default=str), encoding="utf-8")
    print(f"written -> {OUT / 'report.md'}, {OUT / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
