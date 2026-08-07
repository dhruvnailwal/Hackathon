"""Model matrix evaluation — test every candidate model per analysis stage.

For each of the six model families (time correlation / network / stat-ML /
benford / structuring / behavioral) plus the fusion layer, generate ALL
candidate variants and score each against the answer key (recall@5/10,
precision@5/10, per-type recall). Then evaluate ensemble substitutions and
fusion algorithms, and persist the full matrix to results/model_zoo_matrix/.

Unlike model_zoo_eval.py (which benchmarks a fixed handful), this script is
the exhaustive "try different models per stage" harness.

Usage:
    python scripts/model_matrix_eval.py [--out results/model_zoo_matrix]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval import load_truth, map_truth_to_entities, to_records, recall_precision  # noqa: E402
from aml.loaders import collect_files  # noqa: E402
from aml.models import (  # noqa: E402
    run_model,
    fuse_rank_borda, fuse_rank_average, fuse_score_mean, fuse_weighted_sum,
    fuse_model_scores,
)
from aml.pipeline import Pipeline  # noqa: E402

# families: name -> list of candidate model names (run_model dispatch)
FAMILIES = {
    "time_correlation": ["time_correlation", "time_correlation_backward",
                         "time_correlation_anypair"],
    "network": ["oddball", "oddball_signed", "burst", "degree_deviation",
                "reciprocity"],
    "statml": ["statml", "statml_eif", "statml_lof", "statml_mahalanobis",
               "statml_pca", "statml_zscore"],
    "benford": ["benford", "benford_ks"],
    "structuring": ["structuring", "structuring_banded"],
    "behavioral": ["behavioral"],
}
# which model name fills the default position for a family (fallback baselines)
FAMILY_DEFAULT = {
    "time_correlation": "time_correlation",
    "network": "oddball",
    "statml": "statml",
    "benford": "benford",
    "structuring": "structuring",
    "behavioral": "behavioral",
}

FUSIONS = {
    "top2": lambda s, df: fuse_model_scores(s, df),
    "borda": lambda s, df: fuse_rank_borda(s, df),
    "borda_nobonus": lambda s, df: fuse_rank_borda(s, df, max_bonus=0.0),
    "rank_avg": lambda s, df: fuse_rank_average(s, df),
    "score_mean": lambda s, df: fuse_score_mean(s, df),
}


def evalscore(scores: dict, truth_anom: set, truth_type: dict) -> dict:
    rank = [{"entity_id": e, "score": s} for e, s in
            sorted(scores.items(), key=lambda kv: -kv[1])]
    return recall_precision(rank, truth_anom, truth_type)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/model_zoo_matrix")
    args = ap.parse_args()

    paths = collect_files([str(ROOT / "data" / "sources")])
    pipe = Pipeline()
    res = pipe.run(paths, with_models=True)
    df = res.unified

    truth = map_truth_to_entities(load_truth(str(ROOT / "data" / "answer_key.json")), res.entity_map)
    truth_anom, truth_type = to_records(res.rankings, truth)
    truth_surp = map_truth_to_entities(load_truth(str(ROOT / "data" / "answer_key_surprise.json")), res.entity_map)
    surp_anom, surp_type = to_records(res.rankings, truth_surp, key_types=("surprise",))
    print(f"unified rows={len(df):,} entities={df['entity_id'].nunique():,} "
          f"anomalies={len(truth_anom)} surprise={len(surp_anom)}")

    matrix = {"meta": {"rows": len(df), "entities": int(df["entity_id"].nunique()),
                       "anomalies": len(truth_anom)},
              "families": {}, "ensembles": {}, "best": {}}

    # ---- 1. per-model scores (compute once, reuse in ensembles) ----------
    scores: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for fam, cands in FAMILIES.items():
        matrix["families"][fam] = {}
        for name in cands:
            try:
                scores[name] = run_model(name, df)
                m = evalscore(scores[name], truth_anom, truth_type)
                s_m = evalscore(scores[name], surp_anom, surp_type) if surp_anom else None
                matrix["families"][fam][name] = {
                    **m,
                    "surprise_recall@10": s_m and s_m["recall@10"],
                }
                print(f"  {name:28s} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f} "
                      f"P@10={m['precision@10']:.3f}")
            except Exception as exc:  # survive a broken candidate
                scores[name] = {}
                errors[name] = f"{type(exc).__name__}: {exc}"
                matrix["families"][fam][name] = {"error": str(exc)}
                print(f"  {name:28s} ERROR {exc}")

    # ---- 2. best single model per family ----------------------------------
    per_family_best: dict[str, str] = {}
    for fam, cands in FAMILIES.items():
        ranked = sorted(
            (c for c in cands
             if matrix["families"][fam].get(c, {}).get("recall@10") is not None),
            key=lambda c: -matrix["families"][fam][c]["recall@10"])
        if ranked:
            per_family_best[fam] = ranked[0]

    # ---- 3. ensembles: swap one family at a time ---------------------------
    base = {"time_correlation": scores["time_correlation"],
            "network": scores["oddball"],
            "statml": scores["statml"],
            "benford": scores["benford"],
            "structuring": scores["structuring"],
            "behavioral": scores["behavioral"]}
    matrix["ensembles"]["baseline"] = {}

    # 3a. replace each family with its best candidate (all-upgrade ensemble)
    upgrade = dict(base)
    for fam, best in per_family_best.items():
        if best and best != FAMILY_DEFAULT[fam]:
            upgrade[FAMILY_DEFAULT[fam]] = scores[best]
    ensembles = {"baseline": base, "best_per_family": upgrade}
    for fam in FAMILIES:
        best = per_family_best.get(fam)
        if best and best != FAMILY_DEFAULT[fam]:
            variant = dict(base)
            variant[FAMILY_DEFAULT[fam]] = scores[best]
            label = f"swap_{fam}_to_{best}"
            ensembles[label] = variant

    for vname, v_scores in ensembles.items():
        matrix["ensembles"][vname] = {}
        for fname in FUSIONS:
            rank = FUSIONS[fname](dict(v_scores), df).rank
            m = recall_precision(rank, truth_anom, truth_type)
            s_m = recall_precision(rank, surp_anom, surp_type) if surp_anom else None
            matrix["ensembles"][vname][fname] = {**m, "surprise_recall@10": s_m and s_m["recall@10"]}
            print(f"  {vname:24s}/{fname:12s} R@10={m['recall@10']:.3f} P@10={m['precision@10']:.3f}")

    # ---- 4. winner ----------------------------------------------------------
    best_ens, best_fus, best_r10 = None, None, -1.0
    for vname, fc in matrix["ensembles"].items():
        for fname, m in fc.items():
            if m.get("recall@10") is not None and m["recall@10"] > best_r10:
                best_r10 = m["recall@10"]
                best_ens, best_fus = vname, fname
    matrix["best"] = {"ensemble": best_ens, "fusion": best_fus, "recall@10": best_r10}
    matrix["errors"] = errors

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(matrix, indent=2, default=str), encoding="utf-8")

    md = ["# Model matrix (per-stage candidates + fusion)",
          f"- rows={len(df):,} entities={df['entity_id'].nunique():,} "
          f"anomalies={len(truth_anom)} surprise={len(surp_anom)}",
          ""]
    for fam, cands in FAMILIES.items():
        md.append(f"## {fam}")
        md.append("| model | recall@5 | recall@10 | precision@10 | surprise@10 |")
        md.append("|---|---|---|---|---|")
        for c in cands:
            m = matrix["families"][fam].get(c) or {}
            if "error" in m:
                md.append(f"| {c} | error: {m['error']} | | | |")
                continue
            md.append(f"| {c} | {m.get('recall@5', '-'):.3f} | {m.get('recall@10', '-'):.3f} | "
                      f"{m.get('precision@10', '-'):.3f} | "
                      f"{m.get('surprise_recall@10', '-') if m.get('surprise_recall@10') is not None else '-'} |")
        md.append("")
    md.append("## Ensembles × fusion recall@10")
    md.append("| ensemble | " + " | ".join(FUSIONS) + " |")
    md.append("|" + "---|" * (len(FUSIONS) + 1))
    for vname, fc in matrix["ensembles"].items():
        cells = []
        for f in FUSIONS:
            v = fc.get(f, {}).get("recall@10")
            cells.append(f"{v:.3f}" if v is not None else "-")
        md.append(f"| {vname} | " + " | ".join(cells) + " |")
    md.append("")
    md.append(f"**Winner**: ensemble={best_ens} fusion={best_fus} recall@10={best_r10:.3f}")
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("=" * 60)
    print("WINNER:", best_ens, best_fus, f"recall@10={best_r10:.3f}")
    print(f"stored -> {out / 'metrics.json'} , {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())