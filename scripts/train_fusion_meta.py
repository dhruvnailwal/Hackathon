"""Train the meta-learner fusion model (offline, one command).

Generates several labelled synthetic datasets (the two-tier generator with
storyline + surprise answer keys), runs the full pipeline over each, builds
per-entity features (six model-family scores + behaviour profile) and trains
RandomForest vs XGBoost. The better model (higher CV average precision) is
persisted to src/aml/assets/fusion_meta.joblib and loaded by the pipeline at
analysis time — with an automatic Borda fallback if it is ever missing.

Also prints a quick holdout comparison: Borda vs meta-learner recall@10.

Usage:
    python scripts/train_fusion_meta.py [--seeds 100,101,102,103] [--out src/aml/assets/fusion_meta.joblib]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from aml.data_generator.emit import generate_all  # noqa: E402
from aml.fusion_meta import (  # noqa: E402
    build_meta_features, train_meta_classifier, save_meta_model,
)
from aml.models import run_model, fuse_rank_borda  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402
from eval import load_truth, map_truth_to_entities, to_records  # noqa: E402

FAMILIES = ("time_correlation", "network", "statml", "benford", "structuring",
            "behavioral")


def run_dataset(tmp: Path, seed: int, surprise_seed: int) -> tuple:
    """Run the pipeline over one generated dataset; return (X, y, rankings)."""
    generate_all(tmp, seed=seed, surprise_seed=surprise_seed,
                 n_background=100, n_per_storyline=3, n_surprise=10)
    paths = [str(p) for p in sorted((tmp / "sources").glob("*"))]
    res = Pipeline(min_events_per_entity=8).run(paths, with_models=True)
    truth = load_truth(str(tmp / "answer_key.json"))
    truth = map_truth_to_entities(truth, res.entity_map)
    anom, _ = to_records(res.rankings, truth)
    X = build_meta_features(res.model_scores, res.unified)
    y = pd.Series(1, index=list(anom)).reindex(X.index).fillna(0).astype(int)
    return X, y, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100,101,102,103")
    ap.add_argument("--out", default=str(ROOT / "src" / "aml" / "assets" / "fusion_meta.joblib"))
    ap.add_argument("--holdout-seed", type=int, default=104)
    args = ap.parse_args()

    all_X, all_y = [], []
    for s in [int(x) for x in args.seeds.split(",")]:
        with tempfile.TemporaryDirectory() as td:
            X, y, _ = run_dataset(Path(td), seed=s, surprise_seed=s + 100)
            all_X.append(X)
            all_y.append(y)
        print(f"seed {s}: {len(X)} entities, {int(y.sum())} anomalies")
    X = pd.concat(all_X)
    y = pd.concat(all_y)
    print(f"total: {len(X)} rows, {int(y.sum())} anomalies")

    clf, cv = train_meta_classifier(X, y)
    print(f"meta-learner CV average precision: {cv}")
    save_meta_model(clf, Path(args.out))
    print(f"saved -> {args.out}")

    # holdout sanity check: Borda vs meta on an unseen seed
    with tempfile.TemporaryDirectory() as td:
        X2, y2, res2 = run_dataset(Path(td), seed=args.holdout_seed,
                                   surprise_seed=args.holdout_seed + 100)
        clf2, _ = train_meta_classifier(X2, y2)  # fresh model on holdout data
        meta_rank = [{"entity_id": e, "score": float(s)} for e, s in
                     zip(X2.index, clf2.predict_proba(X2.values)[:, 1])]
        meta_rank.sort(key=lambda r: -r["score"])
        truth = load_truth(str(Path(td) / "answer_key.json"))
        truth = map_truth_to_entities(truth, res2.entity_map)
        anom, _ = to_records(res2.rankings, truth)
        n = max(len(anom), 1)
        b10 = sum(1 for r in res2.rankings[:10] if r["entity_id"] in anom)
        m10 = sum(1 for r in meta_rank[:10] if r["entity_id"] in anom)
        print(f"holdout seed {args.holdout_seed}: anomalies={len(anom)}")
        print(f"  Borda    recall@10 = {b10 / n:.3f}")
        print(f"  Meta     recall@10 = {m10 / n:.3f}")


if __name__ == "__main__":
    main()