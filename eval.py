"""Evaluation harness (plan §4).

`eval.py [--key data/answer_key.json] [--surprise-key data/answer_key_surprise.json]`

- runs the pipeline over data/sources
- maps answer-key identities onto pipeline entity IDs via the resolution table
- reports recall@k / precision@k (k=5,10), per-anomaly-type recall, and the
  leave-one-model-out ablation that validates fusion value (H1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aml.pipeline import Pipeline  # noqa: E402

K_LIST = (5, 10)


def load_truth(key_path: str) -> pd.DataFrame:
    with open(key_path) as fh:
        data = json.load(fh)
    rows = []
    for pid, info in data["entities"].items():
        rows.append({
            "truth_person_id": pid,
            "type": info.get("person_type", "background"),
            "storyline": info.get("storyline", ""),
            "name": info.get("name", ""),
            "phone": info.get("phone", ""),
            "account": info.get("account", ""),
            "handle": info.get("handle", ""),
        })
    return pd.DataFrame(rows)


def map_truth_to_entities(truth: pd.DataFrame, entity_map) -> pd.DataFrame:
    """Map each truth person onto pipeline entity IDs via shared identifiers."""
    id_to_entity = {str(k): v for k, v in entity_map.entity_id_by_identifier.items()}
    name_to_entity = {str(k): v for k, v in entity_map.name_to_entity.items()}

    def resolve(row):
        found = set()
        for key in ("account", "phone", "handle"):
            ident = str(row.get(key, ""))
            if ident and ident in id_to_entity:
                found.add(id_to_entity[ident])
        if not found and row.get("name"):
            name = str(row["name"])
            if name in name_to_entity:
                found.add(name_to_entity[name])
            else:
                m, _ = process.extractOne(name, list(name_to_entity),
                                          scorer=fuzz.token_sort_ratio,
                                          score_cutoff=88.0)
                if m:
                    found.add(name_to_entity[m])
        return found

    truth["entity_ids"] = truth.apply(resolve, axis=1)
    return truth


def to_records(rankings, truth, key_types=("storyline", "surprise")):
    """entity_id -> is_anomaly (and type)."""
    truth_anom = set()
    truth_type = {}
    for _, r in truth.iterrows():
        for eid in r["entity_ids"]:
            if r["type"] in key_types:
                truth_anom.add(eid)
                truth_type[eid] = r.get("storyline") or r["type"]
    # entities that appear anomalous but got no entity mapping cannot be scored
    return truth_anom, truth_type


def recall_precision(rankings, truth_anom, truth_type, ks=K_LIST):
    out = {}
    for k in ks:
        top = rankings[:k]
        hit = sum(1 for r in top if r["entity_id"] in truth_anom)
        out[f"recall@{k}"] = hit / max(len(truth_anom), 1)
        out[f"precision@{k}"] = hit / k
    # per-type recall (how many of each anomaly type got into top-10)
    top10 = {r["entity_id"] for r in rankings[:10]}
    type_recall = {}
    for t in set(truth_type.values()):
        ids = {e for e, ty in truth_type.items() if ty == t}
        if ids:
            type_recall[t] = len(ids & top10) / len(ids)
    out["type_recall@10"] = type_recall
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sources")
    ap.add_argument("--key", default="data/answer_key.json")
    ap.add_argument("--surprise-key", default="data/answer_key_surprise.json")
    args = ap.parse_args()

    from aml.loaders import collect_files
    paths = collect_files([args.data])
    pipe = Pipeline()
    res = pipe.run(paths, with_models=True)

    truth = load_truth(args.key)
    truth = map_truth_to_entities(truth, res.entity_map)

    truth_anom, truth_type = to_records(res.rankings, truth)
    metrics = recall_precision(res.rankings, truth_anom, truth_type)

    print("=" * 60)
    print("PIPELINE SUMMARY")
    for rs in res.per_source:
        print(f"  {rs.source:8s} {rs.file}  rows={rs.n_rows} entities={rs.n_entities}")
    print("Sufficiency:")
    for k, v in res.sufficiency.items():
        print(f"  {k:18s} {v.status}")
    print("=" * 60)
    print(f"EVALUATION  (anomalies = {len(truth_anom)} truth entities, top10 = "
          f"{len(res.rankings[:10])})")
    for k, v in metrics.items():
        if k == "type_recall@10":
            print(f"  recall@10 by type:")
            for t, r in sorted(v.items()):
                print(f"    {t:16s} {r:.2f}")
        else:
            print(f"  {k:14s} {v:.3f}")
    print("=" * 60)
    print("TOP 10 RANKINGS")
    for i, r in enumerate(res.rankings[:10], 1):
        flag = "  <-- ANOMALY" if r["entity_id"] in truth_anom else ""
        print(f"  {i:2d}. {r['entity_id']}  score={r['score']:.3f}  fired={r['models_fired']}{flag}")

    # ---- ablation: leave-one-model-out --------------------------------
    print("=" * 60)
    print("ABLATION (leave-one-model-out, recall@10)")
    full_metrics = metrics
    for model, scores in res.model_scores.items():
        others = {m: s for m, s in res.model_scores.items() if m != model}
        # fused ranking without `model` (equal-weight mean)
        ids = set()
        for s in others.values():
            ids.update(s.keys())
        agg = {}
        for eid in ids:
            vals = [s.get(eid, 0.0) for s in others.values()]
            agg[eid] = float(np.mean(vals))
        ranking = sorted(agg.items(), key=lambda kv: -kv[1])
        top10 = [e for e, _ in ranking[:10]]
        hit = sum(1 for e in top10 if e in truth_anom)
        print(f"  -{model:16s} recall@10={hit/max(len(truth_anom),1):.3f}  "
              f"(full={full_metrics['recall@10']:.3f})")


if __name__ == "__main__":
    main()