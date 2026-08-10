"""Evaluate the pipeline across the whole dataset suite.

Runs the full pipeline (with_models=True) over every dataset in
data/datasets/, scores the default Borda ranking AND the trained meta
fusion against each dataset's answer keys, and writes a per-mode summary
(mean/median recall@5/10, precision@10) plus a full metrics.json.

Usage:
    python scripts/dataset_suite_eval.py [--suite data/datasets]
                                         [--out results/dataset_suite]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# aml first so torch loads before the pandas/sklearn stack (Windows).
from aml.loaders import collect_files  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402

from eval import load_truth, map_truth_to_entities, to_records, recall_precision  # noqa: E402

MODES = ("full", "none", "pair_bank_cdr", "pair_bank_social", "pair_cdr_social")


def score_dataset(dataset: Path, fusion: str) -> dict:
    paths = collect_files([str(dataset / "sources")])
    pipe = Pipeline(fusion=fusion)
    res = pipe.run(paths, with_models=True)
    truth = map_truth_to_entities(
        load_truth(str(dataset / "answer_key.json")), res.entity_map)
    truth_anom, truth_type = to_records(res.rankings, truth)

    m = recall_precision(res.rankings, truth_anom, truth_type)
    surp = None
    s_path = dataset / "answer_key_surprise.json"
    if s_path.exists():
        s_truth = load_truth(str(s_path))
        s_mapped = map_truth_to_entities(s_truth, res.entity_map)
        surp_anom, _ = to_records(res.rankings, s_mapped, key_types=("surprise",))
        if surp_anom:
            surp = recall_precision(res.rankings, surp_anom, {})
    return {
        "n_entities": len(res.rankings),
        "n_anomalies": len(truth_anom),
        "recall@5": m["recall@5"], "recall@10": m["recall@10"],
        "precision@5": m["precision@5"], "precision@10": m["precision@10"],
        "surprise_recall@10": (surp or {}).get("recall@10"),
        "type_recall@10": dict(m["type_recall@10"]),
        "top10": res.rankings[:10],
        "sufficiency": {k: v.status for k, v in res.sufficiency.items()},
    }


def mean(vals) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(ROOT / "data" / "datasets"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dataset_suite"))
    ap.add_argument("--only-modes", nargs="*", default=list(MODES))
    args = ap.parse_args()

    suite = Path(args.suite)
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    datasets = [d for d in manifest["datasets"] if d["mode"] in args.only_modes]

    all_rows = []
    per_mode = {mode: {"borda": [], "meta": []} for mode in args.only_modes}
    for d in datasets:
        dataset = suite / d["name"]
        row = {"mode": d["mode"], "dataset": d["name"]}
        for fusion in ("borda", "meta"):
            t0 = time.time()
            try:
                m = score_dataset(dataset, fusion)
                m = {k: v for k, v in m.items() if k not in ("top10", "type_recall@10")}
                row[f"{fusion}_recall@10"] = m["recall@10"]
                row[f"{fusion}_recall@5"] = m["recall@5"]
                row[f"{fusion}_precision@10"] = m["precision@10"]
                per_mode[d["mode"]][fusion].append(m["recall@10"])
                status = "ok"
            except Exception as exc:  # pragma: no cover - defensive
                status = f"ERROR {type(exc).__name__}: {exc}"
            row["elapsed_s"] = round(time.time() - t0, 1)
            print(f"{d['mode']:16s} {d['name']:24s} {fusion:6s} "
                  f"R@10={row.get(f'{fusion}_recall@10', float('nan')):.3f} "
                  f"({status}) {row['elapsed_s']}s")
        all_rows.append(row)

    summary = {}
    for mode in args.only_modes:
        summary[mode] = {
            "n_datasets": len(per_mode[mode]["borda"]),
            "borda_mean_recall@10": round(mean(per_mode[mode]["borda"]), 3),
            "meta_mean_recall@10": round(mean(per_mode[mode]["meta"]), 3),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"datasets": all_rows, "summary": summary}, indent=2), encoding="utf-8")

    md = ["# Dataset suite evaluation",
          "",
          f"- datasets evaluated: {len(all_rows)}",
          "",
          "## Per-mode mean recall@10",
          "| mode | datasets | borda R@10 | meta R@10 |",
          "|---|---|---|---|"]
    for mode in args.only_modes:
        s = summary[mode]
        md.append(f"| {mode} | {s['n_datasets']} | {s['borda_mean_recall@10']:.3f} | "
                  f"{s['meta_mean_recall@10']:.3f} |")
    md += ["", "## Per dataset", "| mode | dataset | borda R@10 | meta R@10 |", "|---|---|---|---|"]
    for r in all_rows:
        md.append(f"| {r['mode']} | {r['dataset']} | "
                  f"{r.get('borda_recall@10', '-')} | {r.get('meta_recall@10', '-')} |")
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("=" * 60)
    print("SUMMARY (mean recall@10)")
    for mode, s in summary.items():
        print(f"  {mode:16s} borda={s['borda_mean_recall@10']:.3f} "
              f"meta={s['meta_mean_recall@10']:.3f}")
    print(f"stored -> {out / 'metrics.json'} , {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
