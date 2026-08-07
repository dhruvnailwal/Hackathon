"""Research-format validation (§4 / hardening).

Generates the SAME scenario in four real-world layouts (CSV / JSONL / kv.log /
prose.log) and asserts the pipeline reaches near-identical recall on each.
A detector that really understands untabular data must recover comparable
evidence from every layout; an extractor that only works on its own CSV
round-trip fails this check.

Usage:  python scripts/research_formats_validate.py [--out results/research_formats]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aml.data_generator.emit_research import emit_formats  # noqa: E402
from aml.loaders import collect_files  # noqa: E402
from aml.pipeline import Pipeline  # noqa: E402
from eval import load_truth, map_truth_to_entities, recall_precision, to_records  # noqa: E402

FORMATS = ("csv", "jsonl", "kvlog", "prose")
TOLERANCE = 0.12  # any format within +-0.12 of the CSV baseline (main recall@10)


def score_folder(folder: Path, key: Path, surprise_key: Path) -> dict:
    paths = collect_files([str(folder)])
    res = Pipeline().run(paths, with_models=True)

    truth = map_truth_to_entities(load_truth(str(key)), res.entity_map)
    truth_anom, truth_type = to_records(res.rankings, truth)
    main = recall_precision(res.rankings, truth_anom, truth_type)

    surprise = None
    if surprise_key.exists():
        s_mapped = map_truth_to_entities(load_truth(str(surprise_key)), res.entity_map)
        s_anom, s_type = to_records(res.rankings, s_mapped, key_types=("surprise",))
        if s_anom:
            surprise = recall_precision(res.rankings, s_anom, s_type)

    return {
        "recall@10": main["recall@10"],
        "precision@10": main["precision@10"],
        "recall@5": main["recall@5"],
        "surprise_recall@10": surprise and surprise["recall@10"],
        "sufficiency": {k: v.status for k, v in res.sufficiency.items()},
        "n_entities": len(res.rankings),
        "top10": res.rankings[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/research_formats")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="rfx_"))
    emit_formats(work, seed=args.seed)
    key = work / "answer_key.json"
    surp = work / "answer_key_surprise.json"

    results = {}
    for name in FORMATS:
        results[name] = score_folder(work / name / "sources", key, surp)

    baseline = results["csv"]["recall@10"]
    failures = []
    for name in FORMATS[1:]:
        rec = results[name]["recall@10"]
        if abs(rec - baseline) > TOLERANCE:
            failures.append(f"{name}: recall@10={rec:.3f} vs csv={baseline:.3f}")

    # snapshot + report
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(
        json.dumps({"baseline_csv": baseline, "results": results}, indent=2, default=str),
        encoding="utf-8")
    md = ["# Research-format validation",
          f"| format | recall@10 | precision@10 | surprise recall@10 |",
          "|---|---|---|--|"]
    for name in FORMATS:
        r = results[name]
        md.append(f"| {name} | {r['recall@10']:.3f} | {r['precision@10']:.3f} "
                  f"| {r['surprise_recall@10'] if r['surprise_recall@10'] else '-'} |")
    (out / "eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("=" * 60)
    print("RESEARCH-FORMAT VALIDATION")
    for name in FORMATS:
        r = results[name]
        print(f"  {name:8s} recall@10={r['recall@10']:.3f} precision@10={r['precision@10']:.3f} "
              f"surprise={r['surprise_recall@10']}")
    print("=" * 60)
    shutil.rmtree(work, ignore_errors=True)
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("OK: all formats within tolerance of CSV baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())