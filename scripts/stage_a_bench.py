"""Stage A benchmark (plan §3): header-only vs content-only vs hybrid detection.

Runs every case of the stress matrix under all three detection modes and
reports, per mode:
  - slot adoption  : fraction of expected slots that resolved (recall)
  - slot accuracy  : fraction of resolved slots that map to the true attribute
                     (precision; an actor-ID column misread as counterparty is
                     a *wrong* resolution for content probing)
  - source accuracy: classification matches the generating source
  - rescue cases   : files where header-only fails but hybrid succeeds (H2)

Usage:
    python scripts/stage_a_bench.py
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aml.loaders import load_any
from aml.schema_detect import detect_dataframe
from stage_a_stress import (FORMATS, HEADER_STYLES, SUBSETS, ATTRS,  # noqa: E402
                            _header_for, build_rows)

MODES = ("header", "content", "hybrid")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=0, help="0 = full matrix")
    ap.add_argument("--out", default=str(ROOT / "results" / "stage_a_bench"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for source, subsets in SUBSETS.items():
        for label, attrs, n_rows in subsets:
            for hstyle in HEADER_STYLES:
                for fmt_name in FORMATS:
                    cases.append((source, label, attrs, n_rows, hstyle, fmt_name))
    if args.cases:
        random.Random(0).shuffle(cases)
        cases = cases[: args.cases]

    tmp = Path(tempfile.mkdtemp(prefix="stage_a_bench_"))
    stats = {m: {"adopted": 0, "expected": 0, "wrong": 0,
                 "source_ok": 0, "source_total": 0} for m in MODES}
    rescue: List[dict] = []
    failures: List[dict] = []
    try:
        for idx, (source, label, attrs, n_rows, hstyle, fmt_name) in enumerate(cases):
            header = _header_for(attrs, hstyle)
            rows = build_rows(attrs, n_rows, header=header, sparse=label == "sparse_null")
            path = FORMATS[fmt_name](tmp / f"case{idx}", header, rows)
            loaded = load_any(str(path))

            per_mode: Dict[str, dict] = {}
            col_attr = {h: a for h, a in zip(header, attrs)}
            slot_attr = {ATTRS[a]["slot"]: a for a in attrs if ATTRS[a]["slot"]}
            expected_slots = set(slot_attr)
            per_mode: Dict[str, dict] = {}
            for mode in MODES:
                det = detect_dataframe(loaded.df, file=path.name, mode=mode)
                resolved = {}
                for rc in det.columns.values():
                    if rc.slot:
                        resolved.setdefault(rc.slot, rc.original)
                stats[mode]["expected"] += len(expected_slots)
                adopted = len(expected_slots & set(resolved))
                stats[mode]["adopted"] += adopted
                wrong = 0
                for slot, col in resolved.items():
                    if col not in col_attr:
                        continue  # unmapped in ground truth (e.g. noise) — not a wrong claim
                    actual_attr = col_attr[col]
                    if slot not in expected_slots or actual_attr != slot_attr[slot]:
                        wrong += 1
                stats[mode]["wrong"] += wrong
                if det.source_type == source:
                    stats[mode]["source_ok"] += 1
                stats[mode]["source_total"] += 1
                per_mode[mode] = {
                    "source": det.source_type,
                    "resolved": sorted(resolved),
                    "expected": sorted(expected_slots),
                }

            # H2 rescue: header-only failed, hybrid succeeded
            missing_header = expected_slots - set(per_mode["header"]["resolved"])
            missing_hybrid = expected_slots - set(per_mode["hybrid"]["resolved"])
            rescued = missing_header - missing_hybrid
            if rescued:
                rescue.append({
                    "case": idx, "source": source, "subset": label,
                    "header_style": hstyle, "format": fmt_name,
                    "rescued_slots": sorted(rescued),
                    "attrs": attrs,
                })
            if not per_mode["hybrid"]["resolved"]:
                failures.append({"case": idx, "source": source, "subset": label,
                                 "header_style": hstyle, "format": fmt_name})
            (out_dir / f"case_{idx:04d}.json").write_text(
                json.dumps(per_mode, indent=2, default=str), encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ----- report -----------------------------------------------------------
    lines = ["# Stage A benchmark — detection-mode comparison",
             "", f"generated: {pd.Timestamp.now()}",
             f"cases: {len(cases)}", ""]
    lines.append("| mode | adoption (recall) | wrong (precision loss) | source accuracy |")
    lines.append("|---|---|---|---|")
    for m in MODES:
        s = stats[m]
        adopt = s["adopted"] / max(s["expected"], 1)
        wrong = s["wrong"] / max(s["adopted"], 1)
        src = s["source_ok"] / max(s["source_total"], 1)
        lines.append(f"| {m:8s} | {adopt:.3f} ({s['adopted']}/{s['expected']}) "
                     f"| {wrong:.3f} ({s['wrong']}/{s['adopted']}) | {src:.3f} |")
    lines.append("")
    lines.append(f"## H2 rescue cases (header-only misses, hybrid catches): {len(rescue)}")
    if rescue:
        lines.append("")
        lines.append("| case | source | subset | header style | format | rescued slots |")
        lines.append("|---|---|---|---|---|---|")
        for r in rescue:
            lines.append(f"| {r['case']} | {r['source']} | {r['subset']} | "
                         f"{r['header_style']} | {r['format']} | {','.join(r['rescued_slots'])} |")
    lines.append("")
    lines.append(f"## Zero-resolution cases (hybrid resolves nothing): {len(failures)}")
    for f in failures:
        lines.append(f"- {f['source']}/{f['subset']}/{f['header_style']}/{f['format']}")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Stage A bench: {len(cases)} cases -> {out_dir}")
    for line in lines[4:11]:
        print(line)
    print(f"  H2 rescue cases: {len(rescue)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
