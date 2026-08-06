"""Standard report generation (markdown + JSON) for a PipelineResult.

Writes a plain, audit-friendly analysis report so the same run can be
shared as a file without the desktop UI.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _slot_table(per_source) -> List[dict]:
    rows = []
    for rs in per_source:
        for slot in sorted(rs.resolved_fields):
            rows.append({"source": rs.source, "file": Path(rs.file).name,
                         "slot": slot, "status": "resolved"})
    return rows


def generate_report(result, config, report_dir: str | Path = "data/report") -> dict:
    """Build and write the report. Returns a dict of written file paths."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    markdown = _markdown(result, config)
    json_body = _json(result, config)

    (report_dir / "report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "report.json").write_text(
        json.dumps(json_body, indent=2, default=str), encoding="utf-8"
    )
    return {
        "markdown": str(report_dir / "report.md"),
        "json": str(report_dir / "report.json"),
    }


def _statuses(res) -> Dict[str, str]:
    return {v.model: v.status for v in res.sufficiency.values()}


def _markdown(res, cfg) -> str:
    L: List[str] = []
    L.append("# Anomaly Lens — analysis report")
    L.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    L.append(f"Source files: `{cfg.sources_dir}`")
    unified = res.unified
    n_entities = unified["entity_id"].nunique() if not unified.empty else 0
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| events | {len(unified):,} |")
    L.append(f"| entities | {n_entities:,} |")
    L.append(f"| sources | {len(res.per_source)} |")
    L.append(f"| cross-source links | {getattr(res.entity_map, 'n_cross_source', 0)} |")
    blocked = sum(1 for v in res.sufficiency.values() if v.status == "BLOCKED")
    L.append(f"| blocked models | {blocked} |")
    L.append("")

    L.append("## Source files")
    L.append("")
    L.append("| file | source type | rows | entities | resolved slots | warnings |")
    L.append("|---|---|---|---|---|---|")
    for rs in res.per_source:
        warns = "; ".join(rs.warnings)
        slots = ", ".join(sorted(rs.resolved_fields))
        L.append(f"| {Path(rs.file)} | {rs.source} | {rs.n_rows:,} | {rs.n_entities:,} | {slots} | {warns} |")
    L.append("")

    L.append("## Models auto-activated")
    L.append("")
    L.append("| model | verdict | reason |")
    L.append("|---|---|---|")
    for v in res.sufficiency.values():
        L.append(f"| {v.model} | {v.status} | {v.reason} |")
    L.append("")

    L.append("## Ranked insights")
    L.append("")
    L.append("| rank | entity | score | n_events | models fired |")
    L.append("|---|---|---|---|---|")
    for i, r in enumerate(res.rankings[:cfg.top_k], 1):
        L.append(f"| {i} | {r['entity_id']} | {r['score']:.3f} | {r['n_events']} | "
                 f"{', '.join(r['models_fired'])} |")
    L.append("")

    L.append("## Explanations")
    L.append("")
    if res.explanations:
        for eid, text in list(res.explanations.items())[:cfg.top_k]:
            L.append(f"- **{eid}**: {text}")
    else:
        L.append("None.")
    L.append("")
    return "\n".join(L)


def _json(res, cfg) -> dict:
    unified = res.unified
    col = "entity_id"
    n_entities = int(unified[col].nunique()) if not unified.empty else 0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "events": int(len(unified)),
            "entities": n_entities,
            "sources": len(res.per_source),
            "cross_source_links": int(getattr(res.entity_map, "n_cross_source", 0)),
            "blocked_models": sum(1 for v in res.sufficiency.values() if v.status == "BLOCKED"),
        },
        "sources": [
            {
                "file": getattr(rs, "file", ""),
                "source": rs.source,
                "rows": rs.n_rows,
                "entities": rs.n_entities,
                "resolved_slots": sorted(rs.resolved_fields),
                "warnings": rs.warnings,
            }
            for rs in res.per_source
        ],
        "model_verdicts": _statuses(res),
        "rankings": res.rankings[:cfg.top_k],
        "explanations": dict(list(res.explanations.items())[:cfg.top_k]),
    }