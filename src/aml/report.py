"""Report generation (markdown + HTML + PDF + JSON) for a PipelineResult.

What a human reads from a report is *plain-language findings* — the top
findings are written as sentences about real people and real records, never
as model names or scores. Machine-readable detail (per-record scores, model
verdicts) lives in report.json only.

  * report.md   — plain markdown with embedded charts
  * report.html — self-contained styled HTML (charts inlined as base64)
  * report.pdf  — printable PDF built with reportlab
  * report.json — machine-readable payload
  * charts/     — the matplotlib PNGs referenced by all formats
"""
from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from . import visuals
from . import insights as insights_mod

MINT = "#3fbf8f"
SKY = "#5aa7e8"
TEAL = "#35c7c2"
AMBER = "#e0a03d"
RED = "#e0566b"
NAVY = "#1e3b33"
MUTED = "#6b857d"
LIGHT = "#eef5f1"

RISK_COLORS = {
    "HIGH": RED,
    "MEDIUM": AMBER,
    "LOW": MINT,
}

# Judge-facing report branding (shared by markdown / HTML / PDF)
REPORT_TITLE = "Cross-Source Anomaly Intelligence Brief"
REPORT_SUBTITLE = ("Automatic screening of financial, telecom and social "
                   "activity — every flagged person is backed by concrete, "
                   "explainable evidence.")


def _generated_line(res) -> str:
    """Timestamp + scope line under the report title."""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    n_files = len(res.per_source)
    n_src = len({rs.source for rs in res.per_source})
    return (f"Generated {ts} · {n_files} source file(s) across "
            f"{n_src} data type(s) · analysis depth auto-adapted to the data")


def _fmt(v: float) -> str:
    return f"{v:,.0f}"


def _unbold(s: str) -> str:
    """Strip **emphasis** markers (used by the insights engine)."""
    return s.replace("**", "")


def _htmlize(s: str) -> str:
    """Escape HTML + turn **emphasis** into <b> tags."""
    parts = re.split(r"\*\*(.+?)\*\*", s)
    out = []
    for i, part in enumerate(parts):
        cleaned = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(f"<b>{cleaned}</b>" if i % 2 else cleaned)
    return "".join(out)


def _pdfize(s: str) -> str:
    """Escape for reportlab Paragraph, with **emphasis** -> <b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>",
                  s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# shared content
# ---------------------------------------------------------------------------
def _insights_for(res, cfg) -> List[dict]:
    return insights_mod.build_insights(res, top_k=cfg.top_k)


def _overview_lines(res, insight_list) -> List[str]:
    """Plain opening sentences: what was reviewed, what stood out."""
    unified = res.unified
    n_events = int(len(unified))
    n_people = int(unified["entity_id"].nunique()) if not unified.empty else 0
    n_files = len(res.per_source)
    lines = [
        f"We looked through **{_fmt(n_events)} records** across "
        f"**{n_files} file(s)** and mapped them to **{_fmt(n_people)} people**.",
    ]
    if insight_list:
        lines.append(f"**{len(insight_list)} people** stood out enough to be worth "
                     f"a closer look — they are listed below, most important first.")
    else:
        lines.append("**Nobody stood out** from the crowd in the records we reviewed.")
    return lines


def _has_timeline(res) -> bool:
    return (res.unified is not None and not res.unified.empty
            and "timestamp" in res.unified.columns
            and res.unified["timestamp"].notna().any())


# ---------------------------------------------------------------------------
# markdown (self-contained: charts inlined as base64 PNG data URIs so the
# .md file renders everywhere — email, GitHub, other machines — without the
# charts/ folder travelling with it)
# ---------------------------------------------------------------------------
def _md_chart(chart: Path) -> str:
    if not Path(chart).exists():
        return ""
    b64 = base64.b64encode(Path(chart).read_bytes()).decode("ascii")
    return f"![{Path(chart).stem}](data:image/png;base64,{b64})"


def _markdown(res, cfg, charts: Dict[str, Path]) -> str:
    insight = _insights_for(res, cfg)
    L: List[str] = []
    L.append(f"# {REPORT_TITLE}")
    L.append(f"*{_generated_line(res)}*")
    L.append("")
    L.append(REPORT_SUBTITLE)
    L.append("")
    for line in _overview_lines(res, insight):
        L.append(line)
    L.append("")
    L.append(_md_chart(charts["people"]))
    L.append("")

    L.append("## The people who stand out")
    L.append("")
    if insight:
        for it in insight:
            L.append(f"### {it['rank']}. {it['name']} — {it['risk']}")
            L.append("")
            for j, bullet in enumerate(it["evidence"]):
                prefix = "-" if j == 0 else "  -"
                L.append(f"{prefix} {_unbold(bullet)}")
            L.append("")
    else:
        L.append("None.")
        L.append("")

    L.append("## Where the records came from")
    L.append("")
    L.append("| file | source | records | people |")
    L.append("|---|---|---|---|")
    if res.per_source:
        for rs in res.per_source:
            L.append(f"| {Path(rs.file).name} | {rs.source} | {rs.n_rows:,} | "
                     f"{rs.n_entities:,} |")
    else:
        L.append("| — | — | 0 | 0 |")
    L.append("")
    if res.per_source:
        L.append(_md_chart(charts["sources"]))
        L.append("")

    L.append("## When it happened")
    L.append("")
    if _has_timeline(res):
        L.append(_md_chart(charts["timeline"]))
    else:
        L.append("No timestamped records were found in the files.")
    L.append("")

    if charts.get("network") and charts.get("people") is not None:
        L.append("## Who the flagged people are connected to")
        L.append("")
        L.append(_md_chart(charts["network"]))
        L.append("")

    if charts.get("map"):
        L.append("## The activity map")
        L.append("")
        L.append(_md_chart(charts["map"]))
        L.append("")

    if charts.get("drill"):
        L.append("## A closer look at the people who stand out")
        L.append("")
        for title, img in charts["drill"].items():
            L.append(f"### {title}")
            L.append("")
            L.append(_md_chart(img))
            L.append("")

    L.append("_The full technical detail behind this analysis is kept in "
             "`report.json` — this report is the human summary._")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# HTML (self-contained, charts inlined)
# ---------------------------------------------------------------------------
def _b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _risk_chip(risk: str) -> str:
    color = RISK_COLORS.get(risk, MUTED)
    return (f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:9px;font-weight:800;font-size:11px">{risk}</span>')


def _html_insight_cards(insight_list) -> str:
    if not insight_list:
        return ('<p style="color:#6b857d">Nobody stood out from the crowd '
                'this run.</p>')
    cards = []
    for it in insight_list:
        bullets = "\n".join(f"<li>{_htmlize(b)}</li>" for b in it["evidence"])
        cards.append(
            f'<div class="card person">'
            f'<div class="head"><span class="rank">{it["rank"]}</span>'
            f'<h3>{it["name"]}</h3>{_risk_chip(it["risk"])}</div>'
            f'<ul>{bullets}</ul></div>'
        )
    return "\n".join(cards)


def _html_stats(res, insight_list) -> str:
    unified = res.unified
    n_events = f"{int(len(unified)):,}"
    n_people = f"{unified['entity_id'].nunique():,}" if not unified.empty else "0"
    n_files = len(res.per_source)
    items = [
        (n_events, "records reviewed"),
        (n_people, "people found"),
        (n_files, "source files"),
        (str(len(insight_list)), "people who stand out"),
    ]
    return "\n".join(f'<div class="stat"><b>{v}</b><span>{lbl}</span></div>'
                     for v, lbl in items)


def _html(res, cfg, charts: Dict[str, Path]) -> str:
    insight_list = _insights_for(res, cfg)
    overview = _overview_lines(res, insight_list)
    overview_html = "<br>".join(_htmlize(line) for line in overview)
    stat_chips = _html_stats(res, insight_list)
    insight_cards = _html_insight_cards(insight_list)

    src_rows = "\n".join(
        f"<tr><td>{Path(rs.file).name}</td><td>{rs.source}</td>"
        f"<td>{rs.n_rows:,}</td><td>{rs.n_entities:,}</td></tr>"
        for rs in res.per_source
    ) or '<tr><td colspan="4">no source files loaded</td></tr>'

    timeline_img = _b64(charts["timeline"]) if _has_timeline(res) else ""
    sources_img = _b64(charts["sources"]) if res.per_source else ""
    sources_block = (f'<img src="{sources_img}" style="margin-top:14px">'
                     if sources_img else "")
    timeline_block = (f'<img src="{timeline_img}" style="margin-top:14px">'
                      if timeline_img else
                      '<p style="color:#6b857d">No timestamped records were found '
                      'in the files.</p>')

    network_block = (f'<img src="{_b64(charts["network"])}" style="margin-top:14px">'
                     if charts.get("network") else "")
    map_block = (f'<img src="{_b64(charts["map"])}" style="margin-top:14px">'
                 if charts.get("map") else "")
    drill_imgs = "\n".join(
        f'<p style="margin:14px 0 4px;font-weight:700">{_htmlize(title)}</p>'
        f'<img src="{_b64(img)}">' for title, img in (charts.get("drill") or {}).items())

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Anomaly Lens report</title>
<style>
  body {{ font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color: {NAVY};
         background: linear-gradient(180deg, #e9f8f1 0%, #e6f1f9 55%, #dcedf7 100%);
         margin: 0; padding: 24px; }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 28px; font-weight: 800; margin: 0 0 4px; }}
  .meta {{ color: {MUTED}; font-size: 13px; margin-bottom: 18px; }}
  .card {{ background: rgba(255,255,255,0.75); border: 1px solid rgba(255,255,255,0.9);
           border-radius: 18px; padding: 18px 22px; margin: 16px 0;
           box-shadow: 0 6px 18px rgba(60,120,105,0.12); }}
  .person {{ border-left: 4px solid {TEAL}; }}
  .head {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .head h3 {{ margin: 0; color: {NAVY}; }}
  h2 {{ font-size: 18px; font-weight: 700; color: {NAVY}; margin: 0 0 10px; }}
  .stats {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 14px; }}
  .stat {{ flex: 1; min-width: 120px; background: rgba(255,255,255,0.8);
           border-radius: 14px; padding: 12px 16px; }}
  .stat b {{ font-size: 24px; display: block; color: {TEAL}; }}
  .stat span {{ font-size: 11px; color: {MUTED}; font-weight: 600; }}
  .rank {{ background: {MINT}; color: #fff; width: 26px; height: 26px;
          display: inline-flex; align-items: center; justify-content: center;
          border-radius: 50%; font-weight: 800; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: {MUTED}; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.4px; padding: 8px 10px; border-bottom: 2px solid {MINT}; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid rgba(107,133,125,0.18); }}
  tr:nth-child(even) td {{ background: rgba(63,191,143,0.05); }}
  img {{ max-width: 100%; border-radius: 12px; }}
  code {{ background: rgba(63,191,143,0.12); padding: 1px 6px; border-radius: 6px; }}
  ul {{ margin: 12px 0 0; padding-left: 20px; }}
  li {{ margin-top: 5px; }}
  .footnote {{ color: {MUTED}; font-size: 12px; margin-top: 12px; }}
</style></head>
<body><div class="wrap">
  <h1>{REPORT_TITLE}</h1>
  <div class="meta">{REPORT_SUBTITLE}<br>{_generated_line(res)}</div>
  <div class="card">{overview_html}
    <div class="stats">{stat_chips}</div></div>
  <div class="card"><h2>The people who stand out</h2>{insight_cards}</div>
  <div class="card"><h2>When it happened</h2>{timeline_block}</div>
  <div class="card"><h2>Who the flagged people are connected to</h2>{network_block}</div>
  <div class="card"><h2>The activity map</h2>{map_block}</div>
  <div class="card"><h2>A closer look at the people who stand out</h2>{drill_imgs}</div>
  <div class="card"><h2>Where the records came from</h2>
    <table><thead><tr><th>file</th><th>source</th><th>records</th><th>people</th>
    </tr></thead><tbody>{src_rows}</tbody></table>{sources_block}</div>
  <div class="footnote">The full technical detail behind this analysis is kept
  in <code>report.json</code>. This page is the human-readable summary.</div>
</div></body></html>"""


# ---------------------------------------------------------------------------
# JSON (machine-readable; keeps the technical scores the report omits)
# ---------------------------------------------------------------------------
def _json(res, cfg) -> dict:
    unified = res.unified
    n_entities = int(unified["entity_id"].nunique()) if not unified.empty else 0
    scores = res.model_scores or {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "records": int(len(unified)),
            "people": n_entities,
            "source_files": len(res.per_source),
            "people_who_stand_out": len(res.rankings),
        },
        "sources": [
            {
                "file": getattr(rs, "file", ""),
                "source": rs.source,
                "rows": rs.n_rows,
                "people": rs.n_entities,
                "resolved_slots": sorted(rs.resolved_fields),
                "warnings": rs.warnings,
            }
            for rs in res.per_source
        ],
        "model_scores": {m: dict(sorted(s.items(), key=lambda kv: -kv[1])[:20])
                         for m, s in scores.items()},
        "rankings": res.rankings,
    }


# ---------------------------------------------------------------------------
# PDF (reportlab)
# ---------------------------------------------------------------------------
def _pdf(res, cfg, report_dir: Path, charts: Dict[str, Path]) -> Path:
    """Build a printable PDF led by plain-language findings."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    path = report_dir / "report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title="Anomaly Lens report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], textColor=colors.HexColor(NAVY),
                        fontSize=20, spaceAfter=2)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor(NAVY),
                        fontSize=13, spaceBefore=10, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10.5,
                        textColor=colors.HexColor(NAVY), spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9,
                          textColor=colors.HexColor(NAVY))
    meta = ParagraphStyle("Meta", parent=styles["BodyText"], fontSize=8,
                          textColor=colors.HexColor(MUTED))
    risk_colors = {"HIGH": colors.HexColor(RED), "MEDIUM": colors.HexColor(AMBER),
                   "LOW": colors.HexColor(MINT)}

    insight_list = _insights_for(res, cfg)
    story = [
        Paragraph(REPORT_TITLE, h1),
        Paragraph(f"{_generated_line(res)} · {REPORT_SUBTITLE}", meta),
        Spacer(1, 4 * mm),
    ]
    for line in _overview_lines(res, insight_list):
        story.append(Paragraph(_pdfize(line), body))
    if charts.get("people"):
        story += [Spacer(1, 2 * mm),
                  Image(str(charts["people"]), width=170 * mm, height=64 * mm)]
    story.append(Spacer(1, 3 * mm))

    if insight_list:
        story.append(Paragraph("The people who stand out", h2))
        for it in insight_list:
            rc = risk_colors.get(it["risk"], colors.HexColor(MUTED))
            story.append(Paragraph(f"{it['rank']}. <b>{it['name']}</b> — "
                                   f"<font color='#{rc.hexval()[2:]}'>{it['risk']}</font>", h3))
            for bl in it["evidence"]:
                story.append(Paragraph(f"• {_pdfize(bl)}", body))
        story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Where the records came from", h2))
    story.append(Table(
        [["file", "source", "records", "people"]] +
        [[Path(rs.file).name, rs.source, rs.n_rows, rs.n_entities]
         for rs in res.per_source] or [["—", "—", 0, 0]],
        colWidths=[78 * mm, 24 * mm, 30 * mm, 30 * mm],
        style=TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cfdcd6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor(LIGHT)]),
        ]),
    ))
    if charts.get("sources"):
        story += [Spacer(1, 2 * mm),
                  Image(str(charts["sources"]), width=170 * mm, height=48 * mm)]
    story.append(Spacer(1, 3 * mm))

    if charts.get("timeline") and _has_timeline(res):
        story.append(Paragraph("When it happened", h2))
        story.append(Image(str(charts["timeline"]), width=170 * mm, height=54 * mm))
        story.append(Spacer(1, 3 * mm))

    if charts.get("network"):
        story.append(Paragraph("Who the flagged people are connected to", h2))
        story.append(Image(str(charts["network"]), width=170 * mm, height=90 * mm))
        story.append(Spacer(1, 3 * mm))

    if charts.get("map"):
        story.append(Paragraph("The activity map", h2))
        story.append(Image(str(charts["map"]), width=170 * mm, height=92 * mm))
        story.append(Spacer(1, 3 * mm))

    if charts.get("drill"):
        story.append(Paragraph("A closer look at the people who stand out", h2))
        for title, img in charts["drill"].items():
            story.append(Paragraph(f"<b>{_pdfize(title)}</b>", h3))
            story.append(Image(str(img), width=170 * mm, height=60 * mm))
            story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("The full technical detail behind this analysis is "
                           "kept with this report for the record — this page "
                           "is the human-readable summary.", meta))
    doc.build(story)
    return path


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def generate_report(result, config, report_dir: str | Path = "data/report") -> dict:
    """Build and write the full report. Returns paths of written files."""
    report_dir = Path(report_dir)
    charts_dir = report_dir / "charts"
    report_dir.mkdir(parents=True, exist_ok=True)

    charts = {
        "people": visuals.render_person_attention(result, charts_dir),
        "sources": visuals.render_sources(result, charts_dir),
        "timeline": visuals.render_timeline(result, charts_dir),
        "network": visuals.render_ego_graph(result, charts_dir),
        "map": visuals.render_location_map(result, charts_dir),
    }
    top = _insights_for(result, config) or []
    charts["drill"] = {
        f"{it['name']} ({it['risk']})": visuals.render_entity_timeline(
            result, charts_dir, it["entity_id"])
        for it in top[:3]
    }

    markdown = _markdown(result, config, charts)
    html = _html(result, config, charts)
    json_body = _json(result, config)
    pdf_path = _pdf(result, config, report_dir, charts)

    (report_dir / "report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "report.html").write_text(html, encoding="utf-8")
    (report_dir / "report.json").write_text(
        json.dumps(json_body, indent=2, default=str), encoding="utf-8"
    )
    return {
        "markdown": str(report_dir / "report.md"),
        "html": str(report_dir / "report.html"),
        "pdf": str(pdf_path),
        "json": str(report_dir / "report.json"),
    }