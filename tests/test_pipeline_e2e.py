import pandas as pd

from aml.pipeline import Pipeline
from aml.report import generate_report


def _run(paths, **kw):
    return Pipeline(min_events_per_entity=8, **kw).run(paths, with_models=True)


def test_pipeline_over_mixed_formats(synthetic_dir):
    paths = [str(p) for p in sorted(synthetic_dir.glob("sources/*"))]
    res = _run(paths)
    sources = {rs.source for rs in res.per_source}
    assert {"bank", "cdr"} <= sources
    assert res.unified["entity_id"].notna().any()
    assert res.rankings
    assert res.explanations


def test_pipeline_each_model_autoactivated(synthetic_dir):
    paths = [str(p) for p in sorted(synthetic_dir.glob("sources/*"))]
    res = _run(paths)
    ran = {m for m in res.model_scores if m != "time_correlation"} | (
        {"time_correlation"} if res.model_scores.get("time_correlation") else set())
    assert ran, "expected at least one model to fire"
    assert res.sufficiency["benford"].status == "SUPPORTED"


def test_pipeline_bank_only_degrades_network():
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=300, freq="h"),
        "amt": [float(i % 9000 + 1) for i in range(300)],
        "beneficiary": [f"c{i % 10}" for i in range(300)],
        "acct": ["A1"] * 300,
        "dr_cr": ["debit"] * 300,
    })
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bank.csv"
        df.to_csv(p, index=False)
        res = _run([str(p)])
    assert res.sufficiency["network"].status == "DEGRADED"
    assert res.sufficiency["benford"].status == "SUPPORTED"
    # single-source: the merge_asof cross-source model cannot run, so the
    # engine says DEGRADED rather than claiming SUPPORTED with no scores
    assert res.sufficiency["time_correlation"].status == "DEGRADED"
    assert res.model_scores.get("time_correlation") is None


def test_report_generation(synthetic_dir, tmp_path):
    paths = [str(p) for p in sorted(synthetic_dir.glob("sources/*"))]
    res = _run(paths)
    from aml.config import PipelineConfig
    cfg = PipelineConfig()
    written = generate_report(res, cfg, tmp_path)
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Cross-Source Anomaly Intelligence Brief" in md
    assert "analysis depth auto-adapted to the data" in md
    assert "The people who stand out" in md
    assert "Where the records came from" in md
    assert "When it happened" in md
    # the human report must not expose jargon/scores
    assert "model" not in md.lower()
    # charts must be embedded in the markdown itself (self-contained export)
    assert "data:image/png;base64," in md
    assert md.count("data:image/png;base64,") >= 2
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.pdf").exists()
    assert (tmp_path / "report.html").read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert list(written) == ["markdown", "html", "pdf", "json"]
    charts = list((tmp_path / "charts").glob("*.png"))
    names = {p.name for p in charts}
    assert len(charts) >= 3, f"expected at least 3 charts, got {len(charts)}"
    assert {"network.png", "map.png", "people.png", "sources.png",
            "timeline.png"} <= names
    assert any(n.startswith("entity_") for n in names)
    assert "Who the flagged people are connected to" in md
    assert "The activity map" in md
    from pypdf import PdfReader
    pdf = PdfReader(str(tmp_path / "report.pdf"))
    assert len(pdf.pages) >= 1


def test_report_generation_empty_input(tmp_path):
    """Report renders on empty runs (no sources) without crashing."""
    from aml.config import PipelineConfig
    res = Pipeline().run([], with_models=True)
    written = generate_report(res, PipelineConfig(), tmp_path)
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert ("Nobody stood out" in md
            or "No timestamped records" in md
            or "Where the records came from" in md)
    assert (tmp_path / "report.pdf").exists()
    from pypdf import PdfReader
    assert len(PdfReader(str(tmp_path / "report.pdf")).pages) >= 1


def test_pipeline_empty_paths_does_not_crash():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        res = Pipeline().run([], with_models=True)
    assert res.rankings == []
    assert res.sufficiency["entity_resolution"].status == "BLOCKED"
