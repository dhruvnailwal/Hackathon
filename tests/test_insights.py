"""Tests for the plain-language insights layer."""
from aml.insights import build_insights, display_name
from aml.pipeline import Pipeline


def _run(paths):
    return Pipeline(min_events_per_entity=8).run(paths, with_models=True)


def test_insights_are_plain_and_named(synthetic_dir):
    paths = [str(p) for p in sorted(synthetic_dir.glob("sources/*"))]
    res = _run(paths)
    ins = build_insights(res, top_k=5)
    assert ins
    for it in ins:
        assert it["name"], "display name must resolve from the entity map"
        assert it["name"].startswith("E") is False, "must show a real name, not an id"
        assert isinstance(it["headline"], str) and len(it["headline"]) > 10
        assert it["evidence"], "each flagged person needs at least one concrete reason"
        assert "model" not in it["headline"].lower()
        assert "score" not in it["headline"].lower()
        assert it["risk"] in {"HIGH", "MEDIUM", "LOW"}


def test_display_name_falls_back_to_id(synthetic_dir):
    from aml.pipeline import PipelineResult
    import pandas as pd
    res = PipelineResult.__new__(PipelineResult)
    import types
    ent = types.SimpleNamespace(entity_names={})
    res.entity_map = ent
    res.unified = pd.DataFrame({"entity_id": ["E0001"], "actor_name": [None]})
    assert display_name(res, "E0001") == "E0001"


def test_insights_empty_rankings():
    import pandas as pd
    from aml.pipeline import PipelineResult
    import types
    res = PipelineResult.__new__(PipelineResult)
    res.entity_map = types.SimpleNamespace(entity_names={})
    res.unified = pd.DataFrame(columns=["entity_id", "actor_name", "timestamp",
                                        "counterparty_id", "amount", "source", "direction"])
    res.rankings = []
    assert build_insights(res) == []