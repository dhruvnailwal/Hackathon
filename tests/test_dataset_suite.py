"""Tests for the dataset suite: cross-source correlation variants.

Covers generation (structure matches the requested correlation mode) and
that the full pipeline + eval harness work on every mode.
"""

# aml first: it eagerly imports torch, which is only stable before the
# heavy pandas/sklearn stack has loaded on Windows.
from aml.data_generator.emit import (CORRELATION_MODES, PAIR_SOURCES,
                                     build_scenario, name_column)
from aml.loaders import collect_files
from aml.pipeline import Pipeline

import json
import pandas as pd
import pytest

MODE_SIZES = {"n_background": 25, "n_per_storyline": 1, "n_surprise": 3}


@pytest.fixture(scope="session")
def suite_dir(tmp_path_factory):
    """One small dataset per correlation mode, plus the generation manifest."""
    from aml.data_generator.emit_multi import generate_all_mixed

    out = tmp_path_factory.mktemp("datasets")
    manifest = {"datasets": []}
    for mode in CORRELATION_MODES:
        for k in range(2):
            seed = 500 + k + CORRELATION_MODES.index(mode)
            dataset = out / mode / f"set_{k + 1}"
            generate_all_mixed(dataset, seed=seed, surprise_seed=seed + 9,
                               correlation=mode, **MODE_SIZES)
            manifest["datasets"].append(
                {"mode": mode, "name": f"{mode}/set_{k + 1}"})
    (out / "manifest.json").write_text(str(manifest))
    return out


def _name_sets(dataset) -> dict:
    sets = {}
    for src in ("bank", "cdr", "social"):
        df = pd.read_csv(dataset / "sources" / f"{src}_export.csv",
                         low_memory=False)
        col = name_column(df, src)
        sets[src] = set(df[col].astype(str).dropna().unique())
    return sets


def _overlap(a, b):
    return len(a & b) / max(len(a | b), 1)


# ---------------------------------------------------------------------------
# generation: correlation structure must match the requested mode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", CORRELATION_MODES)
def test_person_sources_match_mode(mode):
    sc = build_scenario(seed=1, surprise_seed=2, correlation=mode, **MODE_SIZES)
    sources = sc["person_sources"]
    assert sources
    for pid, owned in sources.items():
        assert 1 <= len(owned) <= 3
        if mode == "full":
            assert owned == {"bank", "cdr", "social"}
        elif mode == "none":
            assert len(owned) == 1
        else:
            pair = PAIR_SOURCES[mode]
            excluded = {"bank", "cdr", "social"} - pair
            assert owned == pair or owned == excluded
        # every anomaly kept at least one event in an owned source
        for p in sc["people"].values():
            owned = sc["person_sources"][p.person_id]
            assert any(p._events.get(s) for s in owned)


@pytest.mark.parametrize("mode", CORRELATION_MODES)
def test_emitted_files_have_expected_cross_source_overlap(suite_dir, mode):
    sets = _name_sets(suite_dir / mode / "set_1")
    if mode == "full":
        # everyone participates in all three sources; overlap may dip slightly
        # because normal event generation is probabilistic per source
        assert _overlap(sets["bank"], sets["cdr"]) > 0.7
        assert _overlap(sets["bank"], sets["social"]) > 0.7
        assert _overlap(sets["cdr"], sets["social"]) > 0.7
    elif mode == "none":
        assert _overlap(sets["bank"], sets["cdr"]) < 0.15
        assert _overlap(sets["bank"], sets["social"]) < 0.15
        assert _overlap(sets["cdr"], sets["social"]) < 0.15
    else:
        pair = PAIR_SOURCES[mode]
        others = {"bank", "cdr", "social"} - pair
        a, b = sorted(pair)
        assert _overlap(sets[a], sets[b]) > 0.7
        for o in others:
            assert _overlap(sets[a], sets[o]) < 0.15
            assert _overlap(sets[b], sets[o]) < 0.15


def test_every_dataset_has_answer_keys(suite_dir):
    for mode in CORRELATION_MODES:
        for k in (1, 2):
            dataset = suite_dir / mode / f"set_{k}"
            assert (dataset / "answer_key.json").exists()
            assert (dataset / "answer_key_surprise.json").exists()
            assert any((dataset / "sources").glob("*_export.csv"))


# ---------------------------------------------------------------------------
# end-to-end: the pipeline + eval harness must run on every mode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", CORRELATION_MODES)
def test_pipeline_runs_on_every_mode(suite_dir, mode):
    paths = collect_files([str(suite_dir / mode / "set_1" / "sources")])
    res = Pipeline(min_events_per_entity=8).run(paths, with_models=True)
    assert res.rankings
    assert len(res.unified) > 0
    assert all(0.0 <= r["score"] <= 1.0 for r in res.rankings)


@pytest.mark.parametrize("mode", CORRELATION_MODES)
def test_eval_scores_computable_on_every_mode(suite_dir, mode):
    from eval import load_truth, map_truth_to_entities, recall_precision, to_records

    dataset = suite_dir / mode / "set_2"
    paths = collect_files([str(dataset / "sources")])
    res = Pipeline().run(paths, with_models=True)
    truth = map_truth_to_entities(
        load_truth(str(dataset / "answer_key.json")), res.entity_map)
    truth_anom, truth_type = to_records(res.rankings, truth)
    assert truth_anom, "each mode must contain findable anomalies"
    m = recall_precision(res.rankings, truth_anom, truth_type)
    assert 0.0 <= m["recall@10"] <= 1.0
    assert m["recall@10"] > 0.0


def test_generate_script_writes_manifest(tmp_path, monkeypatch):
    import sys as _sys

    import scripts.generate_dataset_suite as gen

    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(_sys, "argv",
                        ["generate_dataset_suite", "--n-seeds", "1",
                         "--n-background", "10", "--n-per-storyline", "1",
                         "--n-surprise", "2"])
    assert gen.main() == 0
    manifest = json.loads((tmp_path / "data" / "datasets" / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert {d["mode"] for d in manifest["datasets"]} == set(CORRELATION_MODES)
    assert (tmp_path / "data" / "datasets" / "suite.md").exists()
