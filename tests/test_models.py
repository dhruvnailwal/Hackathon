import numpy as np
import pandas as pd

from aml import models as am


def _f_df(n_entities=12, events_per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    ts = pd.Timestamp("2024-01-01")
    for ei in range(n_entities):
        for _ in range(events_per):
            rows.append({
                "entity_id": f"E{ei:03d}",
                "counterparty_id": f"C{rng.integers(0, 30)}",
                "source": "bank",
                "event_type": "transaction",
                "amount": float(rng.uniform(1, 20000)),
                "width": ts + pd.Timedelta(minutes=int(rng.integers(1, 200))),
                "direction": rng.choice(["credit", "debit"]),
            })
            ts = ts + pd.Timedelta(minutes=int(rng.integers(1, 200)))
    df = pd.DataFrame(rows)
    df["timestamp"] = df["width"]
    return df.drop(columns=["width"])


def test_build_features_counts_events():
    df = _f_df()
    feats = am.build_features(df)
    assert int(feats.loc["E000", "n_events"]) == 40
    assert {"n_events", "fan_out", "fan_in"} <= set(feats.columns)


def test_benford_returns_unit_range_scores():
    df = _f_df()
    out = am.fit_benford(df)
    assert isinstance(out, dict)
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_structuring_flags_band_entity():
    rng = np.random.default_rng(1)
    rows = []
    ts = pd.Timestamp("2024-01-01")
    for tag, count, band_amt in (("B", 20, False), ("S", 9, True)):
        for _ in range(count):
            amt = float(rng.uniform(8800, 9999)) if band_amt else float(rng.uniform(100, 9000))
            rows.append({"entity_id": tag, "source": "bank", "amount": amt,
                         "timestamp": ts})
            ts += pd.Timedelta(minutes=10)
    df = pd.DataFrame(rows)
    out = am.fit_structuring(df, threshold=10000.0, band=0.88)
    assert out.get("S", 0.0) > 0.5


def test_run_model_dispatch_returns_dicts():
    df = _f_df()
    for name in ("network", "statml", "benford", "structuring", "behavioral"):
        out = am.run_model(name, df)
        assert isinstance(out, dict)


def test_fusion_ranks_and_explains():
    scores = {"benford": {"E001": 0.95, "E002": 0.3},
              "structuring": {"E001": 0.8, "E003": 0.9}}
    df = _f_df(n_entities=4, events_per=5)
    fusion = am.fuse_model_scores(scores, df)
    assert fusion.rank[0]["entity_id"] == "E001"
    assert fusion.rank[0]["score"] >= fusion.rank[1]["score"]
    assert fusion.rank[0]["score"] >= fusion.rank[2]["score"]
    assert "E001" in fusion.explanation
    assert fusion.rank[0]["models_fired"]  # attributes present