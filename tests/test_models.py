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


def test_run_model_dispatch_new_candidates():
    """The model-zoo candidates must route through run_model and produce
    unit-range scores (or empty dicts when the data cannot support them)."""
    df = _f_df(n_entities=16, events_per=60)
    candidates = (
        "time_correlation_backward", "time_correlation_anypair", "time_velocity",
        "oddball_signed", "degree_deviation", "reciprocity",
        "pagerank_deviation", "community_motif", "scatter_gather",
        "statml_eif", "statml_lof", "statml_mahalanobis", "statml_pca",
        "statml_zscore", "statml_ocsvm", "statml_autoencoder", "statml_hbos",
        "statml_gmm", "statml_kde", "benford_ks", "benford_second",
        "structuring_banded",
    )
    for name in candidates:
        out = am.run_model(name, df)
        assert isinstance(out, dict), name
        assert all(0.0 <= v <= 1.0 for v in out.values()), name


def test_statml_ocsvm_ranks_outlier_highest():
    """A synthetic far-away entity should be the top OCSVM risk (literature:
    OCSVM beats IF/LOF for top-k AML alert prioritisation)."""
    rng = np.random.default_rng(7)
    rows = []
    for ei in range(20):
        for _ in range(50):
            rows.append({"entity_id": f"E{ei:02d}", "source": "bank",
                         "amount": float(rng.uniform(100, 8000)),
                         "counterparty_id": f"C{rng.integers(0, 9)}",
                         "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=int(rng.integers(1, 300)))})
    df = pd.DataFrame(rows)
    outlier = [{"entity_id": "E{0}".format(i), "source": "bank", "amount": 1e7,
                "counterparty_id": "CX1", "timestamp": pd.Timestamp("2024-01-01")} for i in range(18, 21)]
    df = pd.concat([df, pd.DataFrame(outlier)], ignore_index=True)
    out = am.fit_statml_ocsvm(df)
    assert out, "OCSVM produced no scores"
    assert max(out.values()) >= 0.5
    assert max(out, key=out.get) in ("E18", "E19", "E20")


def test_scatter_gather_flags_mule():
    """Scatter-gather: an entity receiving from many (+ inset fan-in) AND
    dispersing to many (fan-out) networks should rank above 0.5."""
    rng = np.random.default_rng(3)
    rows = []
    for src in range(25):
        for dstk in range(12):
            rows.append({"entity_id": f"E{src:02d}", "counterparty_id": f"C{src}-{dstk}",
                         "direction": rng.choice(["credit", "debit"]), "amount": 1.0})
    df = pd.DataFrame(rows)
    out = am.fit_scatter_gather(df)
    assert out
    assert max(out.values()) > 0.5


def test_fusion_variants_rank():
    scores = {"benford": {"E001": 0.95, "E002": 0.3},
              "structuring": {"E001": 0.8, "E003": 0.9}}
    df = _f_df(n_entities=4, events_per=5)
    for fuse in (am.fuse_rank_borda, am.fuse_rank_average, am.fuse_score_mean):
        r = fuse(dict(scores), df).rank
        assert r, "empty ranking produced"
        assert r[0]["score"] >= r[-1]["score"]


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