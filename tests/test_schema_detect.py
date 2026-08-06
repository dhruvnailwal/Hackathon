import numpy as np
import pandas as pd

from aml.schema_detect import (
    classify_source,
    detect_columns,
    detect_dataframe,
    load_and_detect,
)

RNG = np.random.default_rng(0)


def _bank_df(n=40):
    return pd.DataFrame({
        "txn_date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "amount": RNG.uniform(1, 9000, n).round(2),
        "beneficiary": [f"acct{int(i)}" for i in RNG.integers(0, 50, n)],
        "account_id": ["ACC1001"] * n,
        "debit/credit": RNG.choice(["credit", "debit"], n),
        "branch_code": RNG.integers(1, 30, n),
    })


def _cdr_df(n=40):
    return pd.DataFrame({
        "call_time": pd.date_range("2024-01-01", periods=n, freq="min"),
        "caller_msisdn": ["911234567890"] * n,
        "callee": [f"9{int(i)}12345" for i in RNG.integers(0, 100, n)],
        "duration_seconds": RNG.integers(1, 600, n),
        "tower_id": [f"T{int(i)}" for i in RNG.integers(0, 20, n)],
    })


def _social_df(n=40):
    return pd.DataFrame({
        "created_utc": pd.date_range("2024-01-01", periods=n, freq="30min"),
        "user_handle": [f"u{int(i)}" for i in RNG.integers(0, 30, n)],
        "mentions": [f"u{int(i)}" for i in RNG.integers(0, 30, n)],
        "geo_tag": [f"loc{int(i)}" for i in RNG.integers(0, 10, n)],
    })


def test_bank_classification_and_slots():
    det = detect_dataframe(_bank_df(), file="x.csv")
    assert det.source_type == "bank"
    assert {"timestamp", "amount", "counterparty"} <= det.resolved_slots


def test_cdr_classification_and_duration():
    det = detect_dataframe(_cdr_df(), file="x.csv")
    assert det.source_type == "cdr"
    assert "duration" in det.resolved_slots
    assert "amount" not in det.resolved_slots


def test_social_classification():
    det = detect_dataframe(_social_df(), file="x.csv")
    assert det.source_type == "social"
    assert "timestamp" in det.resolved_slots


def test_actor_columns_kept_separate_from_counterparty():
    det = detect_dataframe(_bank_df())
    slots = {c.slot for c in det.columns.values()}
    # actor ids are pseudo-slots, not canonical ones
    assert "actor_id" in slots
    assert "actor_name" not in slots


def test_obscure_headers_content_probe(tmp_path):
    """Columns with meaningless names still resolve via content probing."""
    df = pd.DataFrame({
        "col1": pd.date_range("2024-01-01", periods=5, freq="h"),
        "col2": [10.25, 3.50, 99.99, 1.05, 7.75],
        "col3": ["acc1", "acc2", "acc3", "acc4", "acc5"],
    })
    det = detect_dataframe(df)
    assert det.source_type == "bank"
    assert "timestamp" in det.resolved_slots
    assert "amount" in det.resolved_slots


def test_unmapped_columns_reported():
    df = pd.DataFrame({"a": range(5), "zz_noise": ["x"] * 5})
    det = detect_dataframe(df)
    assert det.source_type == "unknown"


def test_classify_source_rules():
    assert classify_source({"amount"}) == "bank"
    assert classify_source({"duration"}) == "cdr"
    assert classify_source({"timestamp"}) == "social"
    assert classify_source(set()) == "unknown"
    # amount outranks duration/timestamp
    assert classify_source({"amount", "duration", "timestamp"}) == "bank"


def test_load_and_detect_from_file(tmp_path):
    import pandas as pd
    df = _bank_df()
    p = tmp_path / "bank.csv"
    df.to_csv(p, index=False)
    det = load_and_detect(str(p))
    assert det.source_type == "bank"
    assert det.n_rows == len(df)
