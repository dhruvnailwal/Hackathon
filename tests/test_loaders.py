import json
from pathlib import Path

import pandas as pd

from aml.loaders import (
    SUPPORTED_EXTENSIONS,
    collect_files,
    load_any,
    load_json,
    load_text,
)


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


def test_csv_delimiter_sniff(tmp_path):
    p = _write(tmp_path, "a.tsv", "ts\tamount\n2024-01-01\t10.5\n")
    r = load_any(str(p))
    assert r.format == "csv"
    assert r.df.shape == (1, 2)
    assert list(r.df.columns) == ["ts", "amount"]


def test_json_array_records(tmp_path):
    p = _write(tmp_path, "b.json", '[{"id":1,"amount":5},{"id":2,"amount":7}]')
    r = load_json(p)
    assert r.df.shape == (2, 2)
    assert r.df["amount"].tolist() == [5, 7]


def test_json_dict_of_lists(tmp_path):
    p = _write(tmp_path, "c.json", '{"ts":["2024-01-01"],"amount":[9.9]}')
    r = load_json(p)
    assert r.df.shape == (1, 2)


def test_json_data_wrapper(tmp_path):
    p = _write(tmp_path, "d.json", '{"data":[{"ts":"2024-01-01","v":1}]}')
    r = load_json(p)
    assert r.df.shape == (1, 2)


def test_ndjson(tmp_path):
    p = _write(tmp_path, "e.jsonl", '{"ts":"2024-01-01","dur":5}\n{"ts":"2024-01-02","dur":9}\n')
    r = load_json(p)
    assert r.format == "jsonl"
    assert r.df.shape == (2, 2)


def test_ndjson_ignores_comments(tmp_path):
    p = _write(tmp_path, "f.ndjson", "# header\n{\"a\":1}\n")
    r = load_json(p)
    assert r.df.shape == (1, 1)


def test_text_keyvalue(tmp_path):
    p = _write(tmp_path, "g.txt",
               "ts=2024-01-01T10:00:00Z handle=u1 mentions=u2\n"
               "ts=2024-01-02T10:00:00Z handle=u3 mentions=u4\n")
    r = load_text(p)
    assert r.df.shape == (2, 3)
    assert {"handle", "ts", "mentions"} <= set(r.df.columns)


def test_text_delimited_table(tmp_path):
    p = _write(tmp_path, "h.log",
               "timestamp\tcaller\tcallee\tseconds\n"
               "2024-01-01\t911\t912\t120\n"
               "2024-01-02\t911\t913\t60\n")
    r = load_text(p)
    assert r.df.shape == (2, 4)


def test_text_unstructured_single_column(tmp_path):
    p = _write(tmp_path, "i.txt", "just some notes\nwithout structure\n")
    r = load_text(p)
    assert list(r.df.columns) == ["text"]
    assert r.warning


def test_text_prose_log_typed_fields(tmp_path):
    """Free-form log lines still yield typed fields (dynamic token inference)."""
    p = _write(tmp_path, "prose.log",
               "2022-09-01 00:20:10 INFO transfer from 8000EBD270 to 8000EBD520 "
               "amount 3697.34 USD\n"
               "2022-09-01 00:25:03 INFO transfer from 8000EBD270 to 8000EBD520 "
               "amount 214.70 USD\n")
    r = load_text(p)
    assert r.format == "text"
    assert {"ts", "amount", "account", "counterparty", "event_type"} <= set(r.df.columns)
    assert r.df["amount"].iloc[0] == "3697.34"
    assert r.df["account"].iloc[0] == "8000EBD270"
    assert r.df["counterparty"].iloc[0] == "8000EBD520"
    assert not r.warning


def test_text_prose_inr_amounts(tmp_path):
    """Rupee-denominated free-form lines normalize to bare numbers."""
    p = _write(tmp_path, "inr.log",
               "2022-09-01 00:20:10 INFO transfer from ACCT-1001 to ACCT-2002 "
               "amount Rs. 9,87,654.00\n"
               "2022-09-01 00:25:03 INFO transfer from ACCT-1001 to ACCT-2002 "
               "amount ₹ 45,000.00\n")
    r = load_text(p)
    assert r.df["amount"].iloc[0] == "987654.00"
    assert r.df["amount"].iloc[1] == "45000.00"


def test_text_prose_ignores_plain_notes(tmp_path):
    p = _write(tmp_path, "notes.txt",
               "the quick brown fox\njumps over the lazy dog\nmeeting at noon\n")
    r = load_text(p)
    assert list(r.df.columns) == ["text"]
    assert r.warning


def test_text_prose_handles_and_duration(tmp_path):
    """Social handles (@x.y) and CDR durations are recovered as typed tokens."""
    p = _write(tmp_path, "mixed.log",
               "2022-09-01 00:20:10 call from 6391000001 to 6391000002 "
               "duration 45 seconds\n"
               "2022-09-01 00:40:00 post by @ana.duarte238 "
               "mentions @ben.tanaka901\n")
    r = load_text(p)
    assert r.format == "text"
    assert not r.warning
    pairs = r.df.sort_values("ts").reset_index(drop=True)
    assert pairs["account"].iloc[0] == "6391000001"
    assert pairs["counterparty"].iloc[0] == "6391000002"
    assert pairs["duration"].iloc[0] == "45"
    assert pairs["event_type"].iloc[0] == "call"
    assert pairs["account"].iloc[1] == "@ana.duarte238"
    assert pairs["counterparty"].iloc[1] == "@ben.tanaka901"
    assert pairs["event_type"].iloc[1] == "post"


def test_unsupported_extension(tmp_path):
    p = _write(tmp_path, "j.xlsx", "nope")
    import pytest
    with pytest.raises(ValueError):
        load_any(str(p))


def test_collect_files_dedup_and_expand(tmp_path):
    _write(tmp_path, "k.csv", "a\n1\n")
    _write(tmp_path, "l.json", "[]")
    _write(tmp_path, "m.bin", "b")
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub, "n.txt", "x")
    files = collect_files([str(tmp_path), str(tmp_path / "k.csv")])
    names = sorted(Path(f).name for f in files)
    assert names == ["k.csv", "l.json", "n.txt"]
