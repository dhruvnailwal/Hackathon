"""Optional tests that download real public sample files from the internet.

Proves the loaders + schema detection behave on real-world files (not just
synthetic or examples): a genuine CSV (pandas docs), a JSON array of records
(JSONPlaceholder), and an unstructured plain-text file (pandas README).

Network-gated: skipped automatically when the network is unavailable.
Run with:  pytest tests/test_internet_samples.py
"""
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from aml.loaders import load_any
from aml.pipeline import Pipeline

INTERNET_SAMPLES = {
    "tips.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
    "users.json": "https://jsonplaceholder.typicode.com/users",
    "readme.txt": "https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md",
}


def _fetch(url: str, dest: Path) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "aml-tests"})
        with urlopen(req, timeout=25) as r:
            dest.write_bytes(r.read())
        return True
    except Exception:
        return False


@pytest.mark.parametrize("name", list(INTERNET_SAMPLES))
def test_internet_file_loads(name, tmp_path):
    dest = tmp_path / name
    if not _fetch(INTERNET_SAMPLES[name], dest):
        pytest.skip(f"network unavailable for {name}")
    loaded = load_any(str(dest))
    assert loaded.format in ("csv", "json", "jsonl", "text")
    assert not loaded.df.empty


def test_pipeline_on_internet_csv(tmp_path):
    dest = tmp_path / "tips.csv"
    if not _fetch(INTERNET_SAMPLES["tips.csv"], dest):
        pytest.skip("network unavailable")
    # tips has currency-like numeric columns + a category; it will classify as
    # 'bank' because `amount`-like columns resolve. It must not crash a full run.
    res = Pipeline().run([str(dest)], with_models=False)
    assert res.per_source
    assert res.per_source[0].n_rows == 244
