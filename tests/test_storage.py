import json
import shutil

from aml import storage
from aml.config import PipelineConfig
from aml.loaders import collect_files
from aml.pipeline import Pipeline


def _run_and_save(synthetic_dir, tmp_path, monkeypatch, label="test"):
    import platformdirs
    root = tmp_path / "userdata"
    monkeypatch.setattr(platformdirs, "user_data_dir",
                        lambda *a, **k: str(root / "TraceWeave"))
    src = tmp_path / "sources"
    src.mkdir(parents=True, exist_ok=True)
    for p in synthetic_dir.glob("sources/*"):
        shutil.copy2(p, src / p.name)
    paths = collect_files([str(src)])
    res = Pipeline().run(paths, with_models=True)
    return storage.save_run(res, PipelineConfig(), paths, label=label)


def test_save_run_roundtrips(synthetic_dir, tmp_path, monkeypatch):
    meta = _run_and_save(synthetic_dir, tmp_path, monkeypatch)
    assert meta["run_id"]
    assert meta["label"] == "test"
    assert meta["summary"]["events"] > 0
    d = storage.run_dir(meta["run_id"])
    assert (d / "report.md").exists()
    assert (d / "report.html").exists()
    assert (d / "report.pdf").exists()
    assert (d / "report.json").exists()
    assert (d / "index.json").exists()
    idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
    assert idx["run_id"] == meta["run_id"]


def test_list_runs_newest_first(synthetic_dir, tmp_path, monkeypatch):
    import time as _t
    a = _run_and_save(synthetic_dir, tmp_path, monkeypatch, label="a")
    _t.sleep(0.02)
    b = _run_and_save(synthetic_dir, tmp_path, monkeypatch, label="b")
    runs = storage.list_runs()
    assert runs[0]["run_id"] == b["run_id"]
    assert runs[1]["run_id"] == a["run_id"]


def test_delete_run(synthetic_dir, tmp_path, monkeypatch):
    meta = _run_and_save(synthetic_dir, tmp_path, monkeypatch)
    assert storage.load_run(meta["run_id"]) is not None
    assert storage.delete_run(meta["run_id"]) is True
    assert storage.load_run(meta["run_id"]) is None