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


def test_save_run_seals_and_verifies_ok(synthetic_dir, tmp_path, monkeypatch):
    meta = _run_and_save(synthetic_dir, tmp_path, monkeypatch)
    assert "chain_hash" in meta["integrity"]
    assert meta["integrity"]["n_files"] > 0
    v = storage.verify_run(meta["run_id"])
    assert v["status"] == "OK"
    assert all(status == "ok" for status in v["files"].values())


def test_verify_run_detects_tampering(synthetic_dir, tmp_path, monkeypatch):
    meta = _run_and_save(synthetic_dir, tmp_path, monkeypatch)
    report_md = storage.run_dir(meta["run_id"]) / "report.md"
    report_md.write_text(report_md.read_text().replace("Cross-Source", "TAMPERED"))
    v = storage.verify_run(meta["run_id"])
    assert v["status"] == "TAMPERED"
    assert v["files"]["report.md"] == "MISMATCH"


def test_verify_chain_links_successive_runs(synthetic_dir, tmp_path, monkeypatch):
    import time as _t
    a = _run_and_save(synthetic_dir, tmp_path, monkeypatch, label="a")
    _t.sleep(1.1)  # run_id resolution is per-second
    b = _run_and_save(synthetic_dir, tmp_path, monkeypatch, label="b")

    b_manifest = json.loads((storage.run_dir(b["run_id"]) / "manifest.json").read_text())
    assert b_manifest["prev_chain_hash"] == a["integrity"]["chain_hash"]

    chain = storage.verify_chain()
    assert chain["n_runs"] == 2
    assert chain["intact"] is True


def test_verify_chain_detects_manifest_tampering(synthetic_dir, tmp_path, monkeypatch):
    meta = _run_and_save(synthetic_dir, tmp_path, monkeypatch)
    manifest_path = storage.run_dir(meta["run_id"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["report.md"] = "0" * 64  # forge a hash without recomputing chain_hash
    manifest_path.write_text(json.dumps(manifest, indent=2))

    chain = storage.verify_chain()
    assert chain["intact"] is False
    assert meta["run_id"] in chain["broken_at"]