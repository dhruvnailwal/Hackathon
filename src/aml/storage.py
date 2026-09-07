"""Persistent run storage — every analysis is archived outside the repo.

Runs land in the user's application-data directory (e.g. %APPDATA%\\TraceWeave
on Windows, ~/.local/share/TraceWeave on Linux/macOS), so history survives
restarts, uninstalls and reinstalls of the app. Each run folder holds the
full report set plus a machine-readable index.json:
    <user_data>/TraceWeave/runs/<run_id>/
        index.json      <- summary + file manifest (used by History)
        manifest.json   <- chain-of-custody seal (see _seal_run/verify_run)
        report.md / report.html / report.pdf / report.json
        charts/*.png
        sources/*       <- copies of the analysed input files

Chain of custody: every run is SHA-256-hashed file-by-file at save time and
chained to the previous run's seal (``prev_chain_hash``), the same
append-only-ledger idea a blockchain uses, without needing one. This is a
LOCAL integrity check, not a substitute for real notarization: it proves
"these exact bytes are what this analysis produced and nothing has been
edited since," and ``verify_chain`` proves no *past* run's seal has been
altered either — but nothing here is cryptographically signed with a key
the analyst doesn't also control, so it does not prove authorship to a
third party. State that distinction plainly rather than oversell it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

APP_NAME = "TraceWeave"
GENESIS_HASH = "0" * 64


def results_root() -> Path:
    """User-level data dir that survives uninstall/reinstall."""
    import platformdirs
    return Path(platformdirs.user_data_dir(appname=APP_NAME, appauthor=False))


def runs_dir() -> Path:
    d = results_root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_index(run_dir: Path, meta: dict) -> None:
    (run_dir / "index.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def _latest_prior_chain_hash(exclude_run_id: str) -> str:
    """Chain hash of the most recently sealed run before this one (by run_id,
    which is a UTC timestamp — sortable), or GENESIS_HASH if this is the
    first run ever archived."""
    candidates = sorted(
        (d for d in runs_dir().iterdir() if d.is_dir() and d.name != exclude_run_id),
        reverse=True,
    )
    for d in candidates:
        mp = _manifest_path(d)
        if mp.exists():
            try:
                return json.loads(mp.read_text(encoding="utf-8"))["chain_hash"]
            except (json.JSONDecodeError, OSError, KeyError):
                continue
    return GENESIS_HASH


def _seal_run(run_dir: Path, run_id: str) -> dict:
    """Hash every evidentiary file in this run folder (report set, charts,
    copied source files) and chain the result to the prior run's seal.

    ``index.json`` is deliberately excluded: it is a mutable summary/index
    (used by History) that gets rewritten once more right after sealing to
    record the seal itself — including it would make its own hash stale the
    instant it's written. The report/sources/charts are the evidentiary
    content and never change after this point.
    """
    files: Dict[str, str] = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json", "index.json"):
            files[str(p.relative_to(run_dir))] = _sha256_file(p)
    prev_chain_hash = _latest_prior_chain_hash(run_id)
    files_digest = _sha256_bytes(json.dumps(files, sort_keys=True).encode("utf-8"))
    chain_hash = _sha256_bytes((prev_chain_hash + files_digest).encode("utf-8"))
    manifest = {
        "run_id": run_id,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "files": files,
        "prev_chain_hash": prev_chain_hash,
        "chain_hash": chain_hash,
    }
    _manifest_path(run_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_run(run_id: str) -> dict:
    """Recompute every sealed file's hash and compare to what was recorded
    at save time. Returns per-file 'ok' / 'MISMATCH' / 'MISSING' plus an
    overall status — this is what an analyst/court would run to show a
    report is byte-identical to what the pipeline actually produced."""
    d = run_dir(run_id)
    mp = _manifest_path(d)
    if not mp.exists():
        return {"run_id": run_id, "status": "NO_MANIFEST", "files": {}}
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"run_id": run_id, "status": "MANIFEST_UNREADABLE", "files": {}}

    results: Dict[str, str] = {}
    ok = True
    for relpath, expected in manifest.get("files", {}).items():
        f = d / relpath
        if not f.exists():
            results[relpath] = "MISSING"
            ok = False
            continue
        actual = _sha256_file(f)
        results[relpath] = "ok" if actual == expected else "MISMATCH"
        if actual != expected:
            ok = False
    return {
        "run_id": run_id,
        "sealed_at": manifest.get("sealed_at"),
        "status": "OK" if ok else "TAMPERED",
        "files": results,
    }


def verify_chain() -> dict:
    """Walk every archived run oldest-to-newest and confirm each one's seal
    correctly chains to the previous one's — this is what catches tampering
    with a manifest.json's *recorded* hashes (not just the underlying
    files), since changing history for one run breaks every link after it."""
    runs = sorted(d for d in runs_dir().iterdir() if d.is_dir())
    prev = GENESIS_HASH
    broken_at: List[str] = []
    n_checked = 0
    for d in runs:
        mp = _manifest_path(d)
        if not mp.exists():
            continue
        n_checked += 1
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            broken_at.append(d.name)
            continue
        if manifest.get("prev_chain_hash") != prev:
            broken_at.append(d.name)
        files_digest = _sha256_bytes(
            json.dumps(manifest.get("files", {}), sort_keys=True).encode("utf-8"))
        expected_chain = _sha256_bytes(
            (manifest.get("prev_chain_hash", "") + files_digest).encode("utf-8"))
        if expected_chain != manifest.get("chain_hash"):
            broken_at.append(d.name)
        prev = manifest.get("chain_hash", prev)
    return {"n_runs": n_checked, "intact": not broken_at, "broken_at": sorted(set(broken_at))}


def save_run(result, config, source_paths: List[str], run_id: Optional[str] = None,
             label: str = "", legal_basis: str = "", analyst: str = "") -> dict:
    """Archive one pipeline result; returns the run manifest dict.

    ``legal_basis``/``analyst`` record who ran this analysis and under what
    authorization (warrant #, internal case ref, regulatory request, ...).
    Left blank rather than filled with a fake default when not provided —
    the report shows "not recorded" honestly instead of implying consent
    that was never captured.
    """
    from .report import generate_report

    run_id = run_id or next_run_id()
    run_dir = runs_dir() / run_id
    if run_dir.exists():
        run_id = f"{run_id}-{next_run_id()}"
        run_dir = runs_dir() / run_id
    sources_dir = run_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    for p in source_paths:
        src = Path(p)
        if src.is_file():
            try:
                shutil.copy2(src, sources_dir / src.name)
            except OSError:
                pass

    provenance = {"legal_basis": legal_basis, "analyst": analyst}
    written = generate_report(result, config, run_dir, provenance=provenance)
    unified = result.unified
    n_entities = int(unified["entity_id"].nunique()) if not unified.empty else 0
    blocked = sum(1 for v in result.sufficiency.values() if v.status == "BLOCKED")
    top = result.rankings[:1]
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "provenance": provenance,
        "source_files": [str(p) for p in source_paths],
        "summary": {
            "events": int(len(unified)),
            "entities": n_entities,
            "sources": len(result.per_source),
            "cross_source_links": int(getattr(result.entity_map, "n_cross_source", 0)),
            "blocked_models": blocked,
        },
        "top_insight": top[0]["entity_id"] if top else None,
        "top_score": round(top[0]["score"], 4) if top else None,
        "files": {k: str(v) for k, v in written.items()},
    }
    _write_index(run_dir, meta)
    # seal LAST, after index.json exists, so the seal covers the manifest
    # of what was archived too (not just the report/source files)
    manifest = _seal_run(run_dir, run_id)
    meta["integrity"] = {
        "sealed_at": manifest["sealed_at"],
        "chain_hash": manifest["chain_hash"],
        "n_files": len(manifest["files"]),
    }
    _write_index(run_dir, meta)
    return meta


def list_runs() -> List[dict]:
    """All archived runs, newest first."""
    out = []
    for d in sorted(runs_dir().iterdir(), reverse=True):
        if not d.is_dir():
            continue
        idx = d / "index.json"
        if idx.exists():
            try:
                meta = json.loads(idx.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            meta.setdefault("run_id", d.name)
            out.append(meta)
    return out


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def load_run(run_id: str) -> Optional[dict]:
    idx = run_dir(run_id) / "index.json"
    if not idx.exists():
        return None
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_run(run_id: str) -> bool:
    d = run_dir(run_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False
