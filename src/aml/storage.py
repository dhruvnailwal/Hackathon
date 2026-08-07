"""Persistent run storage — every analysis is archived outside the repo.

Runs land in the user's application-data directory (e.g. %APPDATA%\\AnomalyLens
on Windows, ~/.local/share/AnomalyLens on Linux/macOS), so history survives
restarts, uninstalls and reinstalls of the app. Each run folder holds the
full report set plus a machine-readable index.json:
    <user_data>/AnomalyLens/runs/<run_id>/
        index.json      <- summary + file manifest (used by History)
        report.md / report.html / report.pdf / report.json
        charts/*.png
        sources/*       <- copies of the analysed input files
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

APP_NAME = "AnomalyLens"


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


def save_run(result, config, source_paths: List[str], run_id: Optional[str] = None,
             label: str = "") -> dict:
    """Archive one pipeline result; returns the run manifest dict."""
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

    written = generate_report(result, config, run_dir)
    unified = result.unified
    n_entities = int(unified["entity_id"].nunique()) if not unified.empty else 0
    blocked = sum(1 for v in result.sufficiency.values() if v.status == "BLOCKED")
    top = result.rankings[:1]
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
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
