"""Multi-format data loaders (CSV / JSON / text) -> DataFrame.

The pipeline accepts *any* of these inputs and relies on Stage-A schema
detection to interpret the columns. Every loader returns a normalised
``LoadedSource`` so downstream code never has to care which format a file
was in.

Format handling:
  * CSV/TSV             — separator sniffed from the first non-empty line
  * JSON (.json array / .jsonl / dict-of-lists) — normalised record-wise
  * TXT/LOG             — best-effort: NDJSON lines, delimited tables,
                          ``key=value`` lines, else a single free-form text
                          column
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class LoadedSource:
    df: pd.DataFrame
    file: str
    format: str  # csv | json | text
    warning: str = ""


SUPPORTED_EXTENSIONS = {".csv", ".csv.gz", ".tsv", ".txt", ".log", ".json", ".jsonl", ".ndjson"}
_DELIMS = [",", "\t", ";", "|"]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def _sniff_sep(first_line: str) -> str:
    for d in _DELIMS:
        if d in first_line:
            return d
    return ","


def load_csv(path: Path) -> LoadedSource:
    head = ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.readline()
    except OSError:
        pass
    sep = _sniff_sep(head)
    df = pd.read_csv(path, sep=sep, low_memory=False)
    warning = "" if sep == "," else f"sniffed delimiter {sep!r}"
    return LoadedSource(df=df, file=str(path), format="csv", warning=warning)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def _normalize_records(records: List) -> List[dict]:
    out: List[dict] = []
    for r in records:
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, list):
            out.append({f"col{i}": v for i, v in enumerate(r)})
    return out


def _records_from_json(obj) -> Optional[List[dict]]:
    """Coerce JSON into a list of records (dicts)."""
    if isinstance(obj, dict):
        for key in ("data", "results", "records", "items"):
            if key in obj and isinstance(obj[key], list):
                obj = obj[key]
                break
        else:
            lists = {k: v for k, v in obj.items() if isinstance(v, list) and v}
            if lists:
                n = len(next(iter(lists.values())))
                if all(len(v) == n for v in lists.values()):
                    return [dict(zip(lists, row)) for row in zip(*lists.values())]
            return [obj]
    if isinstance(obj, list):
        return _normalize_records(obj)
    return None


def load_json(path: Path) -> LoadedSource:
    fmt = str(path.suffix).lower()
    if fmt in (".jsonl", ".ndjson"):
        return load_ndjson(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)
    records = _records_from_json(data) or []
    df = pd.DataFrame(records)
    return LoadedSource(df=df, file=str(path), format="jsonl" if fmt in (".jsonl", ".ndjson") else "json")


def load_ndjson(path: Path) -> LoadedSource:
    records: List = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(_normalize_records(records))
    return LoadedSource(df=df, file=str(path), format="jsonl")


# ---------------------------------------------------------------------------
# TEXT
# ---------------------------------------------------------------------------
def _parse_text_lines(lines: List[str]) -> Optional[List[dict]]:
    """Try structured text. Returns records or None if it looks unstructured."""
    lines = [ln.rstrip("\n") for ln in lines]
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return None

    # pass 1: NDJSON-looking lines
    objs = []
    for ln in non_empty:
        s = ln.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                objs.append(json.loads(s))
            except json.JSONDecodeError:
                pass
    if objs and len(objs) >= 0.5 * len(non_empty):
        return _normalize_records(objs)

    # pass 2: key=value lines
    kv_rows = []
    for ln in non_empty:
        parts = [p for p in ln.split() if "=" in p]
        if len(parts) >= 2:
            try:
                kv_rows.append(dict(p.split("=", 1) for p in parts))
            except ValueError:
                pass
    if kv_rows and len(kv_rows) >= 0.5 * len(non_empty):
        return kv_rows

    # pass 3: delimited table with a header row
    head = lines[0]
    for delim in _DELIMS:
        if delim in head and head.count(delim) >= 1:
            ncols = head.count(delim)
            ok = all(delim in ln or not ln.strip() for ln in lines[1:])
            if ok and sum(1 for ln in lines[1:] if ln.count(delim) == ncols) >= 0.5 * len(lines[1:]):
                header = [c.strip() for c in head.split(delim)]
                rows = []
                for ln in lines[1:]:
                    if not ln.strip():
                        continue
                    cells = ln.split(delim)
                    rows.append(dict(zip(header, cells + [""] * (len(header) - len(cells)))))
                return rows
    return None


def load_text(path: Path) -> LoadedSource:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    records = _parse_text_lines(lines)
    if records:
        df = pd.DataFrame(records)
        warning = ""
    else:
        # unstructured -> single free-form text column
        df = pd.DataFrame({"text": [ln.rstrip("\n") for ln in lines if ln.strip()]})
        warning = "no structured layout detected; loaded as a single free-form text column"
    return LoadedSource(df=df, file=str(path), format="text", warning=warning)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def load_any(path: str) -> LoadedSource:
    """Load a file by its extension into a normalised DataFrame."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".csv", ".csv.gz", ".tsv", ".txt", ".log"):
        if suffix in (".txt", ".log"):
            return load_text(p)
        return load_csv(p)
    if suffix in (".json", ".jsonl", ".ndjson"):
        return load_json(p)
    raise ValueError(f"unsupported file type {suffix!r} for {path} "
                     f"(supported: {sorted(SUPPORTED_EXTENSIONS)})")


def is_supported(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def collect_files(paths: List[str]) -> List[str]:
    """Expand file/dir args into a deduplicated list of supported files."""
    out: List[str] = []
    seen = set()
    for p in paths:
        path = Path(p)
        if path.is_dir():
            candidates = sorted(str(p) for p in path.rglob("*") if p.is_file())
            for candidate in candidates:
                if is_supported(candidate):
                    key = str(Path(candidate).resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        out.append(candidate)
        elif path.is_file() and is_supported(str(path)):
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(str(path))
    return out
