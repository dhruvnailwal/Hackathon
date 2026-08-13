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
import re
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
    enc = "utf-8"
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False)  # default utf-8, strict
    except (UnicodeDecodeError, pd.errors.ParserError) as e:
        if isinstance(e, UnicodeDecodeError):
            # non-UTF8 export (latin-1/windows-1252): retry with a lossy
            # single-byte codec instead of crashing the whole pipeline
            enc = "latin-1"
            df = pd.read_csv(path, sep=sep, low_memory=False, encoding=enc)
        else:
            raise
    warning = "" if sep == "," else f"sniffed delimiter {sep!r}"
    if enc != "utf-8":
        warning = f"{warning} ; non-UTF8 file read as latin-1" if warning else "non-UTF8 file read as latin-1"
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


_KV_RE = re.compile(r"""(\S+)\s*=\s*("(?:\\.|[^"])*"|[^\s=]+)(?=\s|$)""")

# dynamic prose extraction — typed-token scanning for free-form / untabular
# logs. This is the "no predefined layout" path: it classifies *tokens* by
# their shape (ISO timestamp, currency amount, account/phone id, direction,
# event verb) rather than matching a fixed per-source pattern. No Freekhana
# formats are reused — only the idea of token-type inference.
_TS_RE = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?)?"
)
_CURRENCY_SYMBOLS = r"[$€£₹]"
_CURRENCY_SYMBOL_PREFIX = _CURRENCY_SYMBOLS
_CURRENCY_CODES = r"(?:USD|EUR|GBP|INR|CAD|CNY|JPY|SEK|NOK|AUD|CHF|RUB|ZAR)"
_RUPEE_WORD = r"(?:Rs\.?|रु|रू\.?)"
# a currency amount token: symbol prefix (incl. Rs./₹), *or* numeric followed
# by a code/word on either side — all matched before stripping, so
# '_strip_money' always leaves a bare number.
_MONEY_RE = re.compile(
    r"(?:"
    rf"{_CURRENCY_SYMBOL_PREFIX}\s?[\d][\d,]*(?:\.\d{{1,2}})?"
    rf"|{_RUPEE_WORD}\s?[\d][\d,]*(?:\.\d{{1,2}})?"
    rf"|[\d][\d,]*(?:\.\d{{1,2}})?\s?(?:{_CURRENCY_CODES}|{_RUPEE_WORD})"
    r"|(?:{_RUPEE_WORD}|{_CURRENCY_CODES})\s?[\d][\d,]*(?:\.\d{{1,2}})?"
    r")\b",
    re.IGNORECASE,
)


def _strip_money(tok: str) -> str:
    """Reduce a money token to a bare float-ready number.

    Handles ``$1,234.56``, ``Rs. 9,87.00``, ``₹ 500``, ``37.99 USD``, etc.,
    dropping the symbol/word/currency code regardless of which side it fell.
    """
    t = re.sub(_CURRENCY_SYMBOL_PREFIX, "", tok)
    t = re.sub(_RUPEE_WORD, "", t, flags=re.IGNORECASE)
    t = re.sub(_CURRENCY_CODES, "", t, flags=re.IGNORECASE)
    return t.replace(",", "").strip()
_ACCT_RE = re.compile(
    r"\b(?:ACCT[-_]?\w{3,}"
    r"|[A-Z]{2}\d{2}[A-Z0-9]{9,}"
    r"|(?=[A-Z0-9]{6,17}\b)(?=[A-Z0-9]*\d)[A-Z0-9]{6,17})",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\d)\+?\d{10,15}(?!\d)")
_HANDLE_RE = re.compile(r"@[\w.]+")
_DUR_RE = re.compile(r"\b(\d{1,4}(?:\.\d+)?)\s*(?:sec|second|seconds|secs)\b", re.I)
_FROM_TO_RE = re.compile(
    r"\b(?:from|of|by)\s+(@?[\w.\-]+)\b|\b(?:to|into|toward)\s+(@?[\w.\-]+)\b", re.I
)
_CREDIT_WORDS = re.compile(r"\b(?:credit|deposit|received|debit-credit|incoming|in)\b", re.I)
_DEBIT_WORDS = re.compile(r"\b(?:debit|withdraw|paid|sent|outgoing|out|transfer)\b", re.I)
_EVENT_WORDS = re.compile(
    r"\b(?:transfer|payment|deposit|withdrawal|purchase|swift|wire|call|sms|post|message|sent)\b", re.I
)
_EVENT_NAME = {"transfer": "transaction", "payment": "transaction", "deposit": "transaction",
               "withdrawal": "transaction", "swift": "transaction", "wire": "transaction",
               "sent": "transaction", "call": "call", "sms": "message", "post": "post",
               "message": "message"}


def _extract_prose_fields(line: str) -> Optional[dict]:
    """Scan one free-form line for typed tokens; returns a pseudo-record.

    Used only as a *safety net* when a text file is none of NDJSON / kv /
    delimited — real exporters and log shippers emit prose like
    ``2026-01-09 09:00:10 INFO transfer from ACCT-1001 to ACCT-4001 amount
    1000.00 USD``. We recover {ts, amount, account, counterparty, direction,
    event_type} heuristically so the untabular path still yields fields.
    """
    out: dict = {}
    ts = _TS_RE.search(line)
    if ts:
        out["ts"] = ts.group(0)
    money = _MONEY_RE.search(line)
    if money:
        out["amount"] = _strip_money(money.group(0))
    duration = _DUR_RE.search(line)
    if duration:
        out["duration"] = duration.group(1)
    # identifiers: order matters — 'from X' precedes 'to Y'
    handles = [(m.group(0), m.start()) for m in _HANDLE_RE.finditer(line)]
    # mask handles out before ACCT/phone scanning so 'duarte238' inside
    # '@x.y238' is not double-counted as a separate account id
    masked = list(line)
    for m in _HANDLE_RE.finditer(line):
        masked[m.start():m.end()] = " " * (m.end() - m.start())
    masked = "".join(masked)
    accts = [(m.group(0), m.start()) for m in _ACCT_RE.finditer(masked)]
    phones = [(m.group(0), m.start()) for m in _PHONE_RE.finditer(masked)]
    tokens = sorted(accts + phones + handles, key=lambda t: t[1])
    ids = list(dict.fromkeys(t for t, _ in tokens))
    # 'from/of X' -> account (actor), 'to/into X' -> counterparty
    mf = _FROM_TO_RE.search(line)
    if mf and mf.group(1):
        g = mf.group(1)
        if g in ids or g[1:] in ids:
            out["account"] = g
    mt = _FROM_TO_RE.search(line)
    while mt and not mt.group(2):
        mt = _FROM_TO_RE.search(line, mt.end())
    if mt and mt.group(2):
        g = mt.group(2)
        if g in ids or g[1:] in ids:
            out["counterparty"] = g
    for tok in ids:
        if tok == out.get("account") or tok[1:] == out.get("account"):
            continue
        if tok == out.get("counterparty"):
            continue
        out.setdefault("counterparty", tok)
        break
    # NOTE: no lone-id -> account fallback. An id with no explicit 'from/of'
    # marker is ambiguous; only the counterparty slot is claimed, so an
    # entity-less input stays honest (entity resolution degrades).
    if _CREDIT_WORDS.search(line) and not _DEBIT_WORDS.search(line):
        out["direction"] = "credit"
    elif _DEBIT_WORDS.search(line):
        out["direction"] = "debit"
    ev = _EVENT_WORDS.search(line)
    if ev:
        out["event_type"] = _EVENT_NAME.get(ev.group(0).lower(), ev.group(0))
    return out


def _kv_pairs(line: str) -> List[tuple]:
    """Extract (key, value) pairs from a kv log line. Values may be
    double-quoted to contain spaces: ``c0="2024-01-19 09:00:00"``.
    """
    out = []
    for m in _KV_RE.finditer(line):
        k, v = m.group(1), m.group(2)
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        out.append((k, v))
    return out


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

    # pass 2: key=value lines (single-pair lines are valid kv logs; quoted
    # values may contain spaces)
    kv_rows = []
    for ln in non_empty:
        pairs = _kv_pairs(ln)
        if len(pairs) >= 1:
            kv_rows.append(dict(pairs))
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
    # pass 4: dynamic prose inference (free-form / untabular safety net) —
    # typed-token scanning, not a per-source pattern. Accepted when the
    # majority of lines carry a credible dimension mix: a timestamp alone
    # (timestamped log stream) or >=2 of ts/amount/account/counterparty.
    prose_rows = []
    for ln in non_empty:
        r = _extract_prose_fields(ln)
        if r:
            dims = {"ts", "amount", "account", "counterparty"} & set(r)
            if "ts" in r or len(dims) >= 2:
                prose_rows.append(r)
    if prose_rows and len(prose_rows) >= 0.5 * len(non_empty):
        return prose_rows
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
