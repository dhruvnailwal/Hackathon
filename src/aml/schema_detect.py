"""Stage A — dynamic schema detection (§2.2).

Hybrid of (1) rapidfuzz header fuzzy-matching against FIELD_ALIASES and
(2) content-probe inference ("parses as datetime 90%+ -> timestamp; phone
regex -> counterparty; currency precision -> amount"). Source classification
(bank / cdr / social) falls out as a byproduct of which canonical slots resolved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .schema import ACTOR_ID_ALIASES, ACTOR_NAME_ALIASES, EVENT_TYPES, FIELD_ALIASES

FUZZ_THRESHOLD = 80  # plan §2.2
PHONE_RE = re.compile(r"^\+?\d{6,15}$")

# pseudo-slots that the content probe may produce (kept out of canonical slots)
PSEUDO_SLOTS = {"actor_id", "actor_name", "actor_handle"}


@dataclass
class ResolvedColumn:
    original: str
    slot: str  # canonical slot, or "" if unmapped
    method: str  # header | content | unmapped
    score: float = 0.0
    sample: object = None


@dataclass
class SourceDetection:
    source_type: str  # bank | cdr | social
    file: str
    n_rows: int
    columns: Dict[str, ResolvedColumn]  # canonical slot -> ResolvedColumn
    warnings: List[str] = field(default_factory=list)
    raw_actor_fields: List[str] = field(default_factory=list)  # names/phones/accounts/handles

    @property
    def resolved_slots(self) -> set:
        return {c.slot for c in self.columns.values() if c.slot}


def _best_alias(col: str, choices: Dict[str, List[str]]) -> Tuple[Optional[str], float]:
    """Fuzzy-match column header to canonical slots; returns (slot, score)."""
    candidates = {}
    for slot, aliases in choices.items():
        match = process.extractOne(
            col, aliases, scorer=fuzz.token_set_ratio, score_cutoff=FUZZ_THRESHOLD
        )
        if match:
            candidates[slot] = (match[1], match[0])
    if not candidates:
        return None, 0.0
    # prefer slot with best score
    best = max(candidates.items(), key=lambda kv: kv[1][0])
    return best[0], best[1][0]

def _parse_ratio(dt: pd.Series) -> float:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(dt, errors="coerce", utc=False)
    return float(dt.notna().mean())


def _content_probe(col: pd.Series) -> Tuple[Optional[str], str]:
    """Infer slot from sampled values. Returns (slot, probe_name)."""
    sample = col.dropna().head(200).astype(str)
    if sample.empty:
        return None, ""

    dt_ratio = _parse_ratio(pd.Series(sample, dtype=object))
    if dt_ratio > 0.9:
        return "timestamp", f"datetime-parse {dt_ratio:.0%}"

    # phone / account-ish identifiers -> counterparty (edge dimension)
    is_phone = sample.str.fullmatch(PHONE_RE)
    is_acct = sample.str.match(r"^(ACCT|acc)?\d{6,}$")
    if is_phone.mean() > 0.7:
        return "counterparty", "phone-regex"
    if is_acct.mean() > 0.7:
        return "counterparty", "account-regex"

    # currency amount: try float parse with precision
    num = pd.to_numeric(sample.str.replace(r"[$,]", "", regex=True), errors="coerce")
    if num.notna().mean() > 0.9:
        frac = num - num.round(0)
        if (frac.abs() > 0.0).mean() > 0.3:
            return "amount", "numeric-currency"
        return "event_type", "numeric-code"

    # name-ish: two tokens of alphabetic words
    two_words = sample.str.match(r"^[A-Za-z]+\s+[A-Za-z]+$").mean()
    if two_words > 0.7:
        return "actor_name", "name-pattern"

    if sample.str.startswith("@").mean() > 0.7:
        return "actor_handle", "handle-pattern"

    if sample.str.match(r"^T\d+$").mean() > 0.7 or sample.str.match(r"^BRANCH\d+$").mean() > 0.7:
        return "location", "geo-code-pattern"

    return None, ""


def detect_columns(df: pd.DataFrame) -> Dict[str, ResolvedColumn]:
    """Map every column to a canonical slot (or flag as unmapped)."""
    resolved: Dict[str, ResolvedColumn] = {}
    all_choices = {**FIELD_ALIASES, **ACTOR_ID_ALIASES, **ACTOR_NAME_ALIASES}
    for col in df.columns:
        slot, score = _best_alias(str(col), all_choices)
        method = "header"
        if slot is None:
            # content probe fallback
            probed, probe_name = _content_probe(df[col])
            if probed in FIELD_ALIASES or probed in PSEUDO_SLOTS:
                slot, score, method = probed, 70.0, f"content:{probe_name}"
        resolved[col] = ResolvedColumn(original=col, slot=slot or "", method=method, score=score)
    return resolved


def classify_source(resolved_slots: set) -> str:
    """Source type falls out of which canonical slots resolved (§1B)."""
    has_amount = "amount" in resolved_slots
    has_duration = "duration" in resolved_slots
    if has_amount:
        return "bank"
    if has_duration:
        return "cdr"
    if "timestamp" in resolved_slots:
        return "social"
    return "unknown"


def _pick_slot(df: pd.DataFrame, resolved: Dict[str, ResolvedColumn], slot: str) -> Optional[str]:
    """Choose the best column that resolved to `slot` (prefer header match)."""
    cols = [rc.original for rc in resolved.values() if rc.slot == slot]
    if not cols:
        return None
    # prefer a header match, else the content-probe col
    def rank(c):
        rc = resolved[c]
        return (rc.method == "header", rc.score)
    return max(cols, key=rank)


def normalize_source(
    df: pd.DataFrame, resolved: Dict[str, ResolvedColumn], source_type: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Convert raw source df into the canonical long-format dataframe (SCHEMA.md)."""
    warnings: List[str] = []
    out = pd.DataFrame()

    ts_col = _pick_slot(df, resolved, "timestamp")
    amt_col = _pick_slot(df, resolved, "amount")
    cp_col = _pick_slot(df, resolved, "counterparty")
    loc_col = _pick_slot(df, resolved, "location")
    dir_col = _pick_slot(df, resolved, "direction")
    dur_col = _pick_slot(df, resolved, "duration")
    et_col = _pick_slot(df, resolved, "event_type")

    out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce") if ts_col else pd.NaT
    if ts_col is None:
        warnings.append("no timestamp column resolved; events undated")
    out["event_type"] = EVENT_TYPES.get(source_type, source_type)
    out["source"] = source_type
    out["amount"] = None
    if amt_col:
        out["amount"] = pd.to_numeric(
            df[amt_col].astype(str).str.replace(r"[$, ]", "", regex=True), errors="coerce"
        )
    else:
        warnings.append(f"{source_type}: no amount dimension (only {EVENT_TYPES.get(source_type)} events carry it)")
    out["counterparty_id_raw"] = df[cp_col].astype(str).replace("nan", np.nan) if cp_col else None
    if cp_col is None:
        warnings.append(f"{source_type}: no counterparty/edge column resolved")
    out["location"] = df[loc_col].astype(str).replace("nan", np.nan) if loc_col else None
    out["direction"] = df[dir_col].astype(str).str.lower() if dir_col else None
    if dir_col is None and source_type == "bank":
        # derive credit/debit from the txn-type column when it holds credit/debit values
        if et_col is not None:
            vals = df[et_col].astype(str).str.lower()
            out["direction"] = np.where(vals.str.contains("credit"), "credit",
                                        np.where(vals.str.contains("debit"), "debit", None))
        if out["direction"] is None:
            warnings.append("bank: no credit/debit direction column resolved; "
                            "fan-in/fan-out counts treat events as outgoing")
    out["duration"] = pd.to_numeric(df[dur_col], errors="coerce") if dur_col else None

    # actor identifier per source (for entity resolution)
    actor = _pick_slot(df, resolved, "actor_id")
    out["actor_raw"] = df[actor].astype(str).replace("nan", np.nan) if actor else None
    # carry the actor name too, when available
    name_col = _pick_slot(df, resolved, "actor_name")
    if name_col:
        out["actor_name"] = df[name_col].astype(str).replace("nan", np.nan)
    out = out.dropna(subset=["timestamp"]) if ts_col else out
    return out, warnings


def detect_dataframe(df: pd.DataFrame, file: str = "") -> SourceDetection:
    """Stage-A detection over an already-loaded DataFrame (any source format)."""
    resolved = detect_columns(df)
    slots = {rc.slot for rc in resolved.values() if rc.slot}
    source_type = classify_source(slots)
    norm, warnings = normalize_source(df, resolved, source_type)
    raw_actor = [rc.original for rc in resolved.values() if rc.slot in ("actor_name", "actor_handle")]
    return SourceDetection(
        source_type=source_type, file=file, n_rows=len(df),
        columns=resolved, warnings=warnings, raw_actor_fields=raw_actor,
    )


def load_and_detect(path: str, sample_limit: int = 10000) -> SourceDetection:
    """Load any supported file (CSV/JSON/text) and run Stage-A detection."""
    from .loaders import load_any
    loaded = load_any(path)
    df = loaded.df.head(sample_limit)
    det = detect_dataframe(df, file=str(path))
    if loaded.warning:
        det.warnings.insert(0, loaded.warning)
    return det