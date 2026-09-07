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

# content-probe thresholds (tunable; config.yaml -> Pipeline -> ProbeConfig)
@dataclass
class ProbeConfig:
    date_parse_ratio: float = 0.9      # fraction of values that must parse as datetime
    phone_ratio: float = 0.7           # fraction that must match the phone regex
    numeric_ratio: float = 0.9         # fraction that must parse as a number
    id_ratio: float = 0.7              # fraction that must match the account/ID regex
    amount_median_abs: float = 100.0   # |median| above which integer columns count as amounts
    amount_frac_ratio: float = 0.3     # fraction with fractional cents needed for amount
    amount_marker_ratio: float = 0.5   # fraction carrying currency markers
    big_code_magnitude: float = 10000.0  # pure-int column max above which = big numeric code
    amount_integer_median: float = 100.0  # minimum |median| for markerless integer amounts


# currency-marker regexes used by the amount probe
_CURRENCY_SYMBOLS = re.compile(r"[$€£₹¥₩]")
_CURRENCY_CODES = re.compile(
    r"\b(?:USD|EUR|GBP|INR|CAD|CNY|JPY|CHF|AUD|RUB|ZAR|NZD|SGD|SEK|NOK|DKK|Rs\.?)\b", re.I)

_ID_RE = re.compile(r"^(?=.*\d)[A-Za-z0-9][A-Za-z0-9_.\-]{2,19}$")
_PURE_NUM_RE = re.compile(r"^\d+(\.\d+)?$|^[+-]?\d+(\.\d+)?$")
_PAREN_NEG_RE = re.compile(r"^\(.*\)$")


def _parse_amount_value(s: str) -> Optional[float]:
    """Parse one amount-ish string to float.

    Handles: parentheses negatives ``(1,200.00)``, currency symbols/codes
    (``$``/``€``/``₹``/``USD``/``Rs.``), thousands separators (``1,200.00``),
    comma-decimal / euro style (``1.234,56``), and whitespace. Returns None
    for anything that does not look like a number.
    """
    t = s.strip()
    if not t:
        return None
    low = t.lower()
    if low in ("nan", "n/a", "na", "-", "none", "null", "—", "–", "nil", "missing"):
        return None
    neg = False
    if len(t) >= 2 and t[0] == "(" and t[-1] == ")":
        neg, t = True, t[1:-1].strip()
    t = _CURRENCY_SYMBOLS.sub("", t)
    t = _CURRENCY_CODES.sub("", t)
    t = t.strip().replace(" ", "")
    if not t:
        return None
    sign = 1.0
    if t[:1] in ("+", "-"):
        sign = -1.0 if t[0] == "-" else 1.0
        t = t[1:]
    if not t:
        return None
    # 1.234,56 or 1,234.56 (both separators present -> first is thousands)
    m = re.fullmatch(r"\d{1,3}([.,])\d{3}([.,])\d{1,2}", t)
    if m:
        first, second = m.group(1), m.group(2)
        if first != second:
            t = t.replace(first, "").replace(second, ".")
        else:
            t = t.replace(",", "")
        try:
            v = float(t)
            return -v if neg else sign * v
        except ValueError:
            return None
    # single separator, no dot: comma-decimal iff exactly 2 decimals after
    if t.count(",") == 1 and "." not in t:
        before, after = t.split(",")
        if before.isdigit() and after.isdigit():
            t = f"{before}.{after}" if len(after) == 2 else f"{before}{after}"
    elif t.count(".") == 1 and "," not in t:
        before, after = t.split(".")
        # dot-thousands only when the fraction is a full 3-digit group
        # (``1.234`` -> 1234); any other fraction length is a canonical
        # decimal, e.g. float reprs like ``130.85000000000002``
        if after.isdigit() and len(after) == 3 and before.isdigit():
            t = f"{before}{after}"  # 1.234 -> 1234 (thousands dot, no cents)
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else sign * v


def _amount_markers(sample: pd.Series) -> float:
    """Ratio of values carrying any currency marker (symbol, code, parens)."""
    marked = sample.str.contains(r"[$€£₹¥₩]|\(.*\)", regex=True) | \
        sample.str.contains(_CURRENCY_CODES)
    return float(marked.mean())


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
    normalized: pd.DataFrame = None  # canonical long-format frame from Stage-A

    @property
    def resolved_slots(self) -> set:
        return {c.slot for c in self.columns.values() if c.slot}


def _best_alias(col: str, choices: Dict[str, List[str]]) -> Tuple[Optional[str], float]:
    """Fuzzy-match column header to canonical slots; returns (slot, score).

    Headers are case-insensitive in practice (``TXN_DATE`` == ``txn_date``),
    so the raw header is normalised before scoring.
    """
    col_norm = col.lower().strip()
    candidates = {}
    for slot, aliases in choices.items():
        match = process.extractOne(
            col_norm, aliases, scorer=fuzz.token_set_ratio, score_cutoff=FUZZ_THRESHOLD
        )
        if match:
            candidates[slot] = (match[1], match[0])
    if not candidates:
        return None, 0.0
    # prefer slot with best score
    best = max(candidates.items(), key=lambda kv: kv[1][0])
    return best[0], best[1][0]

def _parse_ratio(dt: pd.Series) -> float:
    return float(_parse_datetime_series(pd.Series(dt)).notna().mean())


def _parse_one_datetime(x) -> pd.Timestamp:
    """Parse ONE value with the full tolerance toolkit.

    Plain ``pd.to_datetime`` treats pure-digit strings as ns-epoch (most
    real epochs overflow), so integer-looking strings get a second pass with
    seconds/milliseconds/microseconds units before giving up.

    Returns tz-naive timestamps only: mixed tz-aware + naive values in one
    column make ``pd.to_datetime`` NaN-out the whole series (pandas 3.x).
    """
    import warnings
    def _naive(v):
        return v.tz_localize(None) if v.tzinfo is not None else v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = pd.to_datetime(x, errors="coerce", utc=False)
    if not pd.isna(v):
        return _naive(v)
    if isinstance(x, str) and x.strip().isdigit():
        n = int(x.strip())
        for unit in ("s", "ms", "us"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = pd.to_datetime(n, unit=unit, errors="coerce")
            if not pd.isna(v) and 1990 <= v.year <= 2040:
                return _naive(v)
    return pd.NaT


def _parse_datetime_series(s: pd.Series, dayfirst: Optional[bool] = None) -> pd.Series:
    """Tolerant datetime parse: bulk first, element-wise fallback.

    pandas infers ONE format per series, so mixed-format columns (e.g.
    ISO-with-Z + naive + date-only) must fall back to element-wise parsing.

    Day-first heuristic: when both interpretations parse, month-first (US
    exports) wins; when day-first recovers meaningfully more values
    (e.g. ``25/01/2025`` that month-first rejects), prefer day-first.
    """
    import warnings
    # numeric columns: normalise unix-epoch (s/ms/us) before the ns-epoch
    # default — plain ``pd.to_datetime`` would map seconds-epochs to 1970
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9 and (num.fillna(0) == num.fillna(0).round()).mean() > 0.9:
        for unit, lo, hi in (("s", 1e8, 2e9), ("ms", 1e11, 1.5e12), ("us", 1e14, 1.5e15)):
            if num.dropna().abs().median() >= lo and num.dropna().abs().median() <= hi:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts = pd.to_datetime(num, unit=unit, errors="coerce")
                ok = ts.dropna().dt.year.between(1990, 2040).mean()
                if ok > 0.9:
                    return ts
    if dayfirst is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mf = pd.to_datetime(s, errors="coerce", utc=False)
            df_ = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=True)
        ratio_mf, ratio_df = float(mf.notna().mean()), float(df_.notna().mean())
        dayfirst = ratio_df > ratio_mf + 0.02
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=dayfirst)
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    if float(parsed.notna().mean()) > 0.9 or s.empty:
        return parsed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per = s.astype(str).map(lambda x: _parse_one_datetime(x))
    return pd.Series(per, index=s.index)


def _content_probe(col: pd.Series, probe_cfg: ProbeConfig = ProbeConfig()) -> Tuple[Optional[str], str]:
    """Infer slot from sampled values. Returns (slot, probe_name).

    Order matters (probe specificity):
      1. unix-epoch detection — 10-13 digit integers would otherwise match
         the phone regex and be silently misclassified as counterparties;
      2. long numeric identifiers — ``pd.to_datetime`` parses pure-digit
         strings as ns-epoch timestamps, so phone-like columns MUST be
         classified before the datetime probe;
      3. datetime parse — long pure-digit strings are excluded for the same
         reason; 8-digit ``yyyymmdd`` dates are still counted;
      4. amount parse (symbols / commas / parens) — currency columns;
      5. account / IBAN identifiers -> counterparty;
      6. typed numeric columns (integers without markers) -> amount or
         event_type by magnitude;
      7. name-ish / handle / geo-code fallbacks.
    """
    sample = col.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(200)
    if sample.empty:
        return None, ""

    # 1. unix epoch timestamps (seconds/milliseconds/microseconds)
    ints = pd.to_numeric(sample, errors="coerce")
    if ints.notna().mean() > probe_cfg.numeric_ratio and \
            (ints.fillna(0.0) == ints.fillna(0.0).round()).mean() > 0.9:
        big = ints.dropna().abs()
        if (big >= 1e8).mean() > 0.9:
            for unit, lo, hi in (("s", 1e8, 2e9), ("ms", 1e11, 1.5e12), ("us", 1e14, 1.5e15)):
                ts = pd.to_datetime(ints, unit=unit, errors="coerce")
                ok = ts.notna() & ts.dt.year.between(1990, 2040)
                if float(ok.mean()) > probe_cfg.date_parse_ratio:
                    return "timestamp", f"unix-epoch-{unit}"

    # 2. long numeric identifiers -> counterparty (edge dimension)
    is_phone = sample.str.fullmatch(r"^\+?\d{10,15}$")
    if is_phone.mean() > probe_cfg.phone_ratio:
        return "counterparty", "phone-regex"

    # 3. datetime parse (excluding long pure-digit strings = ns-epoch hazard)
    dt_sample = sample[~sample.str.fullmatch(r"\d{10,}")]
    dt_ratio = _parse_ratio(pd.Series(dt_sample, dtype=object))
    if dt_ratio > probe_cfg.date_parse_ratio:
        return "timestamp", f"datetime-parse {dt_ratio:.0%}"

    # 4. currency amount (symbols / separators / parens / codes)
    parsed = sample.map(_parse_amount_value)
    parsed_n = parsed.dropna()
    if parsed_n.abs().size and float(parsed.notna().mean()) > probe_cfg.numeric_ratio:
        frac = (parsed_n.abs() - parsed_n.abs().round(0)).abs()
        has_marker = _amount_markers(sample) > probe_cfg.amount_marker_ratio
        med_abs = float(parsed_n.abs().median())
        long_digits = float((parsed_n.abs() >= 1e9).mean())  # 10+ digit ids are NOT amounts
        if long_digits <= 0.1 and ((frac > 0.0).mean() > probe_cfg.amount_frac_ratio or has_marker):
            return "amount", "numeric-currency"
        if long_digits <= 0.1 and med_abs >= probe_cfg.amount_integer_median:
            return "amount", "numeric-int-large"

    # 5. geo-code-ish identifiers (tower/branch/lac) BEFORE the generic id
    #    regex — ``BRANCH23``/``T854`` would otherwise resolve as counterparty
    if sample.str.match(r"^(T|TOWER|CELL|LAC|BRANCH)[-_]?\d+$", flags=re.I).mean() > 0.7:
        return "location", "geo-code-pattern"

    # 6. account-ish identifiers -> counterparty (edge dimension)
    is_acct = sample.str.fullmatch(_ID_RE) & ~sample.str.fullmatch(_PURE_NUM_RE)
    if is_acct.mean() > probe_cfg.id_ratio:
        return "counterparty", "account-regex"

    # 7. remaining pure-numeric columns (no markers): big ints -> event_type
    #    code, small ints -> unmapped (ambiguous)
    num = pd.to_numeric(sample, errors="coerce")
    if num.notna().mean() > probe_cfg.numeric_ratio:
        big = float(num.abs().max(skipna=True)) >= probe_cfg.big_code_magnitude
        if big:
            return "event_type", "numeric-code"

    # 8. name-ish: two tokens of alphabetic words. The pattern alone isn't
    #    enough — a low-cardinality categorical column (e.g. currency names
    #    like "UK pounds" / "US Dollar") matches the same two-word shape and
    #    was previously misclassified as a person-name column, which risks
    #    spurious cross-source name-merges in entity_resolve.py. Real name
    #    columns vary a lot more than a handful of repeated category labels,
    #    so also require a minimum distinctness among the sampled values.
    two_words = sample.str.match(r"^[A-Za-z]+\s+[A-Za-z]+$").mean()
    if two_words > 0.7:
        distinct_ratio = sample.nunique() / len(sample)
        if sample.nunique() >= 5 and distinct_ratio > 0.15:
            return "actor_name", "name-pattern"

    if sample.str.startswith("@").mean() > 0.7:
        return "actor_handle", "handle-pattern"

    return None, ""


def detect_columns(df: pd.DataFrame, mode: str = "hybrid",
                   probe_cfg: ProbeConfig = None) -> Dict[str, ResolvedColumn]:
    """Map every column to a canonical slot (or flag as unmapped).

    ``mode`` (plan §3 Stage A candidates):
      * ``header``  — RapidFuzz header matching only (content never consulted)
      * ``content`` — content-probe inference only (headers ignored)
      * ``hybrid``  — header match first, content probe as fallback (default)
    """
    probe_cfg = probe_cfg or ProbeConfig()
    resolved: Dict[str, ResolvedColumn] = {}
    all_choices = {**FIELD_ALIASES, **ACTOR_ID_ALIASES, **ACTOR_NAME_ALIASES}
    for col in df.columns:
        slot, score = None, 0.0
        method = ""
        if mode != "content":
            slot, score = _best_alias(str(col), all_choices)
            if slot:
                method = "header"
        if mode == "content":
            # content probe as the sole signal (headers ignored)
            probed, probe_name = _content_probe(df[col], probe_cfg)
            if probed in FIELD_ALIASES or probed in PSEUDO_SLOTS:
                slot, score, method = probed, 70.0, f"content:{probe_name}"
        elif slot is None and mode == "hybrid":
            # header match first, content probe as fallback
            probed, probe_name = _content_probe(df[col], probe_cfg)
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


def _clean_ids(s: pd.Series) -> pd.Series:
    """String-cast a raw column to canonical dtype, collapsing the many
    'missing' spellings (nan / None / '') to NaN. JSON exports write null as
    the literal string ``None`` while CSV writes ``nan`` — without this, the
    same logical event loaded from two formats would never dedupe."""
    s = s.astype(str)
    return s.where(~s.str.lower().isin(("nan", "none", "<na>", "")), np.nan)


def normalize_source(
    df: pd.DataFrame, resolved: Dict[str, ResolvedColumn], source_type: str,
    probe_cfg: ProbeConfig = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Convert raw source df into the canonical long-format dataframe (SCHEMA.md)."""
    probe_cfg = probe_cfg or ProbeConfig()
    warnings: List[str] = []
    out = pd.DataFrame()

    ts_col = _pick_slot(df, resolved, "timestamp")
    amt_col = _pick_slot(df, resolved, "amount")
    cp_col = _pick_slot(df, resolved, "counterparty")
    loc_col = _pick_slot(df, resolved, "location")
    dir_col = _pick_slot(df, resolved, "direction")
    dur_col = _pick_slot(df, resolved, "duration")
    et_col = _pick_slot(df, resolved, "event_type")

    nan_series = pd.Series(np.nan, index=df.index)
    out["timestamp"] = _parse_datetime_series(df[ts_col]) if ts_col else pd.Series(pd.NaT, index=df.index)
    if ts_col is None:
        warnings.append("no timestamp column resolved; events undated")
    out["event_type"] = EVENT_TYPES.get(source_type, source_type)
    out["source"] = source_type
    # dtype discipline: absent numeric dimensions MUST stay float64/NaN so the
    # unified dataframe never degrades to object dtype (object amount columns
    # crash np.log1p in build_features, models.py:276)
    if amt_col:
        raw = df[amt_col].astype("string").str.strip()
        parsed = raw.map(_parse_amount_value, na_action="ignore")
        ratio = float(parsed.notna().mean())
        if ratio > 0.5:
            out["amount"] = parsed.fillna(np.nan).astype(float).to_numpy()
        else:
            # fall back to the lenient numeric coercion; warn about what fell out
            out["amount"] = pd.to_numeric(raw.str.replace(r"[$, ]", "", regex=True), errors="coerce")
            lost = float(out["amount"].notna().mean())
            if 0 <= lost < 0.9:
                warnings.append(
                    f"{source_type}: amount column '{amt_col}' partially unparseable "
                    f"({1 - lost:.0%} values dropped as NaN)"
                )
    else:
        out["amount"] = nan_series.copy()
        warnings.append(f"{source_type}: no amount dimension (only {EVENT_TYPES.get(source_type)} events carry it)")
    out["counterparty_id_raw"] = _clean_ids(df[cp_col]) if cp_col else nan_series.copy()
    if cp_col is None:
        warnings.append(f"{source_type}: no counterparty/edge column resolved")
    out["location"] = _clean_ids(df[loc_col]) if loc_col else nan_series.copy()
    out["direction"] = df[dir_col].astype(str).str.lower() if dir_col else nan_series.copy()
    if dir_col is None and source_type == "bank":
        # derive credit/debit from the txn-type column when it holds credit/debit values
        if et_col is not None:
            vals = df[et_col].astype(str).str.lower()
            derived = pd.Series(np.nan, index=df.index, dtype=object)
            derived[vals.str.contains("credit")] = "credit"
            derived[vals.str.contains("debit")] = "debit"
            out["direction"] = derived
        if out["direction"] is None:
            warnings.append("bank: no credit/debit direction column resolved; "
                            "fan-in/fan-out counts treat events as outgoing")
    out["duration"] = pd.to_numeric(df[dur_col], errors="coerce") if dur_col else nan_series.copy()

    # actor identifier per source (for entity resolution); fall back to the
    # actor NAME when no ID column resolves (content-only detection cannot
    # tell an actor ID from a counterparty ID, but the name is a valid
    # identity for Stage-B fuzzy resolution)
    actor = _pick_slot(df, resolved, "actor_id") or _pick_slot(df, resolved, "actor_name")
    out["actor_raw"] = _clean_ids(df[actor]) if actor else None
    # carry the actor name too, when available
    name_col = _pick_slot(df, resolved, "actor_name")
    if name_col:
        out["actor_name"] = _clean_ids(df[name_col])
    out = out.dropna(subset=["timestamp"]) if ts_col else out
    return out, warnings


def detect_dataframe(df: pd.DataFrame, file: str = "", mode: str = "hybrid",
                     probe_cfg: ProbeConfig = None) -> SourceDetection:
    """Stage-A detection over an already-loaded DataFrame (any source format).

    ``mode`` selects the detection candidate: ``header`` / ``content`` / ``hybrid``.
    """
    probe_cfg = probe_cfg or ProbeConfig()
    resolved = detect_columns(df, mode=mode, probe_cfg=probe_cfg)
    slots = {rc.slot for rc in resolved.values() if rc.slot}
    source_type = classify_source(slots)
    norm, warnings = normalize_source(df, resolved, source_type, probe_cfg=probe_cfg)
    raw_actor = [rc.original for rc in resolved.values() if rc.slot in ("actor_name", "actor_handle")]
    return SourceDetection(
        source_type=source_type, file=file, n_rows=len(df),
        columns=resolved, warnings=warnings, raw_actor_fields=raw_actor,
        normalized=norm,
    )


def load_and_detect(path: str, sample_limit: int = 10000, mode: str = "hybrid") -> SourceDetection:
    """Load any supported file (CSV/JSON/text) and run Stage-A detection."""
    from .loaders import load_any
    loaded = load_any(path)
    df = loaded.df.head(sample_limit)
    det = detect_dataframe(df, file=str(path), mode=mode)
    if loaded.warning:
        det.warnings.insert(0, loaded.warning)
    return det