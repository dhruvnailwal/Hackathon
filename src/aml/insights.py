"""Plain-language insights for an analysis.

The score tables and model names stay in the JSON payload; what a human
reads from the report is *evidence phrased as insights* —
  * "Transferred money to the same person right after calling them (7×)"
  * "Repeatedly sent amounts just under the 10,000 reporting limit"
  * "Money flowed in from 12 people and out to 9 — a classic funnel"

Every insight is derived from the actual events in the unified dataframe,
never from raw model scores.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPORTING_THRESHOLD = 10000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def display_name(res, entity_id: str) -> str:
    """Prettier name for an entity: the resolved person's name if known,
    else the plain identifier."""
    names = getattr(res.entity_map, "entity_names", None) or {}
    cands = sorted(names.get(entity_id, set()))
    if cands:
        return cands[0]
    sub = res.unified
    if not sub.empty and "actor_name" in sub.columns:
        vals = sub.loc[sub["entity_id"] == entity_id, "actor_name"].dropna().unique()
        if len(vals):
            return str(vals[0])
    return entity_id


def _fmt(v: float) -> str:
    return f"{v:,.0f}"


# ---------------------------------------------------------------------------
# evidence extractors (each returns (happened: bool, sentence: str))
# ---------------------------------------------------------------------------
def _same_party_call_then_transfer(res, eid: str, tol_minutes: int) -> Tuple[bool, str]:
    """Bank transfers to someone soon after a call/post to that same person."""
    df = res.unified[res.unified["entity_id"] == eid]
    comm = df[df["source"] != "bank"].dropna(subset=["timestamp", "counterparty_id"])
    bank = df[df["source"] == "bank"].dropna(subset=["timestamp", "counterparty_id"])
    if comm.empty or bank.empty:
        return False, ""
    comm_ts = pd.to_datetime(comm["timestamp"]).to_numpy()
    comm_cp = comm["counterparty_id"].to_numpy(dtype=object)
    n = 0
    for brow in bank.itertuples():
        bts = pd.Timestamp(brow.timestamp)
        delta = np.abs((comm_ts - np.datetime64(bts.to_datetime64())) / np.timedelta64(1, "m"))
        if ((comm_cp == brow.counterparty_id) & (delta <= tol_minutes)).any():
            n += 1
    if n >= 1:
        return True, (f"transferred money to **the same person** they had just "
                      f"contacted by call/post within {tol_minutes} minutes — "
                      f"{n} matching transfer(s)")
    return False, ""


def _structuring_band(res, eid: str, threshold: float) -> Tuple[bool, str]:
    """Repeated payments engineered to sit just under the reporting limit."""
    df = res.unified[res.unified["entity_id"] == eid]
    amt = df.dropna(subset=["amount"]).copy()
    amt["amount"] = pd.to_numeric(amt["amount"], errors="coerce")
    amt = amt[amt["amount"] > 0]
    if amt.empty:
        return False, ""
    band = amt[(amt["amount"] >= 0.88 * threshold) & (amt["amount"] < threshold)]
    if len(band) >= 3:
        lo, hi = int(0.88 * threshold), int(threshold)
        return True, (f"kept sending amounts between {_fmt(lo)} and {_fmt(hi)} "
                      f"— just under the {_fmt(threshold)} reporting limit — "
                      f"{len(band)} times in total")
    return False, ""


def _scatter_gather(res, eid: str, scale: float = 1.0) -> Tuple[bool, str]:
    """Money arrives from many people AND leaves to many people."""
    df = res.unified[res.unified["entity_id"] == eid]
    edges = df.dropna(subset=["counterparty_id"]).dropna(subset=["amount"])
    if edges.empty:
        return False, ""
    dir_col = (df["direction"].fillna("out") if "direction" in df
               else pd.Series("out", index=df.index))
    in_n = edges[dir_col.reindex(edges.index).astype(str) == "credit"]["counterparty_id"].nunique()
    out_n = edges[dir_col.reindex(edges.index).astype(str) == "debit"]["counterparty_id"].nunique()
    if in_n >= 5 and out_n >= 8:
        return True, (f"money flowed in from **{in_n} different people** and "
                      f"right back out to **{out_n} others** — a classic "
                      f"collect-then-disperse (funnel) pattern")
    if in_n >= 10:
        return True, f"**received money from {in_n} different people** in the period reviewed"
    if out_n >= 10:
        return True, f"**sent money to {out_n} different people** in the period reviewed"
    return False, ""


def _burst_concentration(res, eid: str, hours: float = 48.0) -> Tuple[bool, str]:
    """Many different people dealt with inside a short window."""
    sub = res.unified[res.unified["entity_id"] == eid].dropna(subset=["timestamp", "counterparty_id"])
    if len(sub) < 10:
        return False, ""
    ts = pd.to_datetime(sub["timestamp"]).to_numpy(dtype="datetime64[ns]")
    cp = sub["counterparty_id"].to_numpy(dtype=object)
    window = np.timedelta64(int(hours * 3600), "s")
    best = max(len(set(cp[(ts >= ts[i] - window) & (ts <= ts[i] + window)])) for i in range(len(ts)))
    if best >= 12:
        return True, (f"dealt with **{best} different people inside a single "
                      f"{int(hours)}-hour window** — unusually compressed activity")
    return False, ""


def _dormancy(res, eid: str, quiet_days: float = 30.0) -> Tuple[bool, str]:
    """Long silence then a sudden burst (activation after dormancy)."""
    sub = res.unified[res.unified["entity_id"] == eid].dropna(subset=["timestamp"])
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    if len(sub) < 10:
        return False, ""
    ts = pd.to_datetime(sub["timestamp"])
    gaps = ts.diff().dt.total_seconds().dropna() / 86400.0
    if gaps.empty:
        return False, ""
    gmax = float(gaps.max())
    if gmax >= quiet_days:
        return True, (f"went completely quiet for **{gmax:,.0f} days** then "
                      f"re-activated all at once")
    return False, ""


def _big_lump(res, eid: str, q: float = 0.95, min_days_after: float = 14.0) -> Tuple[bool, str]:
    """One very large transfer followed by silence (laying low)."""
    sub = res.unified[res.unified["entity_id"] == eid].dropna(subset=["amount", "timestamp"])
    if len(sub) < 5:
        return False, ""
    amt = pd.to_numeric(sub["amount"], errors="coerce")
    qv = amt.quantile(q)
    if qv <= 0:
        return False, ""
    big = sub[amt >= qv]
    if big.empty:
        return False, ""
    last_ts = pd.to_datetime(sub["timestamp"]).max()
    big_ts = pd.to_datetime(big["timestamp"]).max()
    days_after = (last_ts - big_ts).total_seconds() / 86400.0
    if days_after >= min_days_after and (big["timestamp"] == big_ts).any():
        b_amount = float(big.loc[big["timestamp"] == big_ts, "amount"].iloc[0])
        return True, (f"moved a **large single amount ({_fmt(b_amount)})** and "
                      f"then went quiet for {days_after:,.0f} days")
    return False, ""


def _odd_amount_shape(res, eid: str, min_rows: int = 15) -> Tuple[bool, str]:
    """Payment amounts that look slightly 'too round' (no natural cents)."""
    sub = res.unified[res.unified["entity_id"] == eid].dropna(subset=["amount"])
    if len(sub) < min_rows:
        return False, ""
    vals = pd.to_numeric(sub["amount"], errors="coerce").dropna()
    if len(vals) < min_rows:
        return False, ""
    no_cents = float((((vals % 1).abs()) < 0.01).mean())
    if no_cents >= 0.5:
        return True, (f"more than half of the amounts ({no_cents:.0%}) are bare "
                      f"**round numbers with no cents** — real bills rarely look "
                      f"this clean")
    return False, ""


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _risk_level(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def build_insights(res, top_k: int = 8, tol_minutes: int = 25,
                   structuring_threshold: float = REPORTING_THRESHOLD) -> List[dict]:
    """Human-readable findings for the top-ranked entities.

    Each entry: {rank, entity_id, name, risk, headline, evidence[]}.
    """
    extractors: List[Callable] = [
        lambda r, e: _same_party_call_then_transfer(r, e, tol_minutes),
        (_scatter_gather),
        (_burst_concentration),
        lambda r, e: _structuring_band(r, e, structuring_threshold),
        (_big_lump),
        (_dormancy),
        _odd_amount_shape,
    ]
    out: List[dict] = []
    for i, r in enumerate(res.rankings[:top_k], 1):
        eid = r["entity_id"]
        evidence = []
        for fn in extractors:
            try:
                hit, sentence = fn(res, eid)
            except Exception:
                continue
            if hit:
                evidence.append(sentence)
        if not evidence:
            evidence.append("activity profile differs noticeably from the rest of "
                            "the people in this dataset")
        out.append({
            "rank": i,
            "entity_id": eid,
            "name": display_name(res, eid),
            "risk": _risk_level(r["score"]),
            "headline": evidence[0],
            "evidence": evidence,
        })
    return out