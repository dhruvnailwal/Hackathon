"""Plain-language insights for an analysis.

The score tables and model names stay in the JSON payload; what a human
reads from the report is *evidence phrased as insights* —
  * "Transferred money to the same person right after calling them (7×)"
  * "Repeatedly sent amounts just under the 10,000 reporting limit"
  * "Money flowed in from 12 people and out to 9 — a classic funnel"

Every insight is derived from the actual events in the unified dataframe,
never from raw model scores. Each evidence sentence carries its *own*
evidence trail — the specific records that back that specific claim (who
the 21 people were, which 18 transactions sat under the threshold, ...) —
rather than one merged pile per person, so a reader can see exactly which
records back exactly which claim.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPORTING_THRESHOLD = 10000.0

# extractors collect (essentially) every matching record — real per-entity
# counts are bounded (tens, not thousands) — this is only a safety ceiling
# against a pathological outlier entity, not a display limit. The display
# limit (how many rows a report actually prints) lives in report.py, which
# also gets the true underlying count so it can say "30 of 45 shown"; the
# JSON payload always carries this full, uncapped list.
_MAX_ROWS_PER_ITEM = 200
_MAX_PAIRS = _MAX_ROWS_PER_ITEM // 2


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


def _fmt_ts(ts) -> str:
    try:
        if pd.isna(ts):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts) if ts is not None else "—"


def _evidence_row(res, row: pd.Series) -> dict:
    """One line-item record backing a piece of evidence: who, what, when,
    with whom, and how much — pulled straight from the unified dataframe."""
    cp = row.get("counterparty_id")
    cp_name = display_name(res, cp) if pd.notna(cp) else "—"
    amount = row.get("amount")
    duration = row.get("duration")
    return {
        "timestamp": _fmt_ts(row.get("timestamp")),
        "source": str(row.get("source")) if pd.notna(row.get("source")) else "—",
        "event_type": str(row.get("event_type")) if pd.notna(row.get("event_type")) else "—",
        "counterparty": cp_name,
        "amount": (f"{float(amount):,.0f}" if pd.notna(amount) else None),
        "duration": (f"{float(duration):,.0f} min" if pd.notna(duration) else None),
    }


def _dedupe_sort(rows: List[dict], cap: int) -> List[dict]:
    seen = set()
    out = []
    for row in sorted(rows, key=lambda r: r["timestamp"]):
        key = (row["timestamp"], row["source"], row["counterparty"], row["amount"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out[:cap]


def _by_person(edf: pd.DataFrame, res, k: int) -> List[dict]:
    """Up to k rows, largest first, one per distinct counterparty — for
    'who are the N people' style evidence."""
    picked = edf.sort_values("amount", ascending=False).drop_duplicates(subset="counterparty_id").head(k)
    return [_evidence_row(res, r) for _, r in picked.iterrows()]


# ---------------------------------------------------------------------------
# evidence extractors
# each returns (happened: bool, sentence: str, trail: List[dict], total: int)
# ``total`` is the true underlying record/person count the sentence refers
# to; ``trail`` may be a capped subset of it (report.py shows "X of Y").
# ---------------------------------------------------------------------------
def _same_party_call_then_transfer(res, eid: str, tol_minutes: int) -> Tuple[bool, str, List[dict], int]:
    """Bank transfers to someone soon after a call/post to that same person."""
    df = res.unified[res.unified["entity_id"] == eid]
    comm = df[df["source"] != "bank"].dropna(subset=["timestamp", "counterparty_id"])
    bank = df[df["source"] == "bank"].dropna(subset=["timestamp", "counterparty_id"])
    if comm.empty or bank.empty:
        return False, "", [], 0
    comm_ts = pd.to_datetime(comm["timestamp"]).to_numpy()
    comm_cp = comm["counterparty_id"].to_numpy(dtype=object)
    n = 0
    trail: List[dict] = []
    for _, brow in bank.iterrows():
        bts = pd.Timestamp(brow["timestamp"])
        delta = np.abs((comm_ts - np.datetime64(bts.to_datetime64())) / np.timedelta64(1, "m"))
        hit = (comm_cp == brow["counterparty_id"]) & (delta <= tol_minutes)
        if hit.any():
            n += 1
            if n <= _MAX_PAIRS:
                crow = comm.iloc[int(np.argmax(hit))]
                trail.append(_evidence_row(res, crow))
                trail.append(_evidence_row(res, brow))
    if n >= 1:
        return True, (f"transferred money to **the same person** they had just "
                      f"contacted by call/post within {tol_minutes} minutes — "
                      f"{n} matching transfer(s)"), _dedupe_sort(trail, len(trail)), n * 2
    return False, "", [], 0


def _structuring_band(res, eid: str, threshold: float) -> Tuple[bool, str, List[dict], int]:
    """Repeated payments engineered to sit just under the reporting limit."""
    df = res.unified[res.unified["entity_id"] == eid]
    amt = df.dropna(subset=["amount"]).copy()
    amt["amount"] = pd.to_numeric(amt["amount"], errors="coerce")
    amt = amt[amt["amount"] > 0]
    if amt.empty:
        return False, "", [], 0
    band = amt[(amt["amount"] >= 0.88 * threshold) & (amt["amount"] < threshold)]
    if len(band) >= 3:
        lo, hi = int(0.88 * threshold), int(threshold)
        trail = [_evidence_row(res, r) for _, r in
                 band.sort_values("timestamp").head(_MAX_ROWS_PER_ITEM).iterrows()]
        return True, (f"kept sending amounts between {_fmt(lo)} and {_fmt(hi)} "
                      f"— just under the {_fmt(threshold)} reporting limit — "
                      f"{len(band)} times in total"), trail, len(band)
    return False, "", [], 0


def _scatter_gather(res, eid: str, scale: float = 1.0) -> Tuple[bool, str, List[dict], int]:
    """Money arrives from many people AND leaves to many people."""
    df = res.unified[res.unified["entity_id"] == eid]
    edges = df.dropna(subset=["counterparty_id"]).dropna(subset=["amount"])
    if edges.empty:
        return False, "", [], 0
    dir_col = (df["direction"].fillna("out") if "direction" in df
               else pd.Series("out", index=df.index))
    in_edges = edges[dir_col.reindex(edges.index).astype(str) == "credit"]
    out_edges = edges[dir_col.reindex(edges.index).astype(str) == "debit"]
    in_n = in_edges["counterparty_id"].nunique()
    out_n = out_edges["counterparty_id"].nunique()

    if in_n >= 5 and out_n >= 8:
        half = _MAX_ROWS_PER_ITEM // 2
        trail = _by_person(in_edges, res, half) + _by_person(out_edges, res, half)
        return True, (f"money flowed in from **{in_n} different people** and "
                      f"right back out to **{out_n} others** — a classic "
                      f"collect-then-disperse (funnel) pattern"), trail, in_n + out_n
    if in_n >= 10:
        return True, f"**received money from {in_n} different people** in the period reviewed", \
            _by_person(in_edges, res, _MAX_ROWS_PER_ITEM), in_n
    if out_n >= 10:
        return True, f"**sent money to {out_n} different people** in the period reviewed", \
            _by_person(out_edges, res, _MAX_ROWS_PER_ITEM), out_n
    return False, "", [], 0


def _burst_concentration(res, eid: str, hours: float = 48.0) -> Tuple[bool, str, List[dict], int]:
    """Many different people dealt with inside a short window."""
    sub = res.unified[res.unified["entity_id"] == eid].dropna(subset=["timestamp", "counterparty_id"])
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    if len(sub) < 10:
        return False, "", [], 0
    ts = pd.to_datetime(sub["timestamp"]).to_numpy(dtype="datetime64[ns]")
    cp = sub["counterparty_id"].to_numpy(dtype=object)
    window = np.timedelta64(int(hours * 3600), "s")
    best_n, best_i = -1, 0
    for i in range(len(ts)):
        mask = (ts >= ts[i] - window) & (ts <= ts[i] + window)
        n_distinct = len(set(cp[mask]))
        if n_distinct > best_n:
            best_n, best_i = n_distinct, i
    if best_n >= 12:
        mask = (ts >= ts[best_i] - window) & (ts <= ts[best_i] + window)
        trail = _by_person(sub.loc[mask], res, _MAX_ROWS_PER_ITEM)
        return True, (f"dealt with **{best_n} different people inside a single "
                      f"{int(hours)}-hour window** — unusually compressed activity"), trail, best_n
    return False, "", [], 0


def _dormancy(res, eid: str, quiet_days: float = 30.0) -> Tuple[bool, str, List[dict], int]:
    """Long silence then a sudden burst (activation after dormancy)."""
    sub = res.unified[res.unified["entity_id"] == eid].dropna(subset=["timestamp"])
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    if len(sub) < 10:
        return False, "", [], 0
    ts = pd.to_datetime(sub["timestamp"])
    gaps = ts.diff().dt.total_seconds().dropna() / 86400.0
    if gaps.empty:
        return False, "", [], 0
    gmax = float(gaps.max())
    if gmax >= quiet_days:
        idx = gaps.idxmax()
        trail = [_evidence_row(res, sub.iloc[idx - 1]), _evidence_row(res, sub.iloc[idx])]
        return True, (f"went completely quiet for **{gmax:,.0f} days** then "
                      f"re-activated all at once"), trail, 2
    return False, "", [], 0


def _big_lump(res, eid: str, q: float = 0.95, min_days_after: float = 14.0) -> Tuple[bool, str, List[dict], int]:
    """One very large transfer followed by silence (laying low)."""
    sub = res.unified[res.unified["entity_id"] == eid].dropna(subset=["amount", "timestamp"])
    if len(sub) < 5:
        return False, "", [], 0
    amt = pd.to_numeric(sub["amount"], errors="coerce")
    qv = amt.quantile(q)
    if qv <= 0:
        return False, "", [], 0
    big = sub[amt >= qv]
    if big.empty:
        return False, "", [], 0
    last_ts = pd.to_datetime(sub["timestamp"]).max()
    big_ts = pd.to_datetime(big["timestamp"]).max()
    days_after = (last_ts - big_ts).total_seconds() / 86400.0
    if days_after >= min_days_after and (big["timestamp"] == big_ts).any():
        row = big.loc[big["timestamp"] == big_ts].iloc[0]
        b_amount = float(row["amount"])
        trail = [_evidence_row(res, row)]
        return True, (f"moved a **large single amount ({_fmt(b_amount)})** and "
                      f"then went quiet for {days_after:,.0f} days"), trail, 1
    return False, "", [], 0


def _odd_amount_shape(res, eid: str, min_rows: int = 15) -> Tuple[bool, str, List[dict], int]:
    """Payment amounts that look slightly 'too round' (no natural cents)."""
    sub = res.unified[res.unified["entity_id"] == eid].dropna(subset=["amount"])
    if len(sub) < min_rows:
        return False, "", [], 0
    vals = pd.to_numeric(sub["amount"], errors="coerce")
    valid = vals.notna()
    sub, vals = sub[valid], vals[valid]
    if len(vals) < min_rows:
        return False, "", [], 0
    no_cents_mask = ((vals % 1).abs() < 0.01)
    no_cents = float(no_cents_mask.mean())
    if no_cents >= 0.5:
        matches = sub[no_cents_mask]
        total = int(no_cents_mask.sum())
        trail = [_evidence_row(res, r) for _, r in
                 matches.sort_values("timestamp").head(_MAX_ROWS_PER_ITEM).iterrows()]
        return True, (f"more than half of the amounts ({no_cents:.0%}) are bare "
                      f"**round numbers with no cents** — real bills rarely look "
                      f"this clean"), trail, total
    return False, "", [], 0


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

    Each entry: {rank, entity_id, name, risk, headline, evidence[], evidence_items[]}.
    ``evidence`` stays a flat list of sentence strings (unchanged shape).
    ``evidence_items`` is the same sentences paired 1:1 with their own
    trail: [{sentence, trail, trail_total}] — the specific records behind
    that specific claim (e.g. the 21 counterparties behind "received money
    from 21 different people"), not one pile shared across every claim.
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
        items: List[dict] = []
        for fn in extractors:
            try:
                hit, sentence, trail, total = fn(res, eid)
            except Exception:
                continue
            if hit:
                items.append({"sentence": sentence, "trail": trail, "trail_total": total})
        if not items:
            items.append({
                "sentence": "activity profile differs noticeably from the rest of "
                            "the people in this dataset",
                "trail": [], "trail_total": 0,
            })
        out.append({
            "rank": i,
            "entity_id": eid,
            "name": display_name(res, eid),
            "risk": _risk_level(r["score"]),
            "headline": items[0]["sentence"],
            "evidence": [it["sentence"] for it in items],
            "evidence_items": items,
        })
    return out
