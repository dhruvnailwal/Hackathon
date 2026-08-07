"""Model families (§2.3-2.6) over the unified dataframe.

Every score function returns {entity_id: float} (higher = more anomalous).
  C time_correlation : merge_asof cross-source colocation timing
  D network          : OddBall egonet density/degree deviation (default);
                        ``burst`` keeps the short-window concentration model
  E statml           : Isolation Forest on a shared per-entity feature vector
                        (``eif``/``lof`` variants live in the model zoo)
  H benford          : first-digit chi-square deviation on amounts
  H structuring      : threshold-proximity structuring rule
  F fusion           : Borda rank fusion -> ranked insights + explanations (G)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as _st
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

EPS = 1e-9


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + EPS)


# ===========================================================================
# shared per-entity feature vector (§2.5) for Stage E
# ===========================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "entity_id" not in df.columns or df["entity_id"].dropna().empty:
        return pd.DataFrame()

    ids = sorted(set(df["entity_id"].dropna().astype(str)))
    rows = df[[c for c in df.columns if c != "entity_id"]].copy()
    rows["entity_id"] = df["entity_id"].astype(str)
    E = pd.DataFrame(index=ids)
    E.index.name = "entity_id"

    g = rows.groupby("entity_id")
    E["n_events"] = g.size().reindex(E.index).fillna(1)

    for et in ("call", "transaction", "post", "credit", "debit"):
        c = rows[rows["event_type"] == et].groupby("entity_id").size()
        E[et] = c.reindex(E.index).fillna(0)

    amt = rows[pd.to_numeric(rows.get("amount"), errors="coerce") > 0] if "amount" in rows else rows.iloc[0:0]
    if not amt.empty:
        ag = amt.groupby("entity_id")["amount"]
        E["amt_total"] = ag.sum().reindex(E.index).fillna(0)
        E["amt_mean"] = ag.mean().reindex(E.index).fillna(0)
        E["amt_max"] = ag.max().reindex(E.index).fillna(0)
        E["amt_std"] = ag.std().reindex(E.index).fillna(0)
    else:
        for c_ in ("amt_total", "amt_mean", "amt_max", "amt_std"):
            E[c_] = 0.0

    cps = rows.dropna(subset=["counterparty_id"])
    dir_col = rows["direction"].fillna("out") if "direction" in rows else pd.Series("out", index=rows.index)
    if not cps.empty:
        fan_out = cps[dir_col.reindex(cps.index).astype(str) != "credit"].groupby("entity_id")["counterparty_id"].nunique()
        fan_in = cps[dir_col.reindex(cps.index).astype(str) == "credit"].groupby("entity_id")["counterparty_id"].nunique()
        E["fan_out"] = fan_out.reindex(E.index).fillna(0)
        E["fan_in"] = fan_in.reindex(E.index).fillna(0)
    else:
        E["fan_out"] = 0.0
        E["fan_in"] = 0.0

    # temporal features
    ts = rows.dropna(subset=["timestamp"]).copy()
    if not ts.empty:
        ts["timestamp"] = pd.to_datetime(ts["timestamp"])
        ts = ts.sort_values("timestamp")
        ts["hour"] = ts["timestamp"].dt.hour
        ts["dow"] = ts["timestamp"].dt.dayofweek

        ts["interval"] = ts.groupby("entity_id")["timestamp"].diff().dt.total_seconds()
        intv = ts.dropna(subset=["interval"])
        i = intv.groupby("entity_id")["interval"]
        E["gap_max"] = i.max().reindex(E.index).fillna(0)
        E["gap_min"] = i.min().reindex(E.index).fillna(0)
        E["gap_std"] = i.std().reindex(E.index).fillna(0)

        hour_hist = ts.groupby(["entity_id", ts["hour"]]).size().unstack(fill_value=0)
        prob = hour_hist.div(hour_hist.sum(axis=1), axis=0).reindex(E.index).fillna(0)
        E["hour_entropy"] = -(prob * np.log(prob + EPS)).sum(axis=1)

        is_night = ((ts["hour"] >= 22) | (ts["hour"] <= 5))
        E["night_ratio"] = is_night.groupby(ts["entity_id"]).mean().reindex(E.index).fillna(0)
        is_weekend = ts["dow"] >= 5
        E["weekend_ratio"] = is_weekend.groupby(ts["entity_id"]).mean().reindex(E.index).fillna(0)
    else:
        for c_ in ("gap_max", "gap_min", "gap_std", "hour_entropy", "night_ratio", "weekend_ratio"):
            E[c_] = 0.0
    return E


# ===========================================================================
# C — time correlation (merge_asof colocation)
# ===========================================================================
def fit_time_correlation(df: pd.DataFrame, tolerance_minutes: int = 25,
                         min_episodes: int = 3) -> Dict[str, float]:
    """merge_asof colocation: a person calls/email then transfers within tolerance.
    Score ramps on sincere SAME-party episodes (plan demo "call 09:12, transfer 09:24")."""
    bank = df[df["source"] == "bank"].copy()
    cdr = df[df["source"] != "bank"].copy()
    if bank.empty or cdr.empty:
        return {}
    bank = bank.dropna(subset=["timestamp", "entity_id", "counterparty_id"])
    cdr = cdr.dropna(subset=["timestamp", "entity_id", "counterparty_id"])
    bank["ts"] = pd.to_datetime(bank["timestamp"])
    cdr["ts"] = pd.to_datetime(cdr["timestamp"])
    bank = bank.sort_values("ts")
    cdr = cdr.sort_values("ts")
    matched = pd.merge_asof(
        cdr, bank, on="ts", direction="forward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes), suffixes=("_x", "_y"),
    )
    matched = matched[matched["entity_id_y"].notna()]
    if matched.empty:
        return {}
    same = (matched["counterparty_id_x"] == matched["counterparty_id_y"]).to_numpy()
    out = {}
    for eid in set(matched["entity_id_x"]):
        s = int(same[matched["entity_id_x"].to_numpy() == eid].sum())
        t = int((~same)[matched["entity_id_x"].to_numpy() == eid].sum())
        if s >= min_episodes:
            out[eid] = min(s / float(min_episodes), 1.0)
        elif t > 0:
            out[eid] = 0.2 * min(t / 3.0, 1.0) + (0.3 * s / float(min_episodes))
    return out


# ===========================================================================
# D — network (short-window burst concentration — "new dense node" signal)
# ===========================================================================
def _burst_concentration(ts: np.ndarray, cp: np.ndarray, hours: float = 36.0) -> int:
    """Max distinct counterparties observed within any `hours` window."""
    ts = ts.astype("int64")
    window = int(hours * 3600.0 * 1e9)
    order = np.argsort(ts)
    ts, cp = ts[order], cp[order]
    n = len(ts)
    best = 0
    for i in range(n):
        seen = {cp[i]}
        j = i - 1
        while j >= 0 and ts[i] - ts[j] <= window:
            seen.add(cp[j])
            j -= 1
        j = i + 1
        while j < n and ts[j] - ts[i] <= window:
            seen.add(cp[j])
            j += 1
        best = max(best, len(seen))
    return best


def fit_network(df: pd.DataFrame) -> Dict[str, float]:
    edges = df.dropna(subset=["entity_id", "counterparty_id", "timestamp"])
    if edges.empty:
        return {}
    bursts = {}
    for eid, sub in edges.groupby("entity_id"):
        bursts[eid] = _burst_concentration(
            sub["timestamp"].to_numpy(dtype="int64"),
            sub["counterparty_id"].to_numpy(dtype=object),
        )
    burst_s = pd.Series(bursts)
    if burst_s.empty:
        return {}
    # sparse-gate: only the top ~15% of the population climbs toward 1.0
    lo = float(burst_s.quantile(0.70))
    hi = float(burst_s.quantile(0.98))
    span = (hi - lo) or 1.0
    out = {}
    for eid, b in burst_s.items():
        s = (b - lo) / span
        out[eid] = min(max(s, 0.0), 1.0)
    return out


# ===========================================================================
# E — stat / ML (Isolation Forest on population-relative features)
# ===========================================================================
_EVID_FEATURES = ["n_events", "amt_total", "amt_mean", "amt_max", "amt_std",
                  "fan_out", "fan_in", "gap_max", "gap_min", "gap_std"]


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Shared Stage-E feature matrix, preprocessed exactly like fit_statml."""
    feats = build_features(df)
    if feats.empty or feats.shape[0] < 5:
        return pd.DataFrame()
    X = feats[[c for c in feats.columns if c != "entity_id"]].copy()
    for c in _EVID_FEATURES:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    X = X.astype(float).apply(lambda s: (s - s.mean()) / (s.std() + EPS))
    X = np.nan_to_num(X.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(X, index=feats.index)


def fit_statml(df: pd.DataFrame, contamination: float = 0.06,
               n_estimators: int = 400) -> Dict[str, float]:
    X = _feature_matrix(df)
    if X.empty:
        return {}
    clf = IsolationForest(contamination=contamination, random_state=0,
                          n_estimators=n_estimators)
    clf.fit(X)
    raw = -clf.score_samples(X)
    return _scoremin(raw, X.index)


# shared piece of the score pipeline
def _scoremin(raw: np.ndarray, index) -> Dict[str, float]:
    s = pd.Series(raw, index=index)
    s = _minmax(s)
    s = np.power(s, 2.0)
    return s.to_dict()


_EIF_GAMMA = 0.5772156649015329


def _eif_path_lengths(X: np.ndarray, n_trees: int = 200, max_depth: int = 14,
                      random_state: int = 0) -> np.ndarray:
    """Extended Isolation Forest (Hariri et al. 2019).

    Axis-parallel cuts of plain IF are replaced by random hyperplanes
    (uniform random unit slopes), so oblique structures that axis cuts can
    never separate are isolated too. Depths are averaged over trees.
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    sample = min(n, 256)
    depths = np.zeros(n)
    for _ in range(n_trees):
        cols = rng.choice(n, size=sample, replace=False)
        pts = X[cols]
        stack = [(np.arange(sample), 0)]
        while stack:
            inds, depth = stack.pop()
            k = len(inds)
            if k <= 1:
                continue
            if depth >= max_depth:
                depths[cols[inds]] += _avg_path_constant(k)
                continue
            v = rng.normal(size=d)
            norm = float(np.linalg.norm(v))
            if norm < EPS:
                depths[cols[inds]] += _avg_path_constant(k)
                continue
            proj = pts[inds] @ (v / norm)
            lo, hi = float(proj.min()), float(proj.max())
            if hi - lo < EPS:
                # identical points project identically under every slope:
                # the node is a leaf, pay the expected residual path c(k)
                depths[cols[inds]] += _avg_path_constant(k)
                continue
            cut = rng.uniform(lo, hi)
            le = inds[proj <= cut]
            gt = inds[proj > cut]
            if len(le) == 0 or len(gt) == 0:
                depths[cols[inds]] += _avg_path_constant(k)
                continue
            depths[cols[inds]] += 1
            stack.append((le, depth + 1))
            stack.append((gt, depth + 1))
    return depths


def _avg_path_constant(n: int) -> float:
    if n <= 1:
        return 1.0
    return 2.0 * np.log(n - 1) + 2.0 * _EIF_GAMMA - 2.0 * (n - 1) / n


def fit_statml_eif(df: pd.DataFrame, n_trees: int = 200,
                   max_depth: int = 14) -> Dict[str, float]:
    """Stage-E candidate: Extended Isolation Forest variant."""
    X = _feature_matrix(df)
    if X.empty:
        return {}
    depths = _eif_path_lengths(X.to_numpy(), n_trees=n_trees, max_depth=max_depth)
    mean_c = _avg_path_constant(len(X))
    scores = np.power(2.0, -(depths / n_trees) / mean_c)
    return _scoremin(scores, X.index)


def fit_statml_lof(df: pd.DataFrame, n_neighbors: int = 20,
                   contamination: float = 0.06) -> Dict[str, float]:
    """Stage-E candidate: Local Outlier Factor on the shared feature matrix."""
    X = _feature_matrix(df)
    if X.empty:
        return {}
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    clf.fit(X)
    raw = -clf.negative_outlier_factor_  # larger = more isolated
    return _scoremin(raw, X.index)


# ===========================================================================
# D — network candidates (OddBall-style density/degree deviation)
# ===========================================================================
def fit_oddball(df: pd.DataFrame) -> Dict[str, float]:
    """OddBall null model (Akoglu et al. 2010) applied to egonets.

    For every entity, let d = number of distinct counterparties (degree) and
    w = number of transactions (edge count). OddBall fits w ~ d^{k} and flags
    entities whose mass deviates from the expected density line — tiny
    counterparty bases carrying large transaction volume (a classic money
    mule / fan-out signature).
    """
    edges = df.dropna(subset=["entity_id", "counterparty_id"])
    if edges.empty:
        return {}
    deg = edges.groupby("entity_id")["counterparty_id"].nunique()
    wgt = edges.groupby("entity_id").size()
    ok = (deg > 0)
    if not ok.any() or ok.sum() < 8:
        return {}
    x = np.log(deg[ok].to_numpy(float))
    y = np.log1p(wgt[ok].to_numpy(float))
    slope, intercept, _, _, _ = _st.linregress(x, y)
    resid = y - (slope * x + intercept)
    resid_s = resid / (resid.std() or 1.0)
    # absolute deviation captures both "densely stuffed" and "light-weight
    # hub" endpoints
    raw = pd.Series(np.abs(resid_s), index=deg[ok].index)
    return _scoremin(raw.to_numpy(), raw.index)


# ===========================================================================
# H — Benford first-digit deviation (POPULATION-relative, p-value calibrated)
# ===========================================================================
def fit_benford(df):
    amt = df.dropna(subset=["amount"])
    amt = amt[amt["amount"] > 0]
    if amt.empty:
        return None
    first_all = np.floor(amt["amount"].to_numpy(float)
                         / np.power(10.0, np.floor(np.log10(amt["amount"].to_numpy(float))))).astype(int)
    mask = (first_all >= 1) & (first_all <= 9)
    global_hist = np.bincount(first_all[mask] - 1, minlength=9).astype(float)
    if global_hist.sum() == 0:
        return None
    exp = global_hist / global_hist.sum()
    out = {}
    for eid, sub in amt.groupby("entity_id"):
        if len(sub) < 8:
            continue
        vals = sub["amount"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            first = np.floor(vals / np.power(10.0, np.floor(np.log10(vals)))).astype(int)
        obs = np.zeros(9)
        for f in first:
            if 1 <= f <= 9:
                obs[f - 1] += 1
        if obs.sum() < 8:
            continue
        obs = obs / obs.sum()
        chi = float(((obs - exp) ** 2 / (exp + EPS)).sum())
        p = float(_st.chi2.sf(chi, df=8))
        out[eid] = min(1.0 - p, 1.0)
    if not out:
        return None
    return out


# ===========================================================================
# H — structuring rule
# ===========================================================================
def fit_structuring(df: pd.DataFrame, threshold: float = 10000.0, band: float = 0.88) -> Dict[str, float]:
    amt = df.dropna(subset=["amount"])
    if amt.empty:
        return None
    band_rows = amt[(amt["amount"] >= band * threshold) & (amt["amount"] < threshold)]
    if band_rows.empty:
        return None
    counts = band_rows.groupby("entity_id").size()
    totals = amt.groupby("entity_id").size().reindex(counts.index).fillna(1)
    score = (counts ** 1.5) / np.sqrt(totals)
    mx = score.max() or 1.0
    return {e: float(v / mx) for e, v in score.items()}


# ===========================================================================
# H — behavioral regime flips: dormancy + silence-after-large-txn
# ===========================================================================
def fit_behavioral(df: pd.DataFrame, dormant_gap_days: float = 45.0,
                   silence_amt_q: float = 0.995, silence_days: float = 15.0) -> Dict[str, float]:
    amt_all = df.dropna(subset=["amount"])["amount"]
    big_threshold = float(amt_all.quantile(silence_amt_q)) if not amt_all.empty else 1e12
    out = {}
    for eid, sub in df.groupby("entity_id"):
        b = sub.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        n = len(b)
        if n < 3:
            continue
        score = 0.0
        # dormancy: long quiet stretch in the middle of an otherwise active life
        if n >= 15:
            gaps = b["timestamp"].diff().dt.total_seconds().dropna() / 86400.0
            if not gaps.empty and gaps.max() >= dormant_gap_days:
                pos = int(gaps.argmax())  # event index BEFORE the long gap
                if pos >= 2 and pos < n - 2:
                    score = max(score, min(0.4 + gaps.max() / 90.0, 1.0))
        # silence-after-large-txn: absolute population-level big amount then quiet
        amt = b.dropna(subset=["amount"])
        if not amt.empty:
            big = amt[amt["amount"] >= big_threshold]
            if not big.empty:
                last_big = big["timestamp"].max()
                days_after = (b["timestamp"].iloc[-1] - last_big).total_seconds() / 86400.0
                if days_after >= silence_days:
                    score = max(score, min(0.5 + days_after / 40.0, 1.0))
        if score > 0.0:
            out[eid] = score
    return out


# ===========================================================================
# F — fusion ranking (equal-weight normalized sum) + G explanations
# ===========================================================================
class Fusion:
    def __init__(self, normalized_scores: Dict[str, Dict[str, float]], df: pd.DataFrame):
        self.scores = normalized_scores
        self.df = df
        self.rank: List[dict] = []
        self.explanation: Dict[str, str] = {}
        self._fuse()

    def _finalize(self, agg: Dict[str, float]) -> None:
        total_events = self.df.groupby("entity_id").size().to_dict() if "entity_id" in self.df else {}
        # deterministic tie-break: (score desc, entity_id asc) — set iteration
        # order is hash-seed dependent and would otherwise leak into the rank
        sorted_agg = sorted(agg.items(), key=lambda kv: (-kv[1], str(kv[0])))
        self.rank = [
            {
                "entity_id": eid,
                "score": round(sc, 4),
                "n_events": int(total_events.get(eid, 0)),
                "models_fired": [m for m, s in self.scores.items() if s.get(eid, 0.0) > 0.6],
            }
            for eid, sc in sorted_agg
        ]
        self.explanation = {
            eid: self._narrate(eid, sc)
            for eid, sc in agg.items()
            if sc > 0.6
        }

    def _fuse(self) -> None:
        all_ids = set()
        for s in self.scores.values():
            all_ids.update(s.keys())
        # top-2 weighted fusion: a strong catch in ONE specialized model must
        # surface, while agreement across models adds confidence
        agg: Dict[str, float] = {}
        for eid in all_ids:
            vals = sorted((s.get(eid, 0.0) for s in self.scores.values()), reverse=True)
            if vals:
                top1, top2 = vals[0], (vals[1] if len(vals) > 1 else 0.0)
                agg[eid] = float(0.7 * top1 + 0.3 * top2)
        self._finalize(agg)

    def _narrate(self, eid: str, score: float) -> str:
        fired = [m for m, s in self.scores.items() if s.get(eid, 0.0) > 0.5]
        parts = [f"Fusion score {score:.2f}"]
        if "time_correlation" in fired:
            parts.append("time model: cross-source call/post landed shortly before a transfer")
        if "network" in fired:
            parts.append("network model: unusual hub/density/role shift")
        if "statml" in fired:
            parts.append("stat model: feature-space outlier vs peer population")
        if "benford" in fired:
            parts.append("benford model: amount first-digit distribution deviates")
        if "structuring" in fired:
            parts.append("structuring model: repeated amounts clustered just under the reporting threshold")
        if "behavioral" in fired:
            parts.append("behavioral model: dormancy duration or silence-after-large-transfer regime flip")
        if not parts[1:]:
            parts.append("aggregate marginal elevation across models")
        return ". ".join(parts)


def fuse_model_scores(scores: Dict[str, Dict[str, float]], df: pd.DataFrame) -> Fusion:
    return Fusion(scores, df)


# ===========================================================================
# F — fusion candidate: Borda-count rank fusion (scale-robust aggregation)
# ===========================================================================
class FusionBorda(Fusion):
    """Alternative to the top-2 weighted Fusion.

    Each model casts a rank vote: the entity with the highest score in a
    model gets (m-1) points, the lowest 0, entities the model never scored
    get 0. Points are summed across models and normalized by the number of
    models — so only cross-model *agreement* elevates an entity, at the cost
    of single-model strong signals (tunable by including the max vote).
    """

    def _fuse(self) -> None:
        models = list(self.scores.values())
        n_models = max(len(models), 1)
        points: Dict[str, float] = {}
        for s in models:
            if not s:
                continue
            m = len(s)
            if m <= 1:
                for eid in s:
                    points[eid] = points.get(eid, 0.0)
                continue
            order = sorted(s, key=lambda e: (s[e], e))
            rank = {e: i / (m - 1) for i, e in enumerate(order)}
            for eid, r in rank.items():
                points[eid] = points.get(eid, 0.0) + r
        agg = {eid: min(p / n_models, 1.0) for eid, p in points.items()}
        self._finalize(agg)


def fuse_rank_borda(scores: Dict[str, Dict[str, float]], df: pd.DataFrame) -> FusionBorda:
    return FusionBorda(scores, df)


# ===========================================================================
# dispatch used by the pipeline
# ===========================================================================
def run_model(name: str, df: pd.DataFrame, tolerance_minutes: int = 25,
              structuring_threshold: float = 10000.0) -> Dict[str, float]:
    if name == "time_correlation":
        return fit_time_correlation(df, tolerance_minutes)
    if name == "network":
        return fit_oddball(df)
    if name == "burst":
        return fit_network(df)
    if name == "statml":
        return fit_statml(df)
    if name == "statml_eif":
        return fit_statml_eif(df)
    if name == "statml_lof":
        return fit_statml_lof(df)
    if name == "benford":
        return fit_benford(df)
    if name == "structuring":
        return fit_structuring(df, threshold=structuring_threshold)
    if name == "behavioral":
        return fit_behavioral(df)
    return {}