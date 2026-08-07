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
from scipy.spatial.distance import mahalanobis as _sp_mahalanobis
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
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


def fit_time_correlation_backward(df: pd.DataFrame, tolerance_minutes: int = 25,
                                  min_episodes: int = 3) -> Dict[str, float]:
    """Reverse colocation: transfer lands, then a call to the same party.
    Picks up 'call after transfer' laundering choreography."""
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
        bank, cdr, on="ts", direction="forward",
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


def fit_time_correlation_anypair(df: pd.DataFrame, tolerance_minutes: int = 25,
                                 min_episodes: int = 3) -> Dict[str, float]:
    """Same-party co-occurrence window: any call to X within tol of any transfer
    to X (no asof pairing — count of same-party overlaps per entity)."""
    bank = df[df["source"] == "bank"].dropna(subset=["timestamp", "entity_id", "counterparty_id"]).copy()
    cdr = df[df["source"] != "bank"].dropna(subset=["timestamp", "entity_id", "counterparty_id"]).copy()
    if bank.empty or cdr.empty:
        return {}
    bank["ts"] = pd.to_datetime(bank["timestamp"]).to_numpy()
    cdr["ts"] = pd.to_datetime(cdr["timestamp"]).to_numpy()
    tol = pd.Timedelta(minutes=tolerance_minutes)
    out = {}
    for eid, sub in bank.groupby("entity_id"):
        calls = cdr[cdr["entity_id"] == eid]
        if calls.empty:
            continue
        same = 0
        for b_row in sub.itertuples():
            for c_row in calls.itertuples():
                if c_row.counterparty_id == b_row.counterparty_id and abs(b_row.ts - c_row.ts) <= tol:
                    same += 1
        if same >= min_episodes:
            out[eid] = min(same / float(min_episodes), 1.0)
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


def fit_statml_mahalanobis(df: pd.DataFrame) -> Dict[str, float]:
    """Stage-E candidate: robust Mahalanobis distance.

    MinCovDet (Rousseeuw) estimates a covariance that resists contamination;
    entities far from the population centre in this metric are structural
    outliers even when every marginal looks normal.
    """
    X = _feature_matrix(df)
    if X.empty:
        return {}
    arr = X.to_numpy()
    try:
        mcd = MinCovDet(store_precision=False, random_state=0).fit(arr)
    except ValueError:
        # too few samples / non-invertible; fall back to plain limiting distance
        try:
            cov = np.cov(arr, rowvar=False)
            inv = np.linalg.pinv(cov + EPS * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            return {}
        mean = arr.mean(axis=0)
        d = np.array([_sp_mahalanobis(row, mean, inv) for row in arr])
        return _scoremin(d, X.index)
    d = np.atleast_1d(mcd.mahalanobis(arr))
    return _scoremin(d, X.index)


def fit_statml_pca(df: pd.DataFrame, n_components: int = 4) -> Dict[str, float]:
    """Stage-E candidate: PCA reconstruction error.

    Projects the feature matrix onto a few dominant components and measures
    reconstruction error — points that need unusual coordinates (isolated in
    residual space) are anomalous.
    """
    X = _feature_matrix(df)
    if X.empty:
        return {}
    arr = X.to_numpy()
    k = min(n_components, arr.shape[1] - 1) if arr.shape[1] > 1 else 1
    if k < 1:
        return {}
    pca = PCA(n_components=k, random_state=0).fit(arr)
    recon = pca.inverse_transform(pca.transform(arr))
    err = np.linalg.norm(arr - recon, axis=1)
    return _scoremin(err, X.index)


def fit_statml_zscore(df: pd.DataFrame) -> Dict[str, float]:
    """Stage-E candidate: polarising z-score (max absolute std across features).

    For each entity take its most extreme unit-normalized feature — a cheap,
    interpretable surrogate that the forests often beat only on population
    shape.
    """
    X = _feature_matrix(df)
    if X.empty:
        return {}
    arr = X.to_numpy()
    z = np.abs(arr).max(axis=1)
    return _scoremin(z, X.index)


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


def fit_oddball_signed(df: pd.DataFrame) -> Dict[str, float]:
    """Signed OddBall residual: only entities ABOVE the density line (their
    graph carries far more weight than degree predicts) are anomalous, not the
    low-maintenance hubs. No abs() — flags mass-stuffed money mules."""
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
    raw = pd.Series(resid, index=deg[ok].index)
    return _scoremin(raw.to_numpy(), raw.index)


def fit_degree_deviation(df: pd.DataFrame) -> Dict[str, float]:
    """Degree z-score: entities whose degree (number of distinct counterparties)
    is far from the population mean — a super-connected hub."""
    edges = df.dropna(subset=["entity_id", "counterparty_id"])
    if edges.empty:
        return {}
    deg = edges.groupby("entity_id")["counterparty_id"].nunique()
    if deg.nunique() < 3:
        return {}
    z = (deg - deg.mean()) / (deg.std() + EPS)
    raw = pd.Series(np.abs(z), index=deg.index)
    return _scoremin(raw.to_numpy(), raw.index)


def fit_reciprocity(df: pd.DataFrame) -> Dict[str, float]:
    """Directed-edge reciprocity deviation.

    Entities that BOTH fan-out to many parties AND are fanned-in from many
    parties create dense two-way clusters = classic launderer hub that moves
    money in and out of a narrow wallet set. Score = product of in/out degree
    normalized against population.
    """
    edges = df.dropna(subset=["entity_id", "counterparty_id"])
    if edges.empty:
        return {}
    dir_col = df["direction"].fillna("out") if "direction" in df else pd.Series("out", index=df.index)
    out_n = edges[dir_col.reindex(edges.index).astype(str) != "credit"].groupby("entity_id")["counterparty_id"].nunique()
    in_n = edges[dir_col.reindex(edges.index).astype(str) == "credit"].groupby("entity_id")["counterparty_id"].nunique()
    both = out_n.reindex(in_n.index.union(out_n.index)).fillna(0) * in_n.reindex(in_n.index.union(out_n.index)).fillna(0)
    if both.empty or both.max() == 0:
        return {}
    log_both = np.log1p(both)
    out = {}
    for eid, v in log_both.items():
        out[eid] = min(v / float(log_both.max()), 1.0)
    return out


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


def fit_benford_ks(df):
    """Benford alternate: population-relative Kolmogorov–Smirnov style.

    Builds the population first-digit distribution (as expected frequencies),
    then reports each entity's max-|obs−exp| "KS statistic" as the deviation.
    Complements the chi-square variant (which over-flags thin samples).
    """
    amt = df.dropna(subset=["amount"])
    amt = amt[amt["amount"] > 0]
    if amt.empty:
        return None
    first_all = np.floor(amt["amount"].to_numpy(float)
                         / np.power(10.0, np.floor(np.log10(amt["amount"].to_numpy(float))))).astype(int)
    mask = (first_all >= 1) & (first_all <= 9)
    if mask.sum() == 0:
        return None
    global_hist = np.bincount(first_all[mask] - 1, minlength=9).astype(float)
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
        ks = float(np.max(np.abs(np.cumsum(obs) - np.cumsum(exp))))
        out[eid] = min(4.0 * ks, 1.0)
    if not out:
        return None
    return out


def fit_structuring_banded(df: pd.DataFrame, threshold: float = 10000.0,
                           bands: int = 3) -> Dict[str, float]:
    """Structuring variant: near-threshold WITHOUT the ÷√total dampening.

    Counts only the band rows and rewards *quantity of band events* directly
    (counts^1.2), so persistent small-cutters rank high without the total-event
    penalty that the baseline applies."""
    amt = df.dropna(subset=["amount"])
    if amt.empty:
        return None
    band_rows = amt[(amt["amount"] >= 0.8 * threshold) & (amt["amount"] < threshold)]
    if band_rows.empty:
        return None
    counts = band_rows.groupby("entity_id").size()
    score = counts ** 1.2
    mx = score.max() or 1.0
    return {e: float(v / mx) for e, v in score.items()}


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
    models — so only cross-model *agreement* elevates an entity.

    A ``max_bonus`` term re-weights single-model strong signals (e.g. a
    colocation entity with time_correlation=1.0) that pure rank-averaging
    buries: points = borda + max_bonus * best_model_score. Keeps agreement
    dominant but never lets a perfect single-model alarm vanish from the top.
    """

    def __init__(self, normalized_scores, df, max_bonus: float = 0.25):
        self.max_bonus = max_bonus
        super().__init__(normalized_scores, df)

    def _fuse(self) -> None:
        models = list(self.scores.values())
        n_models = max(len(models), 1)
        points: Dict[str, float] = {}
        best: Dict[str, float] = {}
        for s in models:
            if not s:
                continue
            m = len(s)
            if m <= 1:
                for eid in s:
                    points[eid] = points.get(eid, 0.0)
                    best[eid] = max(best.get(eid, 0.0), s[eid])
                continue
            order = sorted(s, key=lambda e: (s[e], e))
            rank = {e: i / (m - 1) for i, e in enumerate(order)}
            for eid, r in rank.items():
                points[eid] = points.get(eid, 0.0) + r
                best[eid] = max(best.get(eid, 0.0), s[eid])
        agg = {
            eid: min(p / n_models + self.max_bonus * best.get(eid, 0.0), 1.0)
            for eid, p in points.items()
        }
        self._finalize(agg)


def fuse_rank_borda(scores: Dict[str, Dict[str, float]], df: pd.DataFrame,
                    max_bonus: float = 0.25) -> FusionBorda:
    return FusionBorda(scores, df, max_bonus=max_bonus)


class FusionRankAverage(Fusion):
    """Rank-average fusion (no bonus): each model's rank is averaged over the
    number of models that scored the entity. Pure agreement — a single-model
    alarm cannot surface on its own."""

    def _fuse(self) -> None:
        models = list(self.scores.values())
        n_models = max(len([s for s in models if s]), 1)
        points: Dict[str, float] = {}
        seen: Dict[str, int] = {}
        for s in models:
            if not s:
                continue
            m = len(s)
            if m <= 1:
                for eid in s:
                    points[eid] = points.get(eid, 0.0)
                    seen[eid] = seen.get(eid, 0) + 1
                continue
            order = sorted(s, key=lambda e: (s[e], e))
            rank = {e: i / (m - 1) for i, e in enumerate(order)}
            for eid, r in rank.items():
                points[eid] = points.get(eid, 0.0) + r
                seen[eid] = seen.get(eid, 0) + 1
        agg = {eid: p / max(seen.get(eid, 1), 1) for eid, p in points.items()}
        self._finalize(agg)


class FusionScoreMean(Fusion):
    """Score-mean fusion: arithmetic mean of normalized model scores. Weights
    every model equally; models that never score an entity contribute 0."""

    def _fuse(self) -> None:
        models = [s for s in self.scores.values() if s]
        if not models:
            return self._finalize({})
        ids = set()
        for s in models:
            ids.update(s.keys())
        agg = {}
        for eid in ids:
            vals = [s.get(eid, 0.0) for s in models]
            agg[eid] = float(np.mean(vals))
        self._finalize(agg)


class FusionWeightedSum(Fusion):
    """Weighted-sum fusion: each model's score is weighted by a family weight
    (default uniform 1.0) before summing — keeps score magnitudes, unlike the
    rank fusions."""

    def __init__(self, normalized_scores, df, weights: Dict[str, float] | None = None):
        self.weights = weights or {}
        super().__init__(normalized_scores, df)

    def _fuse(self) -> None:
        ids = set()
        for s in self.scores.values():
            ids.update(s.keys())
        agg = {}
        for eid in ids:
            total = 0.0
            w_total = 0.0
            for m, s in self.scores.items():
                if not s:
                    continue
                total += self.weights.get(m, 1.0) * s.get(eid, 0.0)
                if eid in s:
                    w_total += self.weights.get(m, 1.0)
            agg[eid] = total / max(w_total, 1.0)
        self._finalize(agg)


def fuse_rank_average(scores, df):
    return FusionRankAverage(scores, df)


def fuse_score_mean(scores, df):
    return FusionScoreMean(scores, df)


def fuse_weighted_sum(scores, df, weights=None):
    return FusionWeightedSum(scores, df, weights=weights)


# ===========================================================================
# dispatch used by the pipeline
# ===========================================================================
def run_model(name: str, df: pd.DataFrame, tolerance_minutes: int = 25,
              structuring_threshold: float = 10000.0) -> Dict[str, float]:
    if name == "time_correlation":
        return fit_time_correlation(df, tolerance_minutes)
    if name == "time_correlation_backward":
        return fit_time_correlation_backward(df, tolerance_minutes)
    if name == "time_correlation_anypair":
        return fit_time_correlation_anypair(df, tolerance_minutes)
    if name == "network":
        return fit_oddball(df)
    if name == "burst":
        return fit_network(df)
    if name == "oddball":
        return fit_oddball(df)
    if name == "oddball_signed":
        return fit_oddball_signed(df)
    if name == "degree_deviation":
        return fit_degree_deviation(df)
    if name == "reciprocity":
        return fit_reciprocity(df)
    if name == "statml":
        return fit_statml(df)
    if name == "statml_eif":
        return fit_statml_eif(df)
    if name == "statml_lof":
        return fit_statml_lof(df)
    if name == "statml_mahalanobis":
        return fit_statml_mahalanobis(df)
    if name == "statml_pca":
        return fit_statml_pca(df)
    if name == "statml_zscore":
        return fit_statml_zscore(df)
    if name == "benford":
        return fit_benford(df)
    if name == "benford_ks":
        return fit_benford_ks(df)
    if name == "structuring":
        return fit_structuring(df, threshold=structuring_threshold)
    if name == "structuring_banded":
        return fit_structuring_banded(df, threshold=structuring_threshold)
    if name == "behavioral":
        return fit_behavioral(df)
    return {}