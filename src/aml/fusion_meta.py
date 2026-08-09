"""F — fusion candidate: learned meta-learner (RF / XGBoost) ranking.

The Borda/score fusions assume every model deserves equal weight. The
meta-learner instead *learns* which signals matter: per-entity features are
the per-model scores (six families) plus the shared behavioural profile
(n_events, fan-in/out, amount shape, rhythm), and the target is the synthetic
answer key (storyline + surprise ground truth from the two-tier generator).

Training happens offline in scripts/train_fusion_meta.py; the persisted
model ships in src/aml/assets/fusion_meta.joblib. At inference the pipeline
always keeps a safe fallback: if the model file is missing or prediction
fails (e.g. different feature set), fuse_rank_meta() degrades to Borda so
the demo can never break.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import Fusion, build_features, EPS

_ASSET = Path(__file__).resolve().parent / "assets" / "fusion_meta.joblib"
_loaded = {"clf": None, "path": None}

_META_FEATURES = [
    "n_events", "fan_in", "fan_out", "amt_total", "amt_mean", "amt_std",
    "amt_max", "hour_entropy", "night_ratio", "gap_max", "gap_std",
]


def _log1p(s: pd.Series) -> pd.Series:
    return np.log1p(s.clip(lower=0))


def build_meta_features(scores: Dict[str, Dict[str, float]],
                        df: pd.DataFrame) -> pd.DataFrame:
    """Per-entity row = one column per model family + behaviour profile."""
    model_names = sorted(scores)
    ids = set()
    for s in scores.values():
        ids.update(s.keys())
    if not ids:
        return pd.DataFrame(columns=model_names + _META_FEATURES)
    X = pd.DataFrame(index=sorted(ids))
    for m in model_names:
        X[m] = [scores[m].get(e, 0.0) for e in X.index]
    feats = build_features(df)
    if not feats.empty:
        for c in _META_FEATURES:
            if c in feats.columns:
                X[c] = feats[c].reindex(X.index).fillna(0.0)
    for c in ("amt_total", "amt_mean", "amt_std", "amt_max", "gap_max", "gap_std"):
        if c in X.columns:
            X[c] = _log1p(X[c])
    X = X.fillna(0.0)
    return X


def train_meta_classifier(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """Train RandomForest + XGBoost, return the better one (by CV AP).

    Returns (best_clf, cv_scores) — cv_scores = {"rf": ap, "xgb": ap}.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    labels = y.reindex(X.index).fillna(0).astype(int)
    if labels.nunique() < 2 or len(X) < 12:
        raise ValueError("insufficient labelled data for meta-learner")

    cv = StratifiedKFold(n_splits=min(5, labels.value_counts().min()), shuffle=True,
                         random_state=seed)
    scores = {}
    for name, clf in (
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                      class_weight="balanced", random_state=seed)),
        ("xgb", __import__("xgboost").XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=1.0,
            random_state=seed, eval_metric="aucpr")),
    ):
        try:
            proba = cross_val_predict(clf, X.values, labels.values, cv=cv,
                                      method="predict_proba", n_jobs=1)
            scores[name] = float(average_precision_score(
                labels.values, proba[:, 1]))
        except Exception:
            continue
    if not scores:
        raise ValueError("both meta candidates failed to train")
    winner = max(scores, key=scores.get)
    best = RandomForestClassifier(n_estimators=300, max_depth=6,
                                  class_weight="balanced",
                                  random_state=seed) if winner == "rf" else \
        __import__("xgboost").XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, random_state=seed, eval_metric="aucpr")
    best.fit(X.values, labels)
    return best, scores


def save_meta_model(clf, path: Path) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    return path


def load_meta_model(path: Path = _ASSET):
    import joblib
    if _loaded["path"] == str(path):
        return _loaded["clf"]
    clf = joblib.load(str(path))
    _loaded.update({"clf": clf, "path": str(path)})
    return clf


class FusionMeta(Fusion):
    """Learned ranking: P(anomaly) from the trained meta-learner.

    Falls back to Borda semantics whenever the model is unavailable.
    """

    def __init__(self, normalized_scores, df, clf=None):
        self.clf = clf
        self.fallback = None
        super().__init__(normalized_scores, df)

    def _fuse(self) -> None:
        from .models import fuse_rank_borda
        try:
            clf = self.clf if self.clf is not None else load_meta_model()
            X = build_meta_features(self.scores, self.df)
            if X.empty or not hasattr(clf, "feature_names_in_"):
                raise ValueError("feature mismatch")
            X = X.reindex(columns=clf.feature_names_in_, fill_value=0.0)
            agg = {e: float(p) for e, p in
                   zip(X.index, clf.predict_proba(X.values)[:, 1])}
            self._finalize(agg)
        except Exception:
            fb = fuse_rank_borda(self.scores, self.df)
            self.fallback = True
            self.rank = fb.rank
            self.explanation = fb.explanation


def fuse_rank_meta(scores: Dict[str, Dict[str, float]], df: pd.DataFrame,
                   clf=None) -> FusionMeta:
    """Meta-learner fusion with automatic Borda fallback."""
    return FusionMeta(scores, df, clf=clf)