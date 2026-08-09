"""Unified pipeline: sources -> schema detect -> entity resolve -> long df -> models.

Assembles ONLY the model set the data can support (sufficiency engine) and
produces ranked insights with explanations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .schema import SourceResolved
from .schema_detect import detect_dataframe
from .sufficiency import SufficiencyEngine, SufficiencyVerdict

UNIFIED_COLUMNS = [
    "entity_id", "event_type", "timestamp", "source",
    "amount", "counterparty_id", "location", "direction", "duration",
]


@dataclass
class PipelineResult:
    unified: pd.DataFrame
    per_source: List[SourceResolved]
    sufficiency: Dict[str, SufficiencyVerdict]
    entity_map: object
    rankings: List[dict] = field(default_factory=list)
    explanations: Dict[str, str] = field(default_factory=dict)
    model_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class Pipeline:
    def __init__(self, min_events_per_entity: int = 30, tol_minutes: int = 25,
                 structuring_threshold: float = 10000.0,
                 fusion: str = "borda"):
        self.min_events_per_entity = min_events_per_entity
        self.tol_minutes = tol_minutes
        self.structuring_threshold = structuring_threshold
        self.fusion = fusion

    def ingest(self, paths):
        from .loaders import load_any
        frames = []
        resolved_sources = []
        for path in paths:
            loaded = load_any(path)
            det = detect_dataframe(loaded.df, file=loaded.file)
            if loaded.warning:
                det.warnings.insert(0, loaded.warning)
            norm = det.normalized
            rs = SourceResolved(
                source=det.source_type, file=det.file,
                resolved_fields=det.resolved_slots,
                warnings=list(det.warnings),
                n_rows=det.n_rows, n_entities=0,
            )
            frames.append(norm)
            resolved_sources.append(rs)
        non_empty = [f for f in frames if f.shape[0] > 0]
        if non_empty:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                unified = pd.concat(non_empty, ignore_index=True, sort=False)
        else:
            unified = pd.DataFrame()
        for f in UNIFIED_COLUMNS + ["actor_raw", "counterparty_id_raw", "actor_name"]:
            if f not in unified.columns:
                unified[f] = np.nan
        if "timestamp" in unified.columns:
            unified["timestamp"] = pd.to_datetime(unified["timestamp"], errors="coerce")
        return unified, resolved_sources

    def resolve(self, unified, resolved_sources):
        from .entity_resolve import resolve_entities
        em = resolve_entities(unified)
        unified = unified.copy()
        unified["entity_id"] = unified["actor_raw"].map(em.entity_id_by_identifier)
        unified["counterparty_id"] = unified["counterparty_id_raw"].map(em.entity_id_by_identifier)
        for rs in resolved_sources:
            sub = unified[unified["source"] == rs.source]
            rs.n_entities = int(sub["entity_id"].nunique()) if "entity_id" in sub else 0
        return unified, em

    def run(self, paths, with_models=True) -> PipelineResult:
        unified, resolved_sources = self.ingest(paths)
        unified, em = self.resolve(unified, resolved_sources)
        engine = SufficiencyEngine(resolved_sources, self.min_events_per_entity)
        sufficiency = engine.evaluate()
        result = PipelineResult(
            unified=unified, per_source=resolved_sources,
            sufficiency=sufficiency, entity_map=em,
        )
        if with_models and not unified.empty:
            self._run_models(result)
        return result

    def _run_models(self, result):
        from . import models as _m
        sufficiency = result.sufficiency
        df = result.unified
        scores = {}
        for model in ("time_correlation", "network", "statml", "benford", "structuring", "behavioral"):
            if sufficiency[model].status != "BLOCKED":
                sc = _m.run_model(model, df, tolerance_minutes=self.tol_minutes,
                                  structuring_threshold=self.structuring_threshold)
                if sc:
                    scores[model] = sc
        fusion = self._fuse(scores, df)
        result.model_scores = scores
        result.rankings = fusion.rank
        result.explanations = fusion.explanation
        result.warnings = [v.reason for v in sufficiency.values() if v.status == "BLOCKED"]
        return result

    def _fuse(self, scores, df):
        """Build the fused ranking; unknown fusion names fall back to Borda."""
        from . import models as _m
        if self.fusion == "meta":
            try:
                from .fusion_meta import fuse_rank_meta
                return fuse_rank_meta(scores, df)
            except Exception:
                pass
        if self.fusion in ("score_mean", "rank_avg", "top2"):
            fn = {"score_mean": _m.fuse_score_mean, "rank_avg": _m.fuse_rank_average,
                  "top2": _m.fuse_model_scores}[self.fusion]
            return fn(scores, df)
        return _m.fuse_rank_borda(scores, df)