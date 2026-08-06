"""Data-sufficiency engine (§1B).

For every model, compute fields populated vs required and resolve the verdict:
SUPPORTED / DEGRADED / BLOCKED, each with a human-readable, actionable reason.
The pipeline assembles ONLY the model set the data can support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .schema import (
    MODEL_REQUIRED_FIELDS,
    SOURCE_UNIQUE,
    SOURCE_CAPABILITIES,
    SourceResolved,
)

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_DEGRADED = "DEGRADED"
VERDICT_BLOCKED = "BLOCKED"

EVENT_TYPE_PRETTY = {"bank": "bank (account)", "cdr": "call (phone)", "social": "social (follow/hashtag)"}
EVENT_LABEL_PRETTY = {"bank": "bank", "cdr": "CDR", "social": "social"}


@dataclass
class SufficiencyVerdict:
    model: str
    status: str  # SUPPORTED | DEGRADED | BLOCKED
    populated: Set[str]
    required: Set[str]
    ratio: float
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    def render(self) -> str:
        return f"{self.model}: {self.status} — {self.reason}"


class SufficiencyEngine:
    def __init__(self, resolved_sources: List[SourceResolved], min_events_per_entity: int = 30):
        self.resolved_sources = resolved_sources
        self.min_events_per_entity = min_events_per_entity
        self.present_sources = {rs.source for rs in resolved_sources}
        # union of all resolved canonical fields across sources
        self.populated: Set[str] = set()
        for rs in resolved_sources:
            self.populated |= rs.resolved_fields

    # -- helpers -----------------------------------------------------------
    def _volume_note(self, rs_list: List[SourceResolved]) -> Optional[str]:
        """Return a DEGRADE reason if any source has thin per-entity volume."""
        for rs in rs_list:
            if rs.n_rows and rs.n_entities:
                per_entity = rs.n_rows / max(rs.n_entities, 1)
                if per_entity < self.min_events_per_entity:
                    return (
                        f"entity volume low ({per_entity:.1f} events/entity in {rs.source}); "
                        f"some features dropped for reliability"
                    )
        return None

    def _overall_volume_note(self, n_rows: int, n_entities: int) -> Optional[str]:
        if n_entities and n_rows / n_entities < self.min_events_per_entity:
            return (
                f"insufficient volume for statistical reliability (n≈{n_rows}/{n_entities}"
                f" = {n_rows/max(n_entities,1):.1f}/entity, threshold {self.min_events_per_entity})"
            )
        return None

    # -- per-model bespoke reasons -----------------------------------------
    def evaluate(self) -> Dict[str, SufficiencyVerdict]:
        verd: Dict[str, SufficiencyVerdict] = {}

        # schema detection / entity resolution always run
        verd["schema_detection"] = SufficiencyVerdict(
            "schema_detection", VERDICT_SUPPORTED, self.populated, set(), 1.0,
            f"resolved {len(self.populated)} canonical slots", list(self.populated),
        )
        if self.present_sources:
            verd["entity_resolution"] = SufficiencyVerdict(
                "entity_resolution", VERDICT_SUPPORTED, self.populated, {"counterparty"}, 1.0,
                "counterparty/edge slots resolved", sorted(self.populated),
            )
        else:
            verd["entity_resolution"] = SufficiencyVerdict(
                "entity_resolution", VERDICT_BLOCKED, set(), {"counterparty"}, 0.0,
                "entity_resolution blocked: no source files loaded",
            )

        # time correlation (always, but internal vs cross-source messaging)
        if "timestamp" in self.populated:
            verd["time_correlation"] = SufficiencyVerdict(
                "time_correlation", VERDICT_SUPPORTED, {"timestamp"}, {"timestamp"}, 1.0,
                "timestamp resolved; enables internal + cross-source correlation",
            )
        else:
            verd["time_correlation"] = SufficiencyVerdict(
                "time_correlation", VERDICT_BLOCKED, set(), {"timestamp"}, 0.0,
                "time-correlation model skipped: no timestamp dimension resolved",
            )

        # network
        if "counterparty" in self.populated:
            if len(self.present_sources) < 2:
                verd["network"] = SufficiencyVerdict(
                    "network", VERDICT_DEGRADED, {"counterparty"}, {"counterparty"}, 1.0,
                    "network DEGRADED: single source; edges internal only (add CDR/bank for cross-source strength)",
                )
            else:
                verd["network"] = SufficiencyVerdict(
                    "network", VERDICT_SUPPORTED, {"counterparty"}, {"counterparty"}, 1.0,
                    "counterparty/edge dimension resolved; cross-source graph available",
                )
        else:
            verd["network"] = SufficiencyVerdict(
                "network", VERDICT_BLOCKED, set(), {"counterparty"}, 0.0, self._edge_hint(),
            )

        # statml (needs amount + timestamp for the amount-heavy feature vector)
        if "amount" not in self.populated:
            if "timestamp" in self.populated:
                verd["statml"] = SufficiencyVerdict(
                    "statml", VERDICT_DEGRADED, {"timestamp"}, {"amount", "timestamp"}, 0.5,
                    "stat/ML DEGRADED: no amount resolved; running frequency/duration-only features.",
                )
            else:
                verd["statml"] = SufficiencyVerdict(
                    "statml", VERDICT_BLOCKED, set(), {"amount", "timestamp"}, 0.0,
                    "stat/ML blocked: no amount or timestamp dimension resolved",
                )
        else:
            verd["statml"] = SufficiencyVerdict(
                "statml", VERDICT_SUPPORTED, {"amount", "timestamp"}, {"amount", "timestamp"}, 1.0,
                "amount + timestamp resolved; full feature vector available",
            )

        # benford + structuring need amount
        if "amount" in self.populated:
            verd["benford"] = SufficiencyVerdict(
                "benford", VERDICT_SUPPORTED, {"amount"}, {"amount"}, 1.0,
                "amount resolved; first-digit deviation scoring available",
            )
            verd["structuring"] = SufficiencyVerdict(
                "structuring", VERDICT_SUPPORTED, {"amount"}, {"amount"}, 1.0,
                "amount resolved; threshold-proximity structuring rule available",
            )
            verd["behavioral"] = SufficiencyVerdict(
                "behavioral", VERDICT_SUPPORTED, {"amount", "timestamp"}, {"amount", "timestamp"}, 1.0,
                "timestamp + amount resolved; regime-flip scoring available",
            )
        else:
            msg = "BLOCKED: no amount/counterparty value dimension resolved. Add a bank (account) file to enable it."
            verd["benford"] = SufficiencyVerdict(
                "benford", VERDICT_BLOCKED, set(), {"amount"}, 0.0, msg,
            )
            verd["structuring"] = SufficiencyVerdict(
                "structuring", VERDICT_BLOCKED, set(), {"amount"}, 0.0, msg,
            )
            verd["behavioral"] = SufficiencyVerdict(
                "behavioral", VERDICT_BLOCKED, set(), {"amount", "timestamp"}, 0.0,
                "behavioral blocked: no amount/timestamp dimension resolved",
            )

        # time-correlation DEGRADED if very sparse
        if verd["time_correlation"].status == VERDICT_SUPPORTED:
            note = self._overall_note()
            if note:
                verd["time_correlation"] = SufficiencyVerdict(
                    "time_correlation", VERDICT_DEGRADED, {"timestamp"}, {"timestamp"}, 1.0, note,
                )
        return verd

    # -- helpers -----------------------------------------------------------
    def _overall_note(self) -> Optional[str]:
        total_rows = sum(rs.n_rows for rs in self.resolved_sources)
        total_entities = sum(rs.n_entities for rs in self.resolved_sources)
        return self._overall_volume_note(total_rows, total_entities)

    def _amount_hint(self) -> str:
        return "Missing `amount` dimension: the structuring/Benford views are unavailable."

    def _edge_hint(self) -> str:
        parts = []
        for src in ("bank", "cdr", "social"):
            if src in self.present_sources:
                parts.append(f"you already have {src}")
            else:
                parts.append(f"add a {EVENT_LABEL_PRETTY.get(src, src)} file ({SOURCE_UNIQUE.get(src, '')})")
        return "Network-correlation model blocked: no counterparty/edge dimension resolved. " + " ".join(parts)

    def __contains__(self, item: str) -> bool:
        return item in self.populated