"""Stage B — Entity resolution (§2.3).

Blocking + Jaro-Winkler fuzzy matching to tie one person across bank, CDR and
social sources with no shared ID. Every source gives a person a *different*
identifier (account / phone / handle) but the same name; alternative name
spellings are reconciled with the fuzzy match index.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd
from rapidfuzz import fuzz, process

NAME_THRESHOLD = 88.0  # Jaro-Winkler / token-set join threshold
UNRESOLVED = "__unresolved__"


@dataclass
class EntityResolution:
    entity_id_by_identifier: Dict[str, str]   # raw actor/counterparty id -> entity_id
    name_to_entity: Dict[str, str]            # canonical name -> entity_id
    entity_names: Dict[str, Set[str]]         # entity_id -> observed names
    n_entities: int = 0
    n_cross_source: int = 0
    detail: List[str] = field(default_factory=list)


class _NameIndex:
    """Incremental fuzzy index of canonical names, keyed by entity_id."""

    def __init__(self) -> None:
        self._names: List[str] = []
        self._eid_of: Dict[str, str] = {}

    def add(self, name: str, eid: str) -> None:
        self._names.append(name)
        self._eid_of[name] = eid

    def match(self, name: str) -> tuple:
        if not name:
            return None, 0.0
        res = process.extractOne(name, self._names, scorer=fuzz.token_sort_ratio, score_cutoff=NAME_THRESHOLD)
        return (self._eid_of[res[0]], res[1]) if res else (None, 0.0)


def resolve_entities(
    df: pd.DataFrame,
    actor_col: str = "actor_raw",
    actor_name_col: str = "actor_name",
    counterparty_col: str = "counterparty_id_raw",
) -> EntityResolution:
    if actor_name_col not in df.columns:
        df = df.copy()
        df[actor_name_col] = df[actor_col]
    if "source" not in df.columns:
        df = df.copy()
        df["source"] = ""

    id_to_entity: Dict[str, str] = {}
    name_to_entity: Dict[str, str] = {}
    entity_names: Dict[str, Set[str]] = {}
    index = _NameIndex()

    counter = Counter()
    eid = (lambda: f"E{counter['m']:04d}")

    def next_eid() -> str:
        counter["m"] += 1
        return f"E{counter['m']:04d}"

    def claim(identifier: Optional[str], name: str, source: str) -> str:
        if identifier and identifier in id_to_entity:
            ent = id_to_entity[identifier]
            entity_names.setdefault(ent, set()).add(name)
            return ent
        if name:
            m_eid, score = index.match(name)
            # Name-based linking only merges ACROSS sources. Two identifiers
            # in the SAME source that share a name are different customers
            # (name collisions are common in synthetic + real populations);
            # merging them pollutes every downstream entity-level signal
            # (a dormant person's stream suddenly contains a stranger's
            # events, which is exactly what destroyed the behavioral model).
            if m_eid and source not in _sources_of.get(m_eid, set()):
                if identifier:
                    id_to_entity[identifier] = m_eid
                entity_names.setdefault(m_eid, set()).add(name)
                name_to_entity[name] = m_eid
                _sources_of.setdefault(m_eid, set()).add(source)
                return m_eid
        ent = next_eid()
        if identifier:
            id_to_entity[identifier] = ent
        entity_names[ent] = {name} if name else set()
        name_to_entity[name] = ent
        index.add(name, ent)
        _sources_of.setdefault(ent, set()).add(source)
        return ent

    # sources each entity has actor identifiers in (same-source name
    # collisions must never merge)
    _sources_of: Dict[str, Set[str]] = {}

    # pass 1: build actor entities from (identifier, name, source) triples
    rows = df[[actor_col, actor_name_col, "source"]].dropna(subset=[actor_col]).drop_duplicates()
    for _, row in rows.iterrows():
        claim(
            str(row[actor_col]) if pd.notna(row[actor_col]) else None,
            str(row[actor_name_col]) if pd.notna(row[actor_name_col]) else "",
            str(row["source"]) if pd.notna(row["source"]) else "",
        )

    # pass 2: resolve counterparty identifiers onto the same clusters
    counterparty_entities: Dict[str, str] = {}
    for _cp in df[counterparty_col].dropna().unique():
        _cp = str(_cp)
        if _cp in id_to_entity:
            counterparty_entities[_cp] = id_to_entity[_cp]
        else:
            counterparty_entities[_cp] = UNRESOLVED
    # merge counterparty eids into the same id map (unresolved -> identity)
    for cp, ent in counterparty_entities.items():
        if ent != UNRESOLVED:
            id_to_entity.setdefault(cp, ent)

    # entity_props
    actor_ids: List[str] = df[actor_col].dropna().unique().astype(str)
    n_actor_entities = len(set(id_to_entity[a] for a in actor_ids if a in id_to_entity))

    res = EntityResolution(
        entity_id_by_identifier=id_to_entity,
        name_to_entity=name_to_entity,
        entity_names=dict(entity_names),
        n_entities=counter["m"],
        n_cross_source=0,
    )
    # measure cross-source: an entity seen in >1 source
    source_of_entity: Dict[str, Set[str]] = {}
    sub = df[[actor_col, "source"]].dropna()
    for _, r in sub.iterrows():
        ent = id_to_entity.get(r[actor_col])
        if ent:
            source_of_entity.setdefault(ent, set()).add(r["source"])
    res.n_cross_source = sum(1 for s in source_of_entity.values() if len(s) > 1)
    res.detail.append(f"{res.n_entities} entities, {res.n_cross_source} across >1 source")
    return res