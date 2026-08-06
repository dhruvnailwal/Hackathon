"""Synthetic data generator: base population + predefined storylines + surprise injector.

Two-tier data design (§1C.4 / §2.1): scripted storylines carry documented ground truth;
a separately-seeded randomized injector writes ONLY to a secret answer key so any entity
flagged from that set is genuine generalization, not a lookup table.
"""
from __future__ import annotations

from .population import (
    Person,
    PersonRegistry,
    make_person,
    normal_bank_events,
    normal_cdr_events,
    normal_social_events,
)
from .storylines import (
    STORYLINE_FACTORIES,
    apply_storyline,
)
from .emit import generate_all, apply_surprise_entities
from .emit import generate_all

__all__ = [
    "Person",
    "PersonRegistry",
    "make_person",
    "normal_bank_events",
    "normal_cdr_events",
    "normal_social_events",
    "STORYLINE_FACTORIES",
    "apply_storyline",
    "generate_all",
]