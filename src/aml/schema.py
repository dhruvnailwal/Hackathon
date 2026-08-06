"""Canonical schema, field aliases, and source-capability registry.

This module is the SINGLE SOURCE OF TRUTH for adaptivity (§1B of PLAN.md):
each source type declares which canonical field slots it can fill, and every
model declares `required_features()` so the sufficiency engine is self-describing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set

# ---------------------------------------------------------------------------
# Canonical field slots (SCHEMA.md)
# ---------------------------------------------------------------------------
CANONICAL_SLOTS: FrozenSet[str] = frozenset(
    {
        "timestamp",
        "amount",
        "counterparty",
        "event_type",
        "location",
        "direction",
        "duration",
    }
)

# event_type taxonomy
EVENT_TYPES: Dict[str, str] = {
    "bank": "transaction",
    "cdr": "call",
    "social": "post",
}

# ---------------------------------------------------------------------------
# FIELD_ALIASES — header-name dictionary for dynamic schema detection (§2.2)
# Fuzzy-matched with rapidfuzz (threshold ~80); content-probe is the fallback.
# Actor identities (who the row belongs to) are kept SEPARATE from counterparty
# (who the event is with), so schema detection can disambiguate them.
# ---------------------------------------------------------------------------
FIELD_ALIASES: Dict[str, List[str]] = {
    "timestamp": [
        "timestamp",
        "datetime",
        "time",
        "txn_date",
        "txn_datetime",
        "transaction_date",
        "posted_at",
        "post_time",
        "call_time",
        "created_utc",
        "created_at",
        "date",
        "event_time",
        "when",
    ],
    "amount": [
        "amount",
        "value",
        "txn_amount",
        "transaction_amount",
        "amount_usd",
        "debit_amount",
        "credit_amount",
        "sum",
        "total",
        "amt",
    ],
    "counterparty": [
        "counterparty",
        "counterparty_id",
        "beneficiary",
        "beneficiary_id",
        "payee",
        "recipient",
        "recipient_account",
        "to_account",
        "other_party",
        "remote_number",
        "callee",
        "terminating_number",
        "contact",
        "mentions",
        "tagged_users",
        "hashtags",
        "target",
        "to",
        "dest",
    ],
    "event_type": [
        "event_type",
        "type",
        "txn_type",
        "kind",
        "category",
        "record_type",
    ],
    "location": [
        "location",
        "branch",
        "branch_code",
        "tower",
        "tower_id",
        "cell",
        "cell_id",
        "lac",
        "geo_tag",
        "city",
        "place",
        "terminal",
    ],
    "direction": [
        "direction",
        "dr_cr",
        "debit_credit",
        "in_out",
        "txn_direction",
        "sign",
    ],
    "duration": ["duration", "call_duration", "dur_sec", "seconds", "length", "duration_seconds", "duration_secs"],
}

# who the row belongs to — the actor identity (used for entity resolution)
ACTOR_ID_ALIASES: Dict[str, List[str]] = {
    "actor_id": [
        "subscriber", "subscriber_id", "customer", "customer_id", "client_id",
        "party_id", "iban", "account", "account_id", "account_number", "account_no",
        "msisdn", "caller_msisdn", "originating_number", "caller", "user_id",
        "device_id", "device", "handle", "user_handle", "username", "user_name",
    ],
}
ACTOR_NAME_ALIASES: Dict[str, List[str]] = {
    "actor_name": [
        "name", "full_name", "display_name", "account_holder", "customer_name",
        "account_name", "subscriber_name", "subscriber", "account_holder_name",
        "owner_name", "user_name",
    ],
}

# ---------------------------------------------------------------------------
# Source capability registry (table from §1B)
# ---------------------------------------------------------------------------
SOURCE_CAPABILITIES: Dict[str, Set[str]] = {
    "bank": {"timestamp", "amount", "counterparty", "event_type", "location", "direction"},
    "cdr": {"timestamp", "counterparty", "event_type", "location", "duration"},
    "social": {"timestamp", "event_type", "counterparty"},
}

# Unique capabilities per source (used for messaging)
SOURCE_UNIQUE = {
    "bank": "amount / direction / balance",
    "cdr": "duration / towers / strong edge signal",
    "social": "text sentiment / weak edges",
}


def source_supports(source: str, slot: str) -> bool:
    return slot in SOURCE_CAPABILITIES.get(source, set())


# ---------------------------------------------------------------------------
# Model capability matrix (plan §1B) — which canonical slots each model needs
# ---------------------------------------------------------------------------
MODEL_REQUIRED_FIELDS: Dict[str, Set[str]] = {
    "time_correlation": {"timestamp"},
    "network": {"counterparty"},
    "statml": {"amount", "timestamp"},
    "benford": {"amount"},
    "structuring": {"amount"},
    "entity_resolution": {"counterparty"},
    "schema_detection": set(),
}


@dataclass
class ModelRequirement:
    name: str
    required_fields: Set[str]
    min_events_per_entity: int = 30  # plan §1B DEGRADED rule
    description: str = ""


# ---------------------------------------------------------------------------
# Registry entry: what a resolved source actually populated
# ---------------------------------------------------------------------------
@dataclass
class SourceResolved:
    source: str
    file: str
    resolved_fields: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_entities: int = 0