"""Emit synthetic CSVs (with hostile, varied headers) + answer keys."""
from __future__ import annotations

import json
import random
import secrets
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .population import Person, PersonRegistry, make_person, normal_bank_events, normal_cdr_events, normal_social_events
from .storylines import STORYLINE_FACTORIES, STORYLINE_POOL, STORYLINE_SOURCES, compatible_storylines, apply_storyline

# ---------------------------------------------------------------------------
# Cross-source correlation modes. "full" = everyone appears in all three
# sources (the original scenario). "none" = three disjoint populations, one
# per source. "pair_*" = one population shares the two named sources while a
# second population only appears in the remaining source.
# ---------------------------------------------------------------------------
CORRELATION_MODES = ("full", "none", "pair_bank_cdr", "pair_bank_social",
                     "pair_cdr_social")
PAIR_SOURCES = {
    "pair_bank_cdr": {"bank", "cdr"},
    "pair_bank_social": {"bank", "social"},
    "pair_cdr_social": {"cdr", "social"},
}
_SOURCES = {"bank", "cdr", "social"}


def _assign_sources(correlation: str, rng: random.Random) -> set:
    """Which sources a newly created person participates in."""
    if correlation == "full":
        return set(_SOURCES)
    if correlation == "none":
        return {rng.choice(("bank", "cdr", "social"))}
    pair = PAIR_SOURCES[correlation]
    if rng.random() < 0.7:
        return set(pair)
    return set(_SOURCES - pair)

# ---------------------------------------------------------------------------
# Header templates per source: exercise Stage A dynamic schema detection
# ---------------------------------------------------------------------------
BANK_HEADERS = {
    "A": {"ts": "txn_date", "direction": "debit/credit", "amount": "amount",
          "counterparty_acc": "beneficiary", "branch": "branch_code",
          "name": "account_holder", "account": "account_id"},
    "B": {"ts": "posted_at", "direction": "dr_cr", "amount": "value",
          "counterparty_acc": "payee", "branch": "branch",
          "name": "customer_name", "account": "account_number"},
    "C": {"ts": "transaction_date", "direction": "txn_type", "amount": "amount",
          "counterparty_acc": "recipient_account", "branch": "branch_code",
          "name": "account_name", "account": "account_no"},
}
CDR_HEADERS = {
    "A": {"ts": "call_time", "msisdn": "msisdn", "partner_phone": "remote_number",
          "tower": "tower_id", "duration": "duration_seconds", "name": "subscriber_name"},
    "B": {"ts": "start_time", "msisdn": "caller_msisdn", "partner_phone": "callee",
          "tower": "cell_id", "duration": "length", "name": "subscriber"},
    "C": {"ts": "timestamp", "msisdn": "originating_number", "partner_phone": "terminating_number",
          "tower": "tower", "duration": "duration_secs", "name": "name"},
}
SOCIAL_HEADERS = {
    "A": {"ts": "created_utc", "handle": "user_handle", "mentions": "mentions",
          "geo": "geo_tag", "name": "display_name"},
    "B": {"ts": "post_time", "handle": "handle", "mentions": "tagged_users",
          "geo": "location", "name": "name"},
    "C": {"ts": "timestamp", "handle": "username", "mentions": "hashtags",
          "geo": "place", "name": "full_name"},
}

NOISE_COLUMNS = ["batch_id", "device", "notes"]

# possible emitted column names carrying the person's name, per source
# (the generator renames headers to hostile variants to exercise Stage A)
NAME_COLUMN_CANDIDATES = {
    "bank": sorted({v["name"] for v in BANK_HEADERS.values()}),
    "cdr": sorted({v["name"] for v in CDR_HEADERS.values()}),
    "social": sorted({v["name"] for v in SOCIAL_HEADERS.values()}),
}


def name_column(df: pd.DataFrame, source: str) -> str:
    """Which of the emitted name columns this dataframe actually uses."""
    for cand in NAME_COLUMN_CANDIDATES.get(source, ["name"]):
        if cand in df.columns:
            return cand
    return "name"


def _write_csv(df: pd.DataFrame, path: Path, header_map: Dict[str, str], extra: str = "") -> None:
    ren = {k: v for k, v in header_map.items() if k in df.columns}
    out = df.rename(columns=ren)
    if extra:
        out[extra] = "noise"
    out.to_csv(path, index=False)


def _annotate(person: Person, person_type: str, storyline: str, note: str, extra: dict) -> dict:
    return {
        "person_type": person_type,
        "storyline": storyline,
        "note": note,
        "name": person.name,
        "phone": person.phone,
        "account": person.account,
        "handle": person.handle,
        "details": extra,
    }


def generate_all(
    out_dir: Path,
    seed: int = 42,
    surprise_seed: int = 99,
    n_background: int = 90,
    n_per_storyline: int = 3,
    n_surprise: int = 12,
    threshold: float = 10000.0,
    correlation: str = "full",
) -> Dict[str, Path]:
    files = {}
    scenario = build_scenario(seed=seed, surprise_seed=surprise_seed,
                              n_background=n_background,
                              n_per_storyline=n_per_storyline,
                              n_surprise=n_surprise, threshold=threshold,
                              correlation=correlation)
    bank_rows, cdr_rows, social_rows = scenario["bank_rows"], scenario["cdr_rows"], scenario["social_rows"]
    annotations = scenario["annotations"]
    surprise_ids = scenario["surprise_ids"]

    out_dir = Path(out_dir)
    sources_dir = out_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    rng = scenario["rng"]  # same stream that generated the people (header picks)

    if not bank_rows.empty:
        files["bank"] = _emit_source(out_dir, "bank", bank_rows, rng, "bank_export.csv")
    if not cdr_rows.empty:
        files["cdr"] = _emit_source(out_dir, "cdr", cdr_rows, rng, "cdr_export.csv")
    if not social_rows.empty:
        files["social"] = _emit_source(out_dir, "social", social_rows, rng, "social_export.csv")

    # answer keys
    answer_key = {"entities": annotations, "meta": {"threshold": threshold}}
    (out_dir / "answer_key.json").write_text(json.dumps(answer_key, indent=2))

    # secret surprise key — write-only (used by eval harness only)
    surprise_key = {
        "entities": {pid_: annotations[pid_] for pid_ in surprise_ids},
        "meta": {"seed": surprise_seed},
    }
    (out_dir / "answer_key_surprise.json").write_text(json.dumps(surprise_key, indent=2))

    files["answer_key"] = out_dir / "answer_key.json"
    files["surprise_key"] = out_dir / "answer_key_surprise.json"
    return files


def build_scenario(
    seed: int = 42,
    surprise_seed: int = 99,
    n_background: int = 90,
    n_per_storyline: int = 3,
    n_surprise: int = 12,
    threshold: float = 10000.0,
    correlation: str = "full",
) -> Dict:
    """Deterministic people + events + annotations (the joint fact sheet).

    Everything below is *one* draw from the RNGs; separate outputs (CSVs,
    JSONL streams, free-form logs) rendered from this sheet are identical in
    entity identity and storylines — so any two formats must evaluate to the
    same recall if the dynamic extractor is working.

    ``correlation`` controls cross-source correlation: "full" (everyone in
    all three sources), "none" (disjoint per-source populations) or
    "pair_*" (two correlated sources + a third independent one).
    """
    assert correlation in CORRELATION_MODES, correlation
    rng = random.Random(seed)
    sng = random.Random(surprise_seed)

    people: Dict[str, Person] = {}
    pid = 1

    # --- background population -------------------------------------------------
    for _ in range(n_background):
        p = make_person(f"E{pid:03d}", rng)
        pid += 1
        p._is_scripted = False
        p._is_surprise = False
        p._sources = _assign_sources(correlation, rng)
        people[p.person_id] = p
    # friendship edges within background (small-world so network model has signal)
    bg_ids = list(people.keys())
    for p in people.values():
        for f in rng.sample(bg_ids, min(3, len(bg_ids))):
            if f != p.person_id:
                p.friend_ids.append(f)

    # --- scripted storylines ----------------------------------------------------
    annotations: Dict[str, dict] = {}
    for story in STORYLINE_FACTORIES:
        for i in range(n_per_storyline):
            p = make_person(f"E{pid:03d}", rng)
            pid += 1
            p._is_scripted = story
            p._is_surprise = False
            p._sources = _assign_sources(correlation, rng)
            people[p.person_id] = p
    for p in people.values():
        if not p._is_scripted:
            continue
        for f in rng.sample(list(people.keys()), min(2, len(people))):
            if f != p.person_id and not people[f]._is_scripted:
                p.friend_ids.append(f)

    # --- surprise pool (must NOT include scripted) --------------------------------
    surprise_ids = apply_surprise_entities(people, sng, n_surprise, pid)

    registry = PersonRegistry(people, rng)

    # --- generate events -----------------------------------------------------------
    event_holder: Dict[str, Dict[str, List[dict]]] = {}
    for p in people.values():
        normal: Dict[str, List[dict]] = {}
        if "bank" in p._sources:
            normal["bank"] = normal_bank_events(registry, p, rng)
        if "cdr" in p._sources:
            normal["cdr"] = normal_cdr_events(registry, p, rng)
        if "social" in p._sources:
            normal["social"] = normal_social_events(registry, p, rng)
        p._events = normal
        event_holder[p.person_id] = normal
        annotations[p.person_id] = _annotate(p, "background", "", "", {})

    def _apply(p, rng_, story):
        """Apply a storyline, then drop events outside the person's sources."""
        events, ann = apply_storyline(p, registry, rng_, p._events, story)
        events = {s: evs for s, evs in events.items() if s in p._sources}
        if not any(events.values()):
            story = rng_.choice(compatible_storylines(p._sources))
            events, ann = apply_storyline(p, registry, rng_, p._events, story)
            events = {s: evs for s, evs in events.items() if s in p._sources}
        p._events = events
        event_holder[p.person_id] = events
        return story, ann

    # scripted storylines
    for p in people.values():
        if p._is_scripted:
            story = p._is_scripted
            if story not in compatible_storylines(p._sources):
                story = rng.choice(compatible_storylines(p._sources))
            story, ann = _apply(p, rng, story)
            annotations[p.person_id] = _annotate(p, "storyline", story, ann["note"], ann)

    # surprise storylines (randomized)
    for pid_ in surprise_ids:
        p = people[pid_]
        story = sng.choice(STORYLINE_POOL)
        if story not in compatible_storylines(p._sources):
            story = sng.choice(compatible_storylines(p._sources))
        story, ann = _apply(p, sng, story)
        annotations[p.person_id] = _annotate(p, "surprise", story, ann["note"], ann)

    bank_rows, cdr_rows, social_rows = [], [], []
    for p in people.values():
        for ev in p._events.get("bank", []):
            bank_rows.append(ev)
        for ev in p._events.get("cdr", []):
            cdr_rows.append(ev)
        for ev in p._events.get("social", []):
            social_rows.append(ev)

    return {
        "rng": rng,
        "people": people,
        "annotations": annotations,
        "surprise_ids": surprise_ids,
        "person_sources": {pid_: set(p._sources) for pid_, p in people.items()},
        "bank_rows": pd.DataFrame(bank_rows),
        "cdr_rows": pd.DataFrame(cdr_rows),
        "social_rows": pd.DataFrame(social_rows),
    }


def apply_surprise_entities(people: Dict[str, Person], sng: random.Random, n: int, pid: int) -> List[str]:
    """Pick random NEW entities for the surprise set (never scripted)."""
    pool = [p for p in people.values() if not p._is_scripted]
    chosen = sng.sample(pool, min(n, len(pool)))
    ids = []
    for p in chosen:
        p._is_surprise = True
        ids.append(p.person_id)
    return ids


def _emit_source(out_dir: Path, source: str, df: pd.DataFrame, rng: random.Random, fname: str) -> Path:
    sources_dir = out_dir / "sources"
    if source == "bank":
        template = BANK_HEADERS[rng.choice(list(BANK_HEADERS.keys()))]
        extra = rng.choice(NOISE_COLUMNS)
    elif source == "cdr":
        template = CDR_HEADERS[rng.choice(list(CDR_HEADERS.keys()))]
        extra = rng.choice(NOISE_COLUMNS)
    else:
        template = SOCIAL_HEADERS[rng.choice(list(SOCIAL_HEADERS.keys()))]
        extra = rng.choice(NOISE_COLUMNS)
    df = df.copy()
    # never leak the secret entity id (`pid`) or the raw internal timestamp slot
    df["timestamp"] = pd.to_datetime(df.pop("ts") if "ts" in df.columns else df["timestamp"])
    df = df.drop(columns=[c for c in ["pid"] if c in df.columns], errors="ignore")
    path = sources_dir / fname
    _write_csv(df, path, template, extra=extra)
    return path