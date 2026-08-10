"""Predefined anomaly storylines (§1C.4, §2.1).

Six executable typologies, each with documented ground truth. Factories receive
the person's normal events (per source) and return modified event lists plus an
annotation dict with the anomaly type and human description.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List

import numpy as np

from .population import REPORTING_THRESHOLD, Person, PersonRegistry, _ts

# ---------------------------------------------------------------------------
# storyline 1: colocation — call/post minutes before a transfer to same party
# ---------------------------------------------------------------------------
def storyline_colocation(person, registry, rng, normal):
    bank = list(normal.get("bank", []))
    cdr = list(normal.get("cdr", []))
    partner = registry.random_person(person.person_id)
    # pick ~6 episodes scattered over recent history
    for i in range(6):
        days_ago = rng.randint(2, 40)
        call_ts = _ts(rng, days_ago, hour_bias=rng.randint(8, 18))
        xfer_ts = call_ts + np.timedelta64(rng.randint(10, 26) * 60, "s")  # 10-26 min later
        amount = round(rng.uniform(5000, 48000), 2)
        bank.append({
            "ts": xfer_ts, "direction": "debit", "amount": amount,
            "counterparty_acc": partner.account, "branch": person.home_branch,
            "name": person.name, "account": person.account,
        })
        cdr.append({
            "pid": person.person_id, "ts": call_ts,
            "msisdn": person.phone, "partner_phone": partner.phone,
            "tower": person.home_tower, "duration": round(rng.uniform(30, 180), 1),
            "name": person.name,
        })
    return {"bank": bank, "cdr": cdr}, {
        "type": "colocation",
        "note": f"call then transfer to {partner.person_id} within minutes on 6 episodes",
    }


# ---------------------------------------------------------------------------
# storyline 2: dormant_flip — 40+ days silent then sudden large activity
# ---------------------------------------------------------------------------
def storyline_dormant_flip(person, registry, rng, normal):
    bank = []
    cdr = []
    social = list(normal.get("social", []))
    # dormant window: days_ago 45..150 quiet (keep only last 45 days of normal)
    quiet_until = 45
    for ev in normal.get("bank", []):
        days_ago = (np.datetime64("2026-01-01") - ev["ts"]) / np.timedelta64(1, "D")
        if days_ago <= quiet_until:
            bank.append(ev)
    for ev in normal.get("cdr", []):
        days_ago = (np.datetime64("2026-01-01") - ev["ts"]) / np.timedelta64(1, "D")
        if days_ago <= quiet_until:
            cdr.append(ev)
    # sudden burst: several deposits in a week after dormancy
    for i in range(rng.randint(4, 8)):
        days_ago = rng.randint(3, quiet_until)
        bank.append({
            "ts": _ts(rng, days_ago, hour_bias=9),
            "direction": "credit",
            "amount": round(rng.lognormvariate(9.2, 0.4), 2),  # 10k-40k
            "counterparty_acc": registry.random_person(person.person_id).account,
            "branch": person.home_branch, "name": person.name, "account": person.account,
        })
    # and a flurry of calls to new numbers
    for i in range(rng.randint(5, 10)):
        cdr.append({
            "pid": person.person_id, "ts": _ts(rng, rng.randint(2, quiet_until)),
            "msisdn": person.phone,
            "partner_phone": registry.random_person(person.person_id).phone,
            "tower": person.work_tower, "duration": round(rng.uniform(20, 300), 1),
            "name": person.name,
        })
    return {"bank": bank, "cdr": cdr, "social": social}, {
        "type": "dormant_flip",
        "note": "dormant 45+ days then sudden deposits + call flurry",
    }


# ---------------------------------------------------------------------------
# storyline 3: structuring — many deposits just below reporting threshold
# ---------------------------------------------------------------------------
def storyline_structuring(person, registry, rng, normal):
    bank = list(normal.get("bank", []))
    for i in range(rng.randint(12, 18)):
        amount = round(rng.uniform(0.88, 0.97) * REPORTING_THRESHOLD, 2)
        bank.append({
            "ts": _ts(rng, rng.randint(1, 45), hour_bias=11),
            "direction": "credit", "amount": amount,
            "counterparty_acc": registry.random_person(person.person_id).account,
            "branch": person.home_branch, "name": person.name, "account": person.account,
        })
    return {"bank": bank}, {
        "type": "structuring",
        "note": f"{len(bank)} deposits clustered just under {REPORTING_THRESHOLD:,.0f}",
    }


# ---------------------------------------------------------------------------
# storyline 4: fan_io — many distinct senders fan-in, then fan-out in days
# ---------------------------------------------------------------------------
def storyline_fan_io(person, registry, rng, normal):
    bank = list(normal.get("bank", []))
    day = rng.randint(2, 20)
    senders = [registry.random_person(person.person_id) for _ in range(rng.randint(8, 14))]
    for s in senders:
        bank.append({
            "ts": _ts(rng, day, hour_bias=10), "direction": "credit",
            "amount": round(rng.uniform(800, 2500), 2),
            "counterparty_acc": s.account, "branch": person.home_branch,
            "name": person.name, "account": person.account,
        })
    day2 = max(day - rng.randint(1, 3), 1)
    recipients = [registry.random_person(person.person_id) for _ in range(rng.randint(8, 14))]
    for r in recipients:
        bank.append({
            "ts": _ts(rng, day2, hour_bias=14), "direction": "debit",
            "amount": round(rng.uniform(500, 1800), 2),
            "counterparty_acc": r.account, "branch": person.home_branch,
            "name": person.name, "account": person.account,
        })
    return {"bank": bank}, {
        "type": "fan_io",
        "note": f"fan-in from {len(senders)} senders then fan-out to {len(recipients)} in ~3 days",
    }


# ---------------------------------------------------------------------------
# storyline 5: silence — large transaction then total silence (regime flip)
# ---------------------------------------------------------------------------
def storyline_silence(person, registry, rng, normal):
    bank = list(normal.get("bank", []))
    cdr = []
    social = []
    # keep only old events, then one big debit, then nothing
    cutoff = 30
    for ev in normal.get("bank", []):
        days_ago = (np.datetime64("2026-01-01") - ev["ts"]) / np.timedelta64(1, "D")
        if days_ago > cutoff:
            bank.append(ev)
    bank.append({
        "ts": _ts(rng, cutoff - 1, hour_bias=9), "direction": "debit",
        "amount": round(rng.uniform(200000, 800000), 2),
        "counterparty_acc": registry.random_person(person.person_id).account,
        "branch": person.home_branch, "name": person.name, "account": person.account,
    })
    return {"bank": bank, "cdr": cdr, "social": social}, {
        "type": "silence",
        "note": f"large debit (~{bank[-1]['amount']:,.0f}) then zero activity thereafter",
    }


# ---------------------------------------------------------------------------
# storyline 6: chain — rapid A->B->C->D layering transfers within minutes
# ---------------------------------------------------------------------------
def storyline_chain(person, registry, rng, normal):
    bank = list(normal.get("bank", []))
    chain = [person] + [registry.random_person(prev.person_id) for prev in [person, person, person]]
    hop_ts = []
    for i in range(3):
        hop_ts.append(_ts(rng, rng.randint(1, 10), hour_bias=12))
    amounts = [round(rng.uniform(40000, 120000), 2), round(rng.uniform(30000, 90000), 2), round(rng.uniform(20000, 80000), 2)]
    for i in range(3):
        src, dst = chain[i], chain[i + 1]
        bank.append({
            "ts": hop_ts[i], "direction": "debit", "amount": amounts[i],
            "counterparty_acc": dst.account, "branch": src.home_branch,
            "name": src.name, "account": src.account,
        })
    return {"bank": bank}, {
        "type": "chain",
        "note": "3-hop layering " + " -> ".join(c.person_id for c in chain),
    }


# ---------------------------------------------------------------------------
# storyline 7: post_burst — a day of dozens of posts all mentioning the same
# handles, so social-only entities still carry a detectable signal.
# ---------------------------------------------------------------------------
def storyline_post_burst(person, registry, rng, normal):
    social = list(normal.get("social", []))
    day = rng.randint(2, 15)
    handles = [registry.random_person(person.person_id).handle for _ in range(3)]
    geo = f"LAT{rng.randint(1000, 2000)}"
    for _ in range(rng.randint(20, 30)):
        social.append({
            "pid": person.person_id,
            "ts": _ts(rng, day, hour_bias=10),
            "handle": person.handle,
            "mentions": ",".join(handles),
            "geo": geo,
            "name": person.name,
        })
    return {"social": social}, {
        "type": "post_burst",
        "note": f"{len(social)} posts on one day all mentioning the same {len(handles)} handles",
    }


STORYLINE_FACTORIES: Dict[str, Callable] = {
    "colocation": storyline_colocation,
    "dormant_flip": storyline_dormant_flip,
    "structuring": storyline_structuring,
    "fan_io": storyline_fan_io,
    "silence": storyline_silence,
    "chain": storyline_chain,
    "post_burst": storyline_post_burst,
}

STORYLINE_POOL = list(STORYLINE_FACTORIES.keys())

# which sources each storyline writes events into (used to keep anomaly
# signals inside the sources a person actually participates in)
STORYLINE_SOURCES: Dict[str, set] = {
    "colocation": {"bank", "cdr"},
    "dormant_flip": {"bank", "cdr"},
    "structuring": {"bank"},
    "fan_io": {"bank"},
    "silence": {"bank"},
    "chain": {"bank"},
    "post_burst": {"social"},
}


def compatible_storylines(sources) -> List[str]:
    """Storylines that create events in at least one of the given sources."""
    return [s for s in STORYLINE_POOL if STORYLINE_SOURCES[s] & set(sources)]


def apply_storyline(person, registry, rng, normal, storyline: str):
    factory = STORYLINE_FACTORIES[storyline]
    return factory(person, registry, rng, normal)