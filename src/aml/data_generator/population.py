"""Persona registry + normal behavior generation.

Every person has a *secret* identity (person_id) that is NEVER written to CSV.
The pipeline must resolve one person across sources without a shared ID: bank
events carry account + account_holder name, CDR carries msisdn + subscriber name,
social carries a handle + name.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

FIRST_NAMES = [
    "Ana", "Ben", "Cara", "Diego", "Elena", "Fahad", "Grace", "Hugo", "Ivy", "Jin",
    "Kai", "Lena", "Mara", "Noah", "Omar", "Priya", "Ravi", "Sara", "Tomas",
    "Uma", "Victor", "Wes", "Xin", "Yara", "Zane", "Ava", "Leo", "Nia", "Raj",
]
LAST_NAMES = [
    "Alvarez", "Brooks", "Chen", "Duarte", "ElSayed", "Fischer", "Gomes", "Haddad",
    "Iqbal", "Jansen", "Kim", "Lopes", "Meyer", "Novak", "Okafor", "Patida", "Quinn",
    "Rossi", "Silva", "Tanaka", "Unger", "Vargas", "Wei", "Yamamoto", "Zhou",
    "Anderson", "Bakker", "Carlos", "Diaz", "Mehta",
]

REPORTING_THRESHOLD = 10000.0  # structuring boundary
DAYS_HISTORY = 150


@dataclass
class Person:
    person_id: str
    name: str
    first: str = ""
    last: str = ""
    phone: str = ""     # CDR subscriber msisdn
    account: str = ""   # bank account id
    handle: str = ""    # social handle
    home_tower: str = ""
    work_tower: str = ""
    home_branch: str = ""
    friend_ids: List[str] = field(default_factory=list)

    def source_bank_key(self) -> str:
        return self.name

    def source_cdr_key(self) -> str:
        return self.name

    def source_social_key(self) -> str:
        return self.name


class PersonRegistry:
    def __init__(self, people: Dict[str, Person], rng: random.Random):
        self.people = people
        self.rng = rng
        self._ids = list(people.keys())
        # prune dead friends
        for p in people.values():
            p.friend_ids = [f for f in p.friend_ids if f in people]

    def ids(self) -> List[str]:
        return list(self._ids)

    def get(self, pid: str) -> Person:
        return self.people[pid]

    def random_person(self, exclude: str = "") -> Person:
        while True:
            p = self.rng.choice(list(self.people.values()))
            if p.person_id != exclude:
                return p


def make_person(pid: str, rng: random.Random) -> Person:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    pc = rng.randint(100, 999)
    phone = f"639{pc:03d}{rng.randint(1000000, 9999999)}"
    acct = f"ACCT{pc:03d}{rng.randint(1000, 9999)}"
    handle = f"@{first.lower()}.{last.lower()}{rng.randint(10, 999)}"
    return Person(
        person_id=pid, name=name, first=first, last=last,
        phone=phone, account=acct, handle=handle,
        home_tower=f"T{rng.randint(100, 999)}",
        work_tower=f"T{rng.randint(100, 999)}",
        home_branch=f"BRANCH{rng.randint(10, 99)}",
    )


def _ts(rng: random.Random, days_ago: int, hour_bias: int = 0, minute_jitter: int = 30) -> np.datetime64:
    """Random timestamp `days_ago` days before reference date, day-heavy distribution."""
    if rng.random() < 0.82:
        hour = (hour_bias + rng.randint(8, 20)) % 24
    else:
        hour = (hour_bias + rng.choice([rng.randint(0, 5), rng.randint(21, 23)])) % 24
    minute = rng.randint(0, minute_jitter)
    sec = rng.randint(0, 59)
    base = np.datetime64("2026-01-01") - np.timedelta64(days_ago, "D")
    return base + np.timedelta64(hour * 3600 + minute * 60 + sec, "s")


def _amount(rng: random.Random) -> float:
    a = rng.lognormvariate(6.5, 0.9)  # median ~600-700
    if a > 9000:
        a = rng.uniform(200, 2200)  # avoid accidental near-threshold structuring
    return round(a, 2)


def normal_bank_events(reg: PersonRegistry, p: Person, rng: random.Random) -> List[dict]:
    """Regular spend + occasional transfers to friends, never carefully timed."""
    events = []
    for day in range(max(1, DAYS_HISTORY - 60), DAYS_HISTORY + 1):
        if rng.random() > 0.45:
            continue
        for _ in range(rng.choice([1, 1, 2])):
            direction = rng.choice(["credit", "debit"])
            amount = _amount(rng)
            counterparty_acc = None
            if rng.random() < 0.28:
                cp = reg.random_person(p.person_id)
                counterparty_acc = cp.account
                if direction == "debit":
                    amount = min(amount, 3000)
            events.append({
                "ts": _ts(rng, day, minute_jitter=45),
                "direction": direction,
                "amount": amount,
                "counterparty_acc": counterparty_acc,
                "branch": p.home_branch if rng.random() < 0.7 else f"BRANCH{rng.randint(10, 99)}",
                "name": p.name,
                "account": p.account,
            })
    return events


def normal_cdr_events(reg: PersonRegistry, p: Person, rng: random.Random) -> List[dict]:
    events = []
    for day in range(max(1, DAYS_HISTORY - 75), DAYS_HISTORY + 1):
        for _ in range(rng.choice([1, 2, 2, 3])):
            if p.friend_ids and rng.random() < 0.6:
                friend = reg.get(rng.choice(p.friend_ids))
                phone = friend.phone
            else:
                phone = reg.random_person().phone
            tower = p.home_tower if rng.random() < 0.6 else p.work_tower
            events.append({
                "pid": p.person_id,
                "ts": _ts(rng, day),
                "msisdn": p.phone,
                "partner_phone": phone,
                "tower": tower,
                "duration": round(rng.lognormvariate(2.8, 0.7), 1),
                "name": p.name,
            })
    return events


def normal_social_events(reg: PersonRegistry, p: Person, rng: random.Random) -> List[dict]:
    events = []
    for day in range(max(1, DAYS_HISTORY - 60), DAYS_HISTORY + 1):
        if rng.random() > 0.3:
            continue
        mentions = ""
        geo = ""
        if p.friend_ids and rng.random() < 0.5:
            handles = [reg.get(f).handle for f in rng.sample(p.friend_ids, min(len(p.friend_ids), 2))]
            mentions = ",".join(handles)
        if rng.random() < 0.2:
            geo = f"LAT{rng.randint(1000, 2000)}"
        events.append({
            "pid": p.person_id,
            "ts": _ts(rng, day),
            "handle": p.handle,
            "mentions": mentions,
            "geo": geo,
            "name": p.name,
        })
    return events