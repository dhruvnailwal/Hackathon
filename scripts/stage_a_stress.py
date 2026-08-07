"""Stage A permutation stress harness — dynamic field extraction across formats.

Runs the full permutation matrix of:
    source template x attribute subset x header style x file format
through loaders -> schema detection (header / content / hybrid modes)
-> sufficiency engine, and asserts:

  1. every loadable permutation resolves the *expected* canonical slots
  2. field detection works WITHOUT meaningful column headings (content probe)
  3. missing attributes produce explicit BLOCKED / DEGRADED "insufficient data"
     verdicts with human-readable reasons (never silent)
  4. end-to-end pipeline runs complete for representative combos

Results are stored under ``results/stage_a/`` (JSON per case + summary.md).
Exit code is non-zero when any unexpected result is found.

Usage:
    python scripts/stage_a_stress.py [--cases 40] [--out results/stage_a]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from aml.loaders import load_any
from aml.pipeline import Pipeline
from aml.schema_detect import detect_dataframe
from aml.sufficiency import VERDICT_BLOCKED, VERDICT_DEGRADED, VERDICT_SUPPORTED

rng = random.Random(7)

# ---------------------------------------------------------------------------
# attribute spec: every attribute carries (value generator, expected slot,
# detectable by content probe?) so the harness can distinguish EXPECTED gaps
# from REGRESSIONS.
# ---------------------------------------------------------------------------
FIRST = ["Ana", "Ben", "Cara", "Diego", "Elena", "Fahad", "Grace"]
LAST = ["Alvarez", "Brooks", "Chen", "Duarte", "ElSayed", "Fischer"]


def _gen_ts():
    return rng.choice([
        "2024-01-%02dT%02d:%02d:00" % (rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59)),
        "2024-01-%02d %02d:%02d:%02d" % (rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)),
        "2024-01-%02dT%02d:%02d:00Z" % (rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59)),
        "2024-01-%02d" % rng.randint(1, 28),
    ])


def _gen_amt():
    return rng.choice([
        round(rng.uniform(1, 48000), 2),
        "$%s" % f"{rng.uniform(1, 48000):,.2f}",
        -round(rng.uniform(1, 5000), 2),
        round(rng.uniform(100, 9000), 0),
    ])


def _gen_cparty():
    return rng.choice([
        "639%09d" % rng.randint(0, 999999999),
        "ACCT%d" % rng.randint(1000000, 9999999),
        "DE89370400440532013000",  # IBAN
    ])


def _gen_dir():
    return rng.choice(["credit", "debit", "CR", "DR", "C", "D"])


def _gen_dur():
    return int(rng.uniform(5, 900))


def _gen_loc():
    return rng.choice(["T%d" % rng.randint(100, 999), "BRANCH%02d" % rng.randint(10, 99),
                       "tower_%d" % rng.randint(100, 999), "cell_%d" % rng.randint(1, 50)])


def _gen_name():
    # small pool: content detection needs the two-word pattern, and a handful
    # of repeated names keeps per-entity volume above the sufficiency gate
    return rng.choice(["Ana Alvarez", "Ben Brooks", "Cara Chen", "Diego Duarte",
                       "Elena ElSayed", "Fahad Fischer", "Grace Gomes", "Hugo Haddad"])


def _gen_actor():
    # small deterministic pool so each case has a handful of stable entities
    # with many events (real exports repeat one id per entity); ~37 events/
    # entity at 300 rows keeps the volume gate satisfied
    return rng.choice([f"ACCT{i:08d}" for i in range(4)] + [f"63900000{i:03d}" for i in range(4)])


def _gen_et():
    return rng.choice(["101", "202", "303"])


ATTRS = {
    "ts":      {"gen": _gen_ts,      "slot": "timestamp",    "content_ok": True},
    "amt":     {"gen": _gen_amt,     "slot": "amount",       "content_ok": True},
    "cparty":  {"gen": _gen_cparty,  "slot": "counterparty", "content_ok": True},
    "dir":     {"gen": _gen_dir,     "slot": "direction",    "content_ok": False},
    "dur":     {"gen": _gen_dur,     "slot": "duration",     "content_ok": True},  # via numeric-code? see below
    "loc":     {"gen": _gen_loc,     "slot": "location",     "content_ok": True},
    "name":    {"gen": _gen_name,    "slot": "actor_name",   "content_ok": True},
    "actor":   {"gen": _gen_actor,   "slot": "actor_id",     "content_ok": True},
    "et":      {"gen": _gen_et,      "slot": "event_type",   "content_ok": True},
    "noise":   {"gen": lambda: rng.choice(["x", "y", "z"]),  "slot": "",           "content_ok": False},
}
# duration numeric-columns fall out as "numeric-code" (event_type) not duration
ATTRS["dur"]["content_ok"] = False
# actor ids are phones/account numbers — content cannot tell them apart from
# counterparty ids (both are identifiers); only header matching disambiguates
ATTRS["actor"]["content_ok"] = False

# realistic header names per attribute (alias dictionary coverage)
HEADER_ALIAS = {
    "ts": "txn_date", "amt": "amount", "cparty": "beneficiary", "dir": "debit/credit",
    "dur": "call_duration", "loc": "branch_code", "name": "account_holder",
    "actor": "account_id", "et": "txn_type", "noise": "batch_id",
}

# source templates (informational only — subsets drive the cases)
TEMPLATES = {
    "bank":   ["ts", "amt", "cparty", "dir", "loc", "name", "actor", "noise"],
    "cdr":    ["ts", "cparty", "dur", "loc", "name", "actor", "noise"],
    "social": ["ts", "cparty", "name", "actor", "noise"],
}

# attribute subsets per source: (label, attrs, rows)
# row counts: 300 keeps ~30 events/entity for the stable actor pool (volume
# SUPPORTED); tiny_rows exercises the DEGRADED volume contract on purpose.
SUBSETS = {
    "bank": [
        ("full",          ["ts", "amt", "cparty", "dir", "loc", "name", "actor"], 300),
        ("minimal",       ["ts", "amt", "cparty"], 300),
        ("no_amount",     ["ts", "cparty", "dir", "loc", "name", "actor"], 300),
        ("no_cparty",     ["ts", "amt", "dir", "loc", "name", "actor"], 300),
        ("no_ts",         ["amt", "cparty", "dir", "loc", "name", "actor"], 300),
        ("tiny_rows",     ["ts", "amt", "cparty"], 12),
        ("sparse_null",   ["ts", "amt", "cparty"], 300),
    ],
    "cdr": [
        ("full",          ["ts", "cparty", "dur", "loc", "name", "actor"], 300),
        ("minimal",       ["ts", "cparty"], 300),
        ("no_cparty",     ["ts", "dur", "loc", "name", "actor"], 300),
        ("no_ts",         ["cparty", "dur", "loc", "name", "actor"], 300),
        ("tiny_rows",     ["ts", "cparty"], 12),
    ],
    "social": [
        ("full",          ["ts", "cparty", "name", "actor"], 300),
        ("minimal",       ["ts"], 300),
        ("no_ts",         ["cparty", "name", "actor"], 300),
        ("tiny_rows",     ["ts"], 12),
    ],
}

HEADER_STYLES = ["alias", "obscure", "upper", "scrambled"]

FORMATS: Dict[str, Callable[[Path, List[str], List[dict]], Path]] = {}


def _write_csv(path: Path, header: List[str], rows: List[dict], delim: str) -> Path:
    import csv
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=delim, quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(c, "") for c in header])
    return path


def _f_csv_comma(p, h, rows, attrs=None):
    return _write_csv(p.with_suffix(".csv"), h, rows, ",")


def _f_csv_tsv(p, h, rows, attrs=None):
    return _write_csv(p.with_suffix(".tsv"), h, rows, "\t")


def _f_csv_semi(p, h, rows, attrs=None):
    return _write_csv(p.with_suffix(".csv"), h, rows, ";")


def _f_csv_pipe(p, h, rows, attrs=None):
    return _write_csv(p.with_suffix(".csv"), h, rows, "|")


def _f_json(p, h, rows, attrs=None):
    path = p.with_suffix(".json")
    path.write_text(json.dumps(rows, default=str), encoding="utf-8")
    return path


def _f_jsonl(p, h, rows, attrs=None):
    path = p.with_suffix(".jsonl")
    path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    return path


def _f_txt_tab(p, h, rows, attrs=None):
    path = p.with_suffix(".txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(h) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in h) + "\n")
    return path


def _f_txt_kv(p, h, rows, attrs=None):
    path = p.with_suffix(".log")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            toks = []
            for c in h:
                v = str(r.get(c, ""))
                if " " in v:
                    v = '"%s"' % v
                toks.append(f"{c}={v}")
            fh.write(" ".join(toks) + "\n")
    return path
def _f_txt_prose(p, h, rows, attrs=None):
    """Free-form log lines: tokens in prose, not key=value (dynamic extractor).

    ``attrs`` maps each header position back to its attribute, so obfuscated
    headers (c0, c1, ...) still render the right values, and only the
    attributes actually present are written — no fabricated dimensions.
    """
    attrs = attrs or list(range(len(h)))
    idx = {a: i for i, a in enumerate(attrs)}

    def _val(r, a):
        i = idx.get(a)
        if i is None or i >= len(h):
            return ""
        return str(r.get(h[i], ""))

    path = p.with_suffix(".log")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            ts = _val(r, "ts")
            amt = _val(r, "amt")
            cparty = _val(r, "cparty")
            actor = _val(r, "actor")
            has_dur = any(_val(r, a) for a in ("dur", "duration"))
            parts = []
            if ts:
                parts.append(ts)
            parts.append("INFO")
            parts.append("call" if has_dur else "transfer")
            if actor:
                parts.append(f"from {actor}")
            if cparty:
                parts.append(f"to {cparty}")
            if amt:
                # honest prose: a bare number is not money — label it with a
                # currency code so the typed-token extractor can type it too
                amt_tok = str(amt)
                if not re.search(r"[$€£₹]|(?:\b(?:USD|EUR|GBP|INR)\b)", amt_tok):
                    amt_tok = f"{amt_tok} USD"
                parts.append(f"amount {amt_tok}")
            if has_dur:
                parts.append("duration 120s")
            fh.write(" ".join(parts) + "\n")
    return path


FORMATS = {
    "csv-comma": _f_csv_comma, "csv-tsv": _f_csv_tsv, "csv-semi": _f_csv_semi,
    "csv-pipe": _f_csv_pipe, "json": _f_json, "jsonl": _f_jsonl,
    "txt-tab": _f_txt_tab, "txt-kv": _f_txt_kv, "txt-prose": _f_txt_prose,
}

# txt-prose renders only the fields the prose template can carry; the other
# formats round-trip every attribute verbatim.
FORMAT_SLOTS = {fmt: None for fmt in FORMATS}
FORMAT_SLOTS["txt-prose"] = {"timestamp", "amount", "counterparty", "actor_id"}

# header styles that defeat header matching (content probe must carry)
CONTENT_ONLY_STYLES = ("obscure", "scrambled")


def _header_for(attrs: List[str], style: str) -> List[str]:
    if style == "alias":
        return [HEADER_ALIAS.get(a, a) for a in attrs]
    if style == "obscure":
        return [f"c{i}" for i in range(len(attrs))]
    if style == "upper":
        return [HEADER_ALIAS.get(a, a).upper().replace("-", "_") for a in attrs]
    # scrambled: alias letters shuffled (noise like "tmuna", "unmoat")
    out = []
    for a in attrs:
        w = list(HEADER_ALIAS.get(a, a).replace("_", ""))
        rng.shuffle(w)
        out.append("".join(w))
    return out


def build_rows(attrs: List[str], n: int, header: List[str] | None = None,
               sparse: bool = False) -> List[dict]:
    """Header-per-index mapping so obfuscated column names still get values."""
    rows = []
    for _ in range(n):
        r = {}
        for i, a in enumerate(attrs):
            if sparse and a not in ("ts", "actor") and rng.random() < 0.45:
                continue
            h = header[i] if header else HEADER_ALIAS.get(a, a)
            r[h] = ATTRS[a]["gen"]()
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# expected sufficiency verdicts per subset
# ---------------------------------------------------------------------------
def expected_verdicts(source: str, label: str, hstyle: str, attrs: List[str]) -> Dict[str, object]:
    """(model, expected status) — the "insufficient data" contract.

    Values may be a single verdict or a tuple of allowed verdicts. Content-only
    detection cannot distinguish an entity-id column from a counterparty-id
    column: when the probe resolves it (as counterparty), the network model is
    DEGRADED (single source, internal edges only) instead of BLOCKED, and both
    are honest outcomes.
    """
    content_only = hstyle in CONTENT_ONLY_STYLES
    no_entity = not (any(a in ("actor", "name") for a in attrs))

    STAT_BLOCK = {"benford", "structuring", "behavioral"}
    ENTITY_NOTE = VERDICT_DEGRADED
    ANY_DEGRADED = (VERDICT_BLOCKED, VERDICT_DEGRADED)

    if label == "tiny_rows":
        return {"time_correlation": VERDICT_DEGRADED}

    if label == "sparse_null":
        # amount/counterparty/timestamp all still resolve (probe tolerates
        # nulls); without an entity dimension every model either degrades via
        # the no-entity note, or (network) via the single-source note.
        return {m: ENTITY_NOTE for m in
                ("benford", "structuring", "behavioral", "statml",
                 "network", "time_correlation")}

    if source == "bank":
        if label == "no_amount":
            out = {**{m: VERDICT_BLOCKED for m in STAT_BLOCK},
                   "statml": VERDICT_DEGRADED}
            if no_entity:
                out["time_correlation"] = VERDICT_DEGRADED
            return out
        if label == "no_cparty":
            out = {"network": ANY_DEGRADED if content_only else VERDICT_BLOCKED}
            if no_entity:
                out["time_correlation"] = VERDICT_DEGRADED
            return out
        if label == "no_ts":
            return {"time_correlation": VERDICT_BLOCKED, "statml": VERDICT_BLOCKED}
        out = {"benford": VERDICT_SUPPORTED, "structuring": VERDICT_SUPPORTED,
               "behavioral": VERDICT_SUPPORTED, "statml": VERDICT_SUPPORTED,
               "network": VERDICT_DEGRADED, "time_correlation": VERDICT_SUPPORTED}
        if no_entity:
            for m in out:
                out[m] = VERDICT_DEGRADED if out[m] == VERDICT_SUPPORTED else out[m]
        return out
    if source == "cdr":
        if label == "no_cparty":
            out = {"network": ANY_DEGRADED if content_only else VERDICT_BLOCKED}
            if no_entity:
                out["time_correlation"] = VERDICT_DEGRADED
            return out
        if label == "no_ts":
            return {"time_correlation": VERDICT_BLOCKED, "statml": VERDICT_BLOCKED}
        out = {"benford": VERDICT_BLOCKED, "structuring": VERDICT_BLOCKED,
               "behavioral": VERDICT_BLOCKED, "statml": VERDICT_DEGRADED,
               "network": VERDICT_DEGRADED, "time_correlation": VERDICT_SUPPORTED}
        if no_entity:
            out["time_correlation"] = VERDICT_DEGRADED
        return out
    # social
    if label == "no_ts":
        return {"time_correlation": VERDICT_BLOCKED, "statml": VERDICT_BLOCKED}
    out = {"benford": VERDICT_BLOCKED, "structuring": VERDICT_BLOCKED,
           "behavioral": VERDICT_BLOCKED, "statml": VERDICT_DEGRADED,
           "network": VERDICT_DEGRADED, "time_correlation": VERDICT_SUPPORTED}
    if "cparty" not in attrs:
        out["network"] = ANY_DEGRADED if content_only else VERDICT_BLOCKED
    if no_entity:
        out["time_correlation"] = VERDICT_DEGRADED
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=0, help="0 = full matrix")
    ap.add_argument("--out", default=str(ROOT / "results" / "stage_a"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    by_method: Dict[str, Dict[str, int]] = {}

    cases = []
    for source, subsets in SUBSETS.items():
        for label, attrs, n_rows in subsets:
            for hstyle in HEADER_STYLES:
                for fmt_name in FORMATS:
                    cases.append((source, label, attrs, n_rows, hstyle, fmt_name))
    if args.cases:
        rng.shuffle(cases)
        cases = cases[:args.cases]

    tmp = Path(tempfile.mkdtemp(prefix="stage_a_stress_"))
    try:
        for idx, (source, label, attrs, n_rows, hstyle, fmt_name) in enumerate(cases):
            case = {"case": idx, "source": source, "subset": label,
                    "header_style": hstyle, "format": fmt_name,
                    "attrs": attrs, "rows": n_rows}
            errors: List[str] = []
            try:
                header = _header_for(attrs, hstyle)
                rows = build_rows(attrs, n_rows, header=header, sparse=label == "sparse_null")
                path = FORMATS[fmt_name](tmp / f"case{idx}", header, rows, attrs)
                loaded = load_any(str(path))
                det = detect_dataframe(loaded.df, file=path.name, mode="hybrid")

                resolved = {rc.slot or "??": rc.method for rc in det.columns.values()}
                for col, rc in det.columns.items():
                    key = rc.slot or "UNMAPPED"
                    m_key = "content" if str(rc.method).startswith("content") else ("header" if rc.method == "header" else "unmapped")
                    by_method.setdefault(key, {})[m_key] = by_method.get(key, {}).get(m_key, 0) + 1

                # slot-level expectations against aliases + content detection
                expected_slots = {ATTRS[a]["slot"] for a in attrs if ATTRS[a]["slot"]}
                if hstyle in CONTENT_ONLY_STYLES:
                    # only content-detectable slots are required from the probe
                    exp = {ATTRS[a]["slot"] for a in attrs if ATTRS[a].get("content_ok")}
                    exp.discard("")  # actor pseudo-slots count via content probe
                else:
                    exp = {s for s in expected_slots if s}
                cap = FORMAT_SLOTS[fmt_name]
                if cap is not None:
                    exp &= cap
                got = {s for s in exp if resolved.get(s, "unmapped") != "unmapped"}
                missing = exp - got
                if missing:
                    errors.append(f"slots not resolved: {sorted(missing)}")

                # sufficiency contract (BLOCKED/DEGRADED "insufficient data" messages)
                # ---- use the FULL pipeline path (ingest + resolve) so entity
                # counts feed the volume / no-entity degradation checks ----
                pipe = Pipeline(min_events_per_entity=30)
                res2 = pipe.run([str(path)], with_models=False)
                verdicts = res2.sufficiency
                for model, want in expected_verdicts(source, label, hstyle, attrs).items():
                    got_v = verdicts[model].status
                    allowed = (want,) if isinstance(want, str) else tuple(want)
                    if got_v not in allowed:
                        errors.append(f"sufficiency {model}: want {want} got {got_v} ({verdicts[model].reason})")
                    if got_v in (VERDICT_BLOCKED, VERDICT_DEGRADED) and not verdicts[model].reason.strip():
                        errors.append(f"sufficiency {model}: {got_v} with empty reason")
                case["verdicts"] = {m: (v.status, v.reason) for m, v in verdicts.items()}
                case["n_detected"] = len(resolved)
            except Exception as exc:  # noqa: BLE001 — stress harness must survive
                errors.append(f"exception: {type(exc).__name__}: {exc}")
            case["errors"] = errors
            if errors:
                failures.append(case)

            (out_dir / f"case_{idx:04d}.json").write_text(
                json.dumps(case, indent=2, default=str), encoding="utf-8")

        # ----- summary -----------------------------------------------------
        n_fail = len(failures)
        n_total = len(cases)
        summary = []
        summary.append(f"# Stage A stress results — {n_total} cases, {n_fail} with errors")
        summary.append("")
        summary.append(f"generated: {pd.Timestamp.now()}")
        summary.append("")
        if failures:
            summary.append("## Failures")
            for c in failures[:80]:
                summary.append(f"- case {c['case']}: {c['source']}/{c['subset']}/"
                               f"{c['header_style']}/{c['format']}")
                for e in c["errors"]:
                    summary.append(f"    - {e}")
            summary.append("")
        summary.append("## Slot resolution by method")
        summary.append("")
        summary.append("| slot | header | content | unmapped |")
        summary.append("|---|---|---|---|")
        for slot in sorted(by_method):
            d = by_method[slot]
            summary.append(f"| {slot} | {d.get('header', 0)} | {d.get('content', 0)} | {d.get('unmapped', 0)} |")
        (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"Stage A stress: {n_total} cases, {n_fail} failures -> {out_dir}")
    for c in failures[:20]:
        print(f"  FAIL case {c['case']}: {c['source']}/{c['subset']}/{c['header_style']}/{c['format']}")
        for e in c["errors"][:3]:
            print(f"      {e}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
