"""Research-format emission — the same scenario, many real-world layouts.

The AML-detection literature converges on a small set of interchange formats
(Elliptic/IBM-AML CSV, AMLSim JSONL streams, key=value audit logs, and
human-narrative log lines). ``build_scenario`` gives one deterministic fact
sheet; this module re-renders it so every format describes the SAME entities,
names, account/phone/handle identifiers, amounts and storylines. A detector
that really understands untabular data must recover comparable evidence from
each layout — that is what ``scripts/research_formats_validate.py`` measures.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .emit import BANK_HEADERS, CDR_HEADERS, SOCIAL_HEADERS, build_scenario

# ---------------------------------------------------------------------------
# renderers — one canonical dict per source row, then serialised per format
# ---------------------------------------------------------------------------
def _pick_headers(source: str, seed: int = 7) -> Dict[str, str]:
    """Deterministic header alias map per source (schema-alias vocabulary)."""
    rng = random.Random(seed)
    table = {"bank": BANK_HEADERS, "cdr": CDR_HEADERS, "social": SOCIAL_HEADERS}[source]
    return table[rng.choice(list(table.keys()))]


def _row_dicts(scenario: Dict, source: str, seed: int = 7) -> List[dict]:
    """Canonical rows with header names routed through schema aliases, so the
    SAME dynamic detection fires whether the row lands in a CSV, JSONL or kv.
    """
    cols = {
        "bank": ["ts", "account", "direction", "amount", "counterparty_acc", "branch", "name"],
        "cdr": ["ts", "msisdn", "partner_phone", "tower", "duration", "name"],
        "social": ["ts", "handle", "mentions", "geo", "name"],
    }[source]
    headers = _pick_headers(source, seed)
    out = []
    for _, r in scenario[f"{source}_rows"].iterrows():
        rec = {headers.get(c, c): (str(pd.Timestamp(r[c])) if c == "ts" else r[c]) for c in cols}
        out.append(rec)
    return out


def render_xlsx_csv(scenario: Dict) -> Dict[str, pd.DataFrame]:
    """IBM-style transactional CSV kept in the generator's header vocabulary."""
    frames: Dict[str, pd.DataFrame] = {}
    for src in ("bank", "cdr", "social"):
        frames[src] = pd.DataFrame(_row_dicts(scenario, src))
    return frames


def render_jsonl(scenario: Dict) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for src in ("bank", "cdr", "social"):
        out[src] = _row_dicts(scenario, src)
    return out


def render_kvlog(scenario: Dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for src in ("bank", "cdr", "social"):
        lines = []
        for rec in _row_dicts(scenario, src):
            toks = []
            for k, v in rec.items():
                if isinstance(v, str) and (" " in v):
                    v = '"' + v + '"'
                toks.append(f"{k}=" + str(v))
            lines.append(" ".join(toks))
        out[src] = lines
    return out


def render_prose(scenario: Dict) -> Dict[str, List[str]]:
    """Human-narrative log lines exercising the typed-token extractor."""
    bank, cdr, social = [], [], []
    for _, r in scenario["bank_rows"].iterrows():
        verb = "deposit" if r["direction"] == "credit" else "transfer"
        dst = r["counterparty_acc"] or "(internal)"
        bank.append(f"{str(pd.Timestamp(r['ts']))} {verb} from {r['account']} "
                    f"to {dst} amount {r['amount']:.2f} USD")
    for _, r in scenario["cdr_rows"].iterrows():
        cdr.append(f"{str(pd.Timestamp(r['ts']))} call from {r['msisdn']} to "
                   f"{r['partner_phone']} at tower {r['tower']} "
                   f"duration {r['duration']:.0f} seconds")
    for _, r in scenario["social_rows"].iterrows():
        social.append(f"{str(pd.Timestamp(r['ts']))} post by {r['handle']} "
                      f"mentions {r['mentions'] or 'no one'}")
    return {"bank": bank, "cdr": cdr, "social": social}


# ---- emission wrapper -----------------------------------------------------
_FILENAME = {"bank": "bank_export", "cdr": "cdr_export", "social": "social_export"}


def emit_formats(
    out_dir: Path,
    seed: int = 42,
    surprise_seed: int = 99,
    n_background: int = 90,
    n_per_storyline: int = 3,
    n_surprise: int = 12,
) -> Dict[str, Path]:
    """Write one folder per format, each with a full sources/ + the SAME keys.

    Layout::

        out_dir/
          answer_key.json            (shared cross-format)
          answer_key_surprise.json   (shared cross-format)
          csv/   sources/{bank,cdr,social}_export.csv
          jsonl/ sources/{...}.jsonl
          kvlog/ sources/{...}.log
          prose/ sources/{...}.log
    """
    out_dir = Path(out_dir)
    scenario = build_scenario(seed=seed, surprise_seed=surprise_seed,
                              n_background=n_background,
                              n_per_storyline=n_per_storyline,
                              n_surprise=n_surprise)

    files: Dict[str, Path] = {}

    # CSV layout
    csv_frames = render_xlsx_csv(scenario)
    csv_dir = out_dir / "csv"
    for src, df in csv_frames.items():
        folder = csv_dir / "sources"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{_FILENAME[src]}.csv"
        df.to_csv(p, index=False)
        files[f"csv_{src}"] = p

    # JSONL layout
    jsonl_dir = out_dir / "jsonl"
    json_recs = render_jsonl(scenario)
    for src, recs in json_recs.items():
        folder = jsonl_dir / "sources"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{_FILENAME[src]}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, default=str) + "\n")
        files[f"jsonl_{src}"] = p

    # key=value logs
    kv_dir = out_dir / "kvlog"
    for src, lines in render_kvlog(scenario).items():
        folder = kv_dir / "sources"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{_FILENAME[src]}.log"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files[f"kvlog_{src}"] = p

    # honest prose logs
    prose_dir = out_dir / "prose"
    for src, lines in render_prose(scenario).items():
        folder = prose_dir / "sources"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{_FILENAME[src]}.log"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files[f"prose_{src}"] = p

    # SHOULD the answer keys be shared? They describe the *scenario*, so yes.
    akey = out_dir / "answer_key.json"
    akey.write_text(json.dumps({"entities": scenario["annotations"]}), encoding="utf-8")
    skeys = {pid_: scenario["annotations"][pid_] for pid_ in scenario["surprise_ids"]}
    (out_dir / "answer_key_surprise.json").write_text(
        json.dumps({"entities": skeys}), encoding="utf-8")
    files["answer_key"] = akey
    files["surprise_key"] = out_dir / "answer_key_surprise.json"
    return files


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    root = Path(__file__).resolve().parents[3] / "data" / "research_formats"
    f = emit_formats(root)
    print("wrote:", sorted(f.keys()))