"""Generate the permanent dirty-data regression fixtures (data/dirty_samples/).

Every file in this folder is a regression fixture for the §3 robustness matrix:
schema detection and every downstream stage must (a) run correctly or produce a
degraded-but-honest result, never (b) crash, never (c) silently mislead.

Re-run after any schema-detect / loader / sufficiency change:
    python scripts/make_dirty_fixtures.py
    python scripts/dirty_audit.py          # classification table
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "dirty_samples"

DAYS = [f"2025-0{d % 6 + 1}-{(d % 27) + 1:02d} 1{d % 9}:{(d * 7) % 60:02d}:00" for d in range(80)]


def _bank_rows(amounts, parties=None, accounts=None, dates=None):
    parties = parties or [f"P{i % 7}" for i in range(len(amounts))]
    accounts = accounts or [f"ACCT{i % 5}" for i in range(len(amounts))]
    dates = dates or DAYS
    if len(dates) < len(amounts):
        dates = (dates * (len(amounts) // len(dates) + 1))[: len(amounts)]
    return [
        {"txn_date": d, "amount": a, "beneficiary": p, "account_no": n, "branch_code": f"BR{i % 4}"}
        for i, (d, a, p, n) in enumerate(zip(dates, amounts, parties, accounts))
    ]


N_OK = 200  # healthy-volume base row count: ~40 events/entity at %5 accounts


def _write(name: str, rows, fmt="csv", encoding="utf-8", sep=","):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with open(path, "w", encoding=encoding, newline="") as fh:
        if fmt == "csv":
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=sep)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        elif fmt == "raw":
            fh.write("\n".join(rows) + "\n")
    print(f"wrote {path}")


def main() -> None:
    # 1. renamed / mixed-case / whitespace headers (all within alias coverage)
    rows = _bank_rows([100.0 + i * 7.5 for i in range(N_OK)])
    renamed = [
        {" TxN Date ": r["txn_date"], "AMOUNT (USD)": r["amount"],
         "ReCiPiEnT aCcOuNt ": r["beneficiary"], "acct no": r["account_no"],
         "branch": r["branch_code"], "internal_ref": f"REF{i}"}
        for i, r in enumerate(rows)
    ]
    _write("renamed_mixed_case.csv", renamed)

    # 2. Spanish headers (language robustness)
    es = [
        {"fecha": r["txn_date"], "monto": r["amount"], "beneficiario": r["beneficiary"],
         "cuenta": r["account_no"], "sucursal": r["branch_code"]}
        for r in rows
    ]
    _write("headers_es.csv", es)

    # 3. unix-epoch timestamp column (seconds)
    epoch = [{"epoch_ts": 1735689600 + i * 7200, "amount": r["amount"],
              "to_account": r["beneficiary"], "account_no": r["account_no"]}
             for i, r in enumerate(rows)]
    _write("unix_epoch.csv", epoch)

    # 4. mixed date formats + junk within ONE column (incl. DD/MM and epoch ms)
    fmt_pool = [
        "2025-01-05 08:15:00", "05/01/2025 08:15:00", "2025-02-14T09:30:00Z",
        "28/01/2025", "1739567700000", "oops-not-a-date", "2025-03-01",
        "01/12/2025 22:05:00",
    ]
    mixed = [{"when": fmt_pool[i % len(fmt_pool)], "amount": r["amount"],
              "beneficiary": r["beneficiary"], "account_no": r["account_no"]}
             for i, r in enumerate(rows)]
    _write("mixed_dates.csv", mixed)

    # 5. currency-mess amounts: symbols, thousands, parens-negatives,
    #    comma-decimal, codes, junk
    amt_strs = ["$1,234.56", "(500.00)", "1 200,50", "€77.00", "1.234,56",
                "12.5 USD", "N/A", "—", "0.00", "-1,000.25", "9,999.99", "500"]
    cur = [{"date": r["txn_date"], "amt": amt_strs[i % len(amt_strs)],
            "to": r["beneficiary"], "iban": r["account_no"]}
           for i, r in enumerate(rows)]
    _write("currency_mess.csv", cur)

    # 6. missing-value hell: '' / NaN / N/A / - / 0-sentinel across fields
    miss = []
    for i, r in enumerate(rows):
        miss.append({
            "txn_date": ["", "NaN", r["txn_date"], "N/A"][i % 4],
            "amount": ["", "N/A", r["amount"], "0"][i % 4],
            "beneficiary": r["beneficiary"] if i % 3 else "",
            "account_no": r["account_no"] if i % 5 else "-",
        })
    _write("missing_values.csv", miss)

    # 7. duplicate / near-duplicate columns ('amount' AND 'Amount ')
    dup = [{"date": r["txn_date"], "amount": r["amount"], "Amount ": r["amount"] + 500.0,
            "to": r["beneficiary"], "acct": r["account_no"]}
           for r in rows]
    _write("dup_columns.csv", dup)

    # 8. no header row at all — content-probe fallback must engage
    raw8 = []
    for i, r in enumerate(_bank_rows(
            [50.25 + i * 3.1 for i in range(N_OK)],
            parties=[f"A{i % 5:04d}" for i in range(N_OK)])):
        raw8.append(f"{r['txn_date']},{r['amount']},{r['beneficiary']},{r['account_no']}")
    _write("no_header.csv", raw8, fmt="raw")

    # 9. latin-1 (cp1252) encoded file with accented names
    rows9 = _bank_rows([200.0 + i for i in range(N_OK)])
    latin = [{"fecha": r["txn_date"], "monto": r["amount"], "beneficiario": "José Martínez",
              "cuenta": r["account_no"]} for r in rows9]
    _write("latin1.csv", latin, encoding="cp1252")

    # 10. semicolon-delimited (inconsistent delimiter)
    _write("semicolon.csv", _bank_rows([150.0 + i for i in range(N_OK)]), sep=";")

    # 11. sparse entities (<30 events/entity -> C/D/E BLOCKED per §1B)
    sparse = [{"txn_date": f"2025-0{d % 4 + 1}-{(d % 27) + 1:02d} 1{d % 9}:00:00",
               "amount": 100.0 + i * 3, "beneficiary": f"P{i % 10}",
               "account_no": f"ACCT{i % 6}"} for i, d in enumerate(range(72))]
    _write("sparse.csv", sparse)

    # 12-14. single-source combinations
    _write("bank_only.csv", _bank_rows([5000.0 + i * 13 for i in range(400)]))
    cdr = [{"call_time": f"2025-0{d % 6 + 1}-{(d % 27) + 1:02d} 2{d % 4}:{(d * 3) % 60:02d}:00",
            "msisdn": f"6391{i % 10:08d}", "terminating_number": f"6399{i % 9:08d}",
            "duration_secs": str(10.0 + d % 60), "tower": f"T{d % 12}"}
           for i, d in enumerate(range(400))]
    _write("cdr_only.csv", cdr)
    soc = [{"created_utc": f"2025-0{d % 6 + 1}-{(d % 27) + 1:02d} 3{d % 9}:00:00",
            "username": f"@user.{i % 10:03d}", "hashtags": f"#{d % 7}",
            "full_name": f"Name {d % 9}"} for i, d in enumerate(range(400))]
    _write("social_only.csv", soc)


if __name__ == "__main__":
    main()