"""Ad-hoc probe of dirty-data scenarios — run once to classify (a)/(b)/(c)."""
import sys, io, json, tempfile
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.schema_detect import detect_dataframe
from aml.pipeline import Pipeline
from aml.sufficiency import SufficiencyEngine


def probe(name, df, expected_source=None, expected_slots=None):
    try:
        det = detect_dataframe(df, file=name)
        src = det.source_type
        slots = det.resolved_slots
        err = []
        if expected_source is not None and src != expected_source:
            err.append(f"source={src} (expected {expected_source})")
        if expected_slots is not None and expected_slots - slots:
            err.append(f"missing slots {expected_slots - slots} (got {slots})")
        status = "(c) SILENT-WRONG " + "; ".join(err) if err else "(a) ok"
        print(f"[{name:45s}] source={src:7s} slots={sorted(slots)} {status}")
        return det
    except Exception as e:
        print(f"[{name:45s}] (b) CRASH: {type(e).__name__}: {str(e)[:100]}")
        return None


bank_rows = pd.DataFrame({
    "txn_date": pd.date_range("2025-01-01", periods=60, freq="D"),
    "amount": [100.0 + i for i in range(60)],
    "recipient": [f"A{i%7}" for i in range(60)],
    "account": [f"ACC{i%5}" for i in range(60)],
})

# 1. Unix-epoch timestamp column
d1 = pd.DataFrame({
    "epoch_ts": [1700000000 + i * 86400 for i in range(60)],
    "amount": [500.0] * 60,
    "to_account": [f"A{i%7}" for i in range(60)],
    "account_no": [f"ACC{i%5}" for i in range(60)],
})
probe("unix_epoch_ts.csv", d1, expected_source="bank", expected_slots={"amount", "timestamp"})

# 2. large integer amounts, no symbols, no decimals
d2 = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=60),
    "sum": [48000 + i * 100 for i in range(60)],
    "beneficiary": [f"A{i%7}" for i in range(60)],
    "iban": [f"ACC{i%5}" for i in range(60)],
})
probe("integer_amounts.csv", d2, expected_source="bank", expected_slots={"amount", "timestamp"})

# 3. paren-negative + euro-style amounts
d3 = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=60),
    "amt": [f"${1000+i},{(i*73)%100:02d}" for i in range(30)] + [f"({1000+i},{(i*73)%100:02d})" for i in range(30)],
    "to": [f"A{i%7}" for i in range(60)],
    "acct": [f"ACC{i%5}" for i in range(60)],
})
det = probe("paren_negatives.csv", d3, expected_source="bank", expected_slots={"amount", "timestamp"})
if det is not None and det.normalized is not None:
    n_parsed = det.normalized["amount"].notna().sum()
    print(f"    -> paren negatives parsed: {n_parsed}/60")

# 4. DD/MM vs MM/DD ambiguity
d4 = pd.DataFrame({
    "date": ["01/02/2025"] * 60,
    "amount": [100.0] * 60,
    "to": [f"A{i%7}" for i in range(60)],
    "acct": [f"ACC{i%5}" for i in range(60)],
})
det = probe("ddmm_date.csv", d4, expected_source="bank")
if det is not None and det.normalized is not None:
    print(f"    -> first parsed date: {det.normalized['timestamp'].iloc[0]} (01/02/2025 -> expect 2025-02-01 if DD/MM)")

# 5. non-UTF8 (latin-1) CSV
raw = "fecha,monto,beneficiario,cuenta\n2025-01-01,100.0,José Martínez,CUENTA1\n" * 5
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
    fh.write(raw.encode("latin-1"))
    latin_path = fh.name
try:
    from aml.loaders import load_any
    loaded = load_any(latin_path)
    det = detect_dataframe(loaded.df, file=latin_path)
    print(f"[latin1.csv] (a) loaded ok, source={det.source_type}")
except Exception as e:
    print(f"[latin1.csv] (b) CRASH: {type(e).__name__}: {str(e)[:100]}")
finally:
    Path(latin_path).unlink()

# 6. no header row
d6 = pd.DataFrame({
    0: ["2025-01-01 10:00:00"] * 20 + ["2025-01-02 10:00:00"] * 20,
    1: ["100.00", "200.00"] * 20,
    2: [f"ACC{i%4}" for i in range(40)],
    3: [f"C{i%3}" for i in range(40)],
})
probe("no_header.csv", d6, expected_source="bank", expected_slots={"amount", "timestamp", "counterparty"})

# 7. bank-only pipeline: time_correlation verdict vs actual score
tmp = Path(tempfile.mkdtemp())
bank_only = tmp / "bank_only.csv"
pd.DataFrame({
    "txn_date": pd.date_range("2025-01-01", periods=500, freq="h"),
    "amount": [100.0 + (i % 50) * 7 for i in range(500)],
    "recipient": [f"A{i%9}" for i in range(500)],
    "account": [f"ACC{i%5}" for i in range(500)],
}).to_csv(bank_only, index=False)
pipe = Pipeline()
res = pipe.run([str(bank_only)])
tc = res.sufficiency["time_correlation"]
scored = res.model_scores.get("time_correlation")
print(f"[bank_only] time_correlation verdict={tc.status} reason='{tc.reason[:70]}' scores={0 if not scored else len(scored)}")
print(f"[bank_only] network={res.sufficiency['network'].status} benford={res.sufficiency['benford'].status} structuring={res.sufficiency['structuring'].status}")

# 8. CDR-only: activation matrix — H blocked, network?
cdr_only = tmp / "cdr_only.csv"
pd.DataFrame({
    "call_time": pd.date_range("2025-01-01", periods=500, freq="h"),
    "msisdn": [f"6391{i:08d}" for i in range(500)],
    "terminating": [f"6399{i:08d}" for i in range(500)],
    "duration_secs": [10.0 + (i % 60) for i in range(500)],
    "tower": [f"T{i%12}" for i in range(500)],
}).to_csv(cdr_only, index=False)
res2 = pipe.run([str(cdr_only)])
for m in ("time_correlation", "network", "statml", "benford", "structuring", "behavioral"):
    v = res2.sufficiency[m]
    print(f"[cdr_only] {m:16s} {v.status:10s} scores={len(res2.model_scores.get(m) or {})}")

# 9. sparse entities (<30 events/entity) — should BLOCK per plan
sparse = tmp / "sparse.csv"
pd.DataFrame({
    "txn_date": pd.date_range("2025-01-01", periods=80, freq="h"),
    "amount": [100.0] * 80,
    "recipient": [f"A{i%10}" for i in range(80)],
    "account": [f"ACC{i%8}" for i in range(80)],  # 8 entities ~10 events each
}).to_csv(sparse, index=False)
res3 = pipe.run([str(sparse)])
for m in ("time_correlation", "network", "statml"):
    v = res3.sufficiency[m]
    print(f"[sparse] {m:16s} {v.status:10s} reason='{v.reason[:80]}'")

# 10. duplicate columns: 'amount' and 'Amount '
d10 = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=60),
    "amount": [100.0] * 60,
    "Amount ": [999.0] * 60,
    "to": [f"A{i%7}" for i in range(60)],
    "acct": [f"ACC{i%5}" for i in range(60)],
})
det = probe("dup_amount_cols.csv", d10, expected_slots={"amount", "timestamp"})
if det is not None:
    amt = det.normalized["amount"]
    print(f"    -> amount used: {'Amount ' if (amt==999).any() else 'amount'}; is it 999? {(amt==999).sum()} rows")
