# TraceWeave — Team Briefing & Revision Doc

Internal reference for the team. This is not the judge-facing pitch deck
(`assets/AML_Pitch_Deck_v2.pptx`) — that's curated for time. This is
everything, for study and Q&A prep. If a number appears in this doc, it was
verified by actually running the code, not estimated.

---

## Table of Contents

1. [30-Second Version](#1-30-second-version)
2. [The Problem](#2-the-problem)
3. [How It Works — Architecture](#3-how-it-works--architecture)
4. [The Canonical Data Schema](#4-the-canonical-data-schema)
5. [The Sufficiency Engine](#5-the-sufficiency-engine)
6. [The Models](#6-the-models)
7. [Synthetic Data & Baseline Evaluation](#7-synthetic-data--baseline-evaluation)
8. [Real-World Validation (SAML-D)](#8-real-world-validation-saml-d)
9. [Why PaySim Didn't Work](#9-why-paysim-didnt-work)
10. [Bugs Found and Fixed](#10-bugs-found-and-fixed)
11. [Legal-Basis Provenance](#11-legal-basis-provenance)
12. [Chain-of-Custody Sealing](#12-chain-of-custody-sealing)
13. [Test Suite & Robustness](#13-test-suite--robustness)
14. [Technology Stack](#14-technology-stack)
15. [Project Structure](#15-project-structure)
16. [Glossary](#16-glossary)
17. [Master Numbers Cheat Sheet](#17-master-numbers-cheat-sheet)
18. [Known Limitations — Say These Yourself](#18-known-limitations--say-these-yourself)
19. [Team Roles & Pitch Plan](#19-team-roles--pitch-plan)
20. [Anticipated Q&A](#20-anticipated-qa)

---

## 1. 30-Second Version

**TraceWeave** reads messy bank, phone-call (CDR), and social-media exports,
figures out on its own what each column means, links the same person across
all three sources, and ranks people by how suspicious their combined
behavior looks — with a plain-English reason for every flag. Its
distinguishing feature: it **never fakes a result**. If the data can't
support a model (not enough sources, not enough volume), it says so
explicitly instead of guessing. It's been validated against a real,
external, independently-labeled anti-money-laundering dataset, not just its
own synthetic generator, and every analysis run is legally-attributed and
tamper-evidently sealed.

---

## 2. The Problem

- The UN estimates **2–5% of global GDP** (~**$0.8–2.0 trillion/year**) is
  laundered. Less than 1% of laundered proceeds are ever seized (FATF).
- Three failure modes in existing tools:
  1. **Cross-source blindness** — a call 19 minutes before a transfer is
     invisible to a bank-only monitor. The link *between* sources is the
     signal.
  2. **Schema fragility** — `txn_date` vs `posted_at` vs `call_time`; a
     hardcoded parser breaks on the first new export format.
  3. **Coverage dishonesty** — tools silently run with missing data instead
     of telling the analyst what's absent.

---

## 3. How It Works — Architecture

```
sources/*.csv · *.json · *.jsonl · *.txt · *.log
        │
        ▼
[Stage A] Dynamic Schema Detection   (RapidFuzz header match + content probe)
        │
        ▼
[Stage B] Entity Resolution          (blocking + Jaro-Winkler fuzzy name match)
        │
        ▼
   Unified long-format dataframe (see §4)
        │
   ┌────┴──────────┬───────────────┬───────────────────┐
   ▼               ▼               ▼                   ▼
[Stage C]       [Stage D]     [Stage E]          [Stage H]
Time            Network        Stat/ML            Benford · structuring ·
Correlation     (graph)        outliers           behavioral · chain
   │               │               │                   │
   └───────────────┴───────────────┴───────────────────┘
                        │
                 [Stage F] Score Fusion & Ranking (Borda / meta-learner)
                        │
                 [Stage G] Explanation Generator
                        │
                 Report: Markdown · HTML · PDF · JSON + Desktop GUI
```

**Stage-by-stage, in plain language:**

1. **Schema detection** — every column header is fuzzy-matched (RapidFuzz,
   threshold ~80) against alias dictionaries; if no header matches, content
   is probed instead (does it parse as a date 90%+ of the time? Match a
   phone regex? Look like currency?). Source type (bank/CDR/social) falls
   out as a side effect of which fields resolved.
2. **Entity resolution** — ties one person across sources using shared
   identifiers where they exist, and Jaro-Winkler fuzzy name matching where
   they don't (cross-source only — same-source name collisions are never
   merged, to avoid blending two different people's histories).
3. **Unified dataframe** — every stage from here on reads/writes the exact
   same table (see §4).
4. **Sufficiency check** — before anything runs, decide honestly what the
   data can support (see §5).
5. **Models run in parallel** — only non-BLOCKED ones.
6. **Fusion** — scores combined into one ranked list (Borda-count by
   default).
7. **Report** — plain-English findings + charts, in 4 formats.

---

## 4. The Canonical Data Schema

Every event, from any source, becomes one row in this table:

| Column | Type | Meaning |
|---|---|---|
| `entity_id` | str | resolved person ID |
| `event_type` | str | `transaction` / `call` / `post` |
| `timestamp` | datetime | when it happened |
| `source` | str | `bank` / `cdr` / `social` |
| `amount` | float (nullable) | value — bank only |
| `counterparty_id` | str (nullable) | the other party |
| `location` | str (nullable) | branch / tower / geo-tag |
| `direction` | str (nullable) | `in` / `out` |
| `duration` | float (nullable) | call seconds |

**What each source can fill:**

| Slot | Bank | CDR | Social |
|---|---|---|---|
| timestamp | yes | yes | yes |
| amount | **yes** | no | no |
| counterparty | yes (account) | yes (phone) | partial (handle/hashtag) |
| location | branch | tower | geo-tag |
| direction | yes | no | no |
| duration | no | yes | no |

---

## 5. The Sufficiency Engine

The single most important design idea in the project. For every model, it
computes what's populated vs. what's required and returns one of three
verdicts, each with a human-readable, actionable reason:

- **SUPPORTED** — runs normally.
- **DEGRADED** — runs on a reduced feature set, and says exactly what's
  missing and why (e.g. "single source: only internal correlation is
  possible").
- **BLOCKED** — doesn't run at all, and says exactly what would unblock it
  (e.g. "add a CDR or bank file to enable it").

**Important nuance to know cold**: the volume check
(`min_events_per_entity = 30`) is a **population average**
(`total_rows / total_entities`), **not a per-entity check**. This means a
handful of individually data-rich entities can still get blocked if the
overall population is dominated by low-activity entities — this was
actually *discovered* during real-data validation (see §8) and is a known,
disclosed architectural characteristic, not a hidden bug.

---

## 6. The Models

Main pipeline runs **7 model families** in parallel (only if not BLOCKED):

| Model | What it catches | One-line mechanism |
|---|---|---|
| `time_correlation` | Cross-source co-location (a call, then a transfer) | `pandas.merge_asof` joins events within a tolerance window (default 25 min) |
| `network` | Fan-out/fan-in hubs, mule-like structure | OddBall-style degree-vs-weight deviation over each person's ego-network |
| `statml` | General behavioral outliers | Isolation Forest over a per-entity feature vector (counts, amounts, fan-in/out, timing entropy) |
| `benford` | Manufactured/fabricated amounts | Chi-square deviation of first-digit distribution from Benford's Law |
| `structuring` | Smurfing (splitting to dodge reporting limits) | Counts transactions just below a threshold (default ₹/$10,000) |
| `behavioral` | Dormant-to-active flips, silence after a big transfer | Person- and population-relative regime-change rules |
| `chain` | Layering (money passed through several accounts fast) | DFS over consecutive "big" debits within a hop window — **this is the one with the recursion bug, see §10** |

A larger **model zoo** (candidates like Extended IF, LOF, OCSVM, HBOS, GMM,
PCA, Mahalanobis, autoencoder, KDE, LSTM autoencoder, TGN graph
autoencoder) exists for research/comparison scripts but isn't part of the
default pipeline.

**Fusion**: default is **Borda-count rank aggregation** with a max-bonus
(so one perfect single-model alarm isn't buried by consensus). Alternatives:
top-2 weighted, rank-average, score-mean, weighted-sum, and a learned
RandomForest/XGBoost meta-learner (falls back to Borda if the trained
asset is missing).

---

## 7. Synthetic Data & Baseline Evaluation

The project's own generator produces two tiers of ground truth:

- **Predefined storylines** — scripted anomaly types with documented
  ground truth (used to verify each model works at all).
- **Randomized surprise injector** — writes only to a secret answer key
  (`data/answer_key_surprise.json`), so flagging one of these entities is
  genuine generalization, not a lookup.

**Anomaly types**: `colocation`, `dormant_flip`, `structuring`, `fan_io`,
`silence`, `post_burst`, `chain`, `background`.

**Current baseline (synthetic, self-generated data)** —
`results/current_baseline/eval.json`:

| Metric | Value |
|---|---|
| recall@5 | 0.190 |
| precision@5 | 0.800 |
| recall@10 | 0.429 |
| precision@10 | 0.900 |
| Entities | 431 |
| Anomalies evaluated | 21 |

Per-type recall@10 (main key): `structuring`=0.67, `silence`=0.67,
`chain`/`colocation`/`dormant_flip`/`fan_io`/`post_burst`=0.33 — every type
recovers *some* signal here.

Surprise hold-out (the real generalization test): recall@10=0.071,
precision@10=0.100, `structuring`=1.00, everything else=0.00. **This is the
number that matters most and the one to lead with if pressed** — the main
key shows the models can learn known storylines, the surprise key shows
they mostly haven't generalized past `structuring` yet.

**Say this plainly if asked**: these numbers are graded on the project's
own generator. That's why real-world validation (§8) matters — it's the
number that isn't self-graded. (These figures were re-benchmarked by
re-running `eval.py` against the current code — an earlier committed
`results/` snapshot had gone stale after later bug fixes and understated
both the main-key recall and per-type coverage.)

**Dirty-data robustness**: 14 deliberately messy fixtures (renamed
headers, Spanish headers, unix epochs, mixed date formats, currency mess,
missing values, duplicate columns, no header at all, latin-1 encoding,
semicolon delimiters, sparse volume, single-source-only files). **14/14
pass** (see §10 for the bug that briefly broke this).

---

## 8. Real-World Validation (SAML-D)

This is the headline new work this session, and the part most likely to
get hard questions — know this section cold.

**What SAML-D is**: a real, peer-reviewed AML transaction dataset (Oztas
et al., *IEEE ICEBE 2023* — "Enhancing Anti-Money Laundering: Development
of a Synthetic Transaction Monitoring Dataset"). 9,504,852 transactions,
9,873 labeled suspicious (**0.104%** true fraud rate), 28 typologies
(Structuring, Smurfing, Layered_Fan_In/Out, Behavioural_Change, Bipartite,
Cash_Withdrawal, Deposit-Send, and more). This is the *same paper* our own
`PLAN.md` already cited as design precedent, before this validation ever
ran. License: CC BY-NC-SA 4.0 (non-commercial — fine for a hackathon,
worth knowing if this ever gets commercialized).

**Why not just use the full 9.5M rows?** Two hard constraints forced a
sample: (1) runtime — entity resolution is a Python-level loop, not
vectorized; (2) the sufficiency engine's population-average volume check
(§5) means a naive random sample gets diluted with too many low-activity
accounts and everything gets `BLOCKED`. So the sample is deliberately
**typology-stratified** (keeps rare typologies like Smurfing from being
drowned out) and **volume-aware** (background accounts chosen from
accounts that *already* naturally clear the 30-events bar, not padded
artificially).

**We ran TWO validation passes, on purpose** — because a single
"precision@10 = 1.00" number does not survive scrutiny on its own:

| | Curated | Harder |
|---|---|---|
| Total scored entities | 4,625 | 7,929 |
| Ground-truth positives resolved | 1,005 | 698 |
| **Fraud prevalence** | **21.7%** | **8.8%** |
| SAML-D's *true* full-dataset rate | 0.104% (for reference — neither sample reaches this) | |

**Precision/recall by k** (both real, reproducible — `scripts/real_dataset_samld.py` and `--harder`):

| k | Curated precision | Harder precision | Curated recall | Harder recall |
|---|---|---|---|---|
| 5 | 1.000 | 1.000 | 0.005 | 0.007 |
| 10 | **1.000** | **0.800** | 0.010 | 0.011 |
| 20 | 0.950 | 0.500 | 0.019 | 0.014 |
| 50 | 0.700 | 0.380 | 0.035 | 0.027 |
| 100 | 0.590 | 0.270 | 0.059 | 0.039 |
| 200 | 0.565 | 0.225 | 0.112 | 0.064 |
| 500 | 0.372 | 0.168 | 0.185 | 0.120 |
| 1000 | 0.264 | 0.154 | 0.263 | 0.221 |

**The story, precisely**: precision@10 drops from **1.00 → 0.80** as
prevalence drops from 21.7% → 8.8%. That's **graceful degradation**, not
collapse — real ranking signal responding predictably to a harder test,
not a lucky number at one cherry-picked setting.

**Why is precision even that high — is it fake?** No, it's real and
reproducible, but it's inflated by design, and you must say so:
- Precision is *extremely* sensitive to base rate. At 21.7%/8.8% fraud
  concentration (vs. SAML-D's real 0.104%), hitting several true
  positives at the very top of a ranking is far easier than it would be
  in a real, ~200x-more-diluted population.
- The chosen fraud accounts were the **highest-volume account within
  each typology** (had to be, to individually clear the 30-event bar) —
  meaning this measures the most clearly-patterned, "textbook" version of
  each fraud type, likely an optimistic ceiling, not an average case.
- The more portable, honest number is **lift over random**: **4.6x**
  better than random ranking at k=10 in the curated pass. Lift survives
  the base-rate inflation better than a bare precision number does.

**The one-line answer if a judge pushes**: *"In a stratified validation
sample against real, external SAML-D data, our top-10 ranked accounts
were all true positives at one setting, dropping to 8/10 at a harder
setting — a 4.6x lift over random. That's evidence the ranking carries
real signal, not a real-world deployment precision claim — real fraud is
far rarer, so absolute precision would be lower in practice."*

---

## 9. Why PaySim Didn't Work

Worth knowing so you don't reach for it as a second validation dataset.
**PaySim** (Kaggle `ealaxi/paysim1`, CC BY-SA 4.0, 6,362,620 rows, 8,213
fraud rows = 0.13%) is a mobile-money simulator. It's structurally
incompatible with TraceWeave's entity-centric design: nearly every sender
account appears **exactly once** (mean 1.0–1.5 transactions/account) — it's
a one-shot-per-transaction fraud-classification dataset, not an
accumulating-history-per-account dataset the way SAML-D and our own
synthetic generator are. No amount of resampling fixes this; the
`min_events_per_entity` gate can never be cleared. Validating against it
was still valuable — it's what surfaced the `nameOrig`/`actor_id` schema
bug (§10, bug 1).

---

## 10. Bugs Found and Fixed

All four found via **real-world data validation** — none were caught by
the synthetic-data test suite, because synthetic fixtures never
accidentally hit these edge cases. This is the concrete proof that
external validation is worth doing, not just a nice-to-have.

**Bug 1 — schema detection lost the entity dimension on real sender/payer columns**
`src/aml/schema.py`. Content-probe fallback for ID-shaped columns could
only ever produce the `counterparty` slot, never `actor_id` — that slot
could *only* be reached via an exact header alias match. PaySim's
`nameOrig` column didn't match any alias, fell through to content-probe,
collided with `nameDest`, and the entity dimension silently vanished (every
entity-scored model reported BLOCKED). **Fix**: added sender/originator/
payer-style aliases (`nameorig`, `sender_account`, `payer`, `originator`,
`remitter`, `from_account`, etc.) to `ACTOR_ID_ALIASES`.

**Bug 2 — chain/layering detector crashed on any transaction cycle**
`src/aml/models.py`, the `chain` model's DFS. No cycle guard — if account A
pays B and B pays A back within the hop window (extremely common in real
banking: refunds, mutual dealings), it recursed forever and crashed the
**entire pipeline run** with `RecursionError`, not just that one model.
This was silently causing 10 of 13 test failures. **Fix**: one-line guard,
`if dst in path: continue`, skipping any node already on the current DFS
path.

**Bug 3 — currency-name columns misclassified as person names**
`src/aml/schema_detect.py`. The content-probe's "two alphabetic words →
name" heuristic (rule 8) matched low-cardinality categorical strings like
"UK pounds" / "US Dollar" just as well as real names like "John Smith" —
risking spurious cross-source identity merges on real multi-source data.
**Fix**: added a distinctness requirement (≥5 unique values, >15%
distinctness in the sample) alongside the pattern match.

**Bug 4 — a dirty-data fixture itself was broken, not the pipeline**
`scripts/make_dirty_fixtures.py`. `cdr_only.csv`'s hour field was built as
the literal string `f"2{d % 9}"`, producing `"20"`–`"28"` — but `"24"`–`"28"`
aren't valid clock hours. ~55% of that fixture's timestamps silently
failed to parse and got dropped, dragging per-entity volume below the
threshold and causing a **false regression** (network/statml wrongly
`BLOCKED` instead of `DEGRADED`). The pipeline was behaving correctly on
genuinely malformed data — the fixture generator had the bug. **Fix**:
`d % 9` → `d % 4`. Dirty-audit restored to a genuine 14/14.

*(A test-hang investigation also happened this session and turned out
NOT to be a real bug — see §13.)*

---

## 11. Legal-Basis Provenance

Every analysis run now records **who authorized it and under what legal
basis** (warrant #, internal case reference, regulatory request), captured
in the GUI before pressing Analyze, and — critically — **printed directly
into the exported report itself** (Markdown/HTML/PDF/JSON), not just
hidden metadata. If left blank, it prints **"not recorded"** honestly
instead of a fake default, matching the sufficiency engine's own
never-hide-an-absence philosophy.

**Why this matters for the Punjab Police angle**: it directly answers
"what gives you the right to analyze this data" — the framing is
*"TraceWeave doesn't collect data; it's an analyst-side triage layer over
data already lawfully obtained (a production order, a court order, a
compliance channel), and every report carries its own record of that
authorization."*

**Honest caveat to state yourself**: none of us are lawyers. We did not
verify exact statutory citations (PMLA/STR-CTR/FIU-IND reporting, BNSS
§94 production-of-documents, DPDP Act 2023 law-enforcement exemptions)
closely enough to cite specific section numbers in front of actual police
officers. Keep the framing at the level above; don't invent citations.

---

## 12. Chain-of-Custody Sealing

Every archived run is **SHA-256 hashed file-by-file** (every report file,
chart, and copied source file — not `index.json` itself, which is a
mutable summary rewritten after sealing) and **chained to the previous
run's seal** — the same append-only-ledger idea a blockchain uses, without
needing one.

- `verify_run(run_id)` — recomputes hashes, flags exactly which file(s)
  don't match what was sealed (`OK` vs `TAMPERED`).
- `verify_chain()` — walks *every* archived run in order and confirms each
  one's seal correctly chains to the previous one's, catching tampering
  with the ledger's own recorded hashes, not just the underlying files.
- Live in the GUI: History dialog → select a run → **"Verify integrity"**
  button → "Verified: N/N files match their sealed hashes" or
  "TAMPERED — report.md".

**Honest limitation to state yourself**: this is a *local* integrity
check. It proves nothing has been edited since sealing, and that history
hasn't been quietly rewritten. It does **not** cryptographically prove
authorship to a third party the way real notarization/signing with a key
you don't control would. Say "tamper-evident," never "tamper-proof."

---

## 13. Test Suite & Robustness

| Check | Result |
|---|---|
| Full pytest suite | **100/100 passing** |
| Dirty-data audit (14 fixtures) | **14/14 pass** |
| Network-dependent test (`test_internet_samples.py`) | excluded from CI count — needs live internet, unrelated to any bug |

**GUI test investigation** (worth knowing, but it's a non-finding):
`test_analyze_flow_enables_and_persists` appeared to hang for many
minutes at one point. Re-ran it 6 times in isolation and once in the full
suite afterward — every single time it passed cleanly in 3–4 seconds. It
only ever hung when this machine was under heavy simultaneous load
(downloading/processing ~1.9GB of Kaggle datasets at the time). There is
no infinite loop in the code; the test's polling bound was hardened from
8s to 30s anyway as cheap insurance against a slow/loaded machine, not
because a real bug was found.

---

## 14. Technology Stack

| Layer | Tech | Why |
|---|---|---|
| Language | Python 3.11+ | whole pipeline, GUI, tooling |
| Data | pandas, NumPy, SciPy | canonical dataframe, `merge_asof`, feature engineering |
| ML | scikit-learn | Isolation Forest, LOF, OCSVM, GMM, PCA, MinCovDet, meta-learner |
| Fuzzy matching | RapidFuzz | schema detection + entity resolution |
| Graph (optional) | networkx | ego-graph rendering, PageRank, community detection |
| Deep models (optional) | PyTorch | LSTM autoencoder, TGN — skip gracefully if absent |
| Charts | matplotlib | every report chart |
| GUI | PySide6 (Qt) | drag-and-drop desktop app |
| PDF | reportlab | printable reports |
| Storage | platformdirs | persistent run archive across restarts/reinstalls |

No API keys, no external services, no network calls in the core pipeline
— fully offline.

---

## 15. Project Structure

```
main_app.py               desktop GUI (drag-drop, History, Verify integrity)
run_pipeline.py           CLI: files in -> report out
run_demo.py               one-command demo (regenerate data + run + report)
eval.py                   evaluation harness (recall@k, ablation, surprise)

src/aml/
  schema.py               canonical slots + alias dictionaries
  schema_detect.py         Stage A: header fuzzy match + content probe
  entity_resolve.py        Stage B: cross-source identity linking
  sufficiency.py           SUPPORTED/DEGRADED/BLOCKED engine
  pipeline.py              orchestrates everything -> PipelineResult
  models.py                all 7 model families + fusion
  fusion_meta.py           learned RF/XGBoost meta-learner
  insights.py              plain-language evidence extraction
  visuals.py               matplotlib report charts
  report.py                Markdown/HTML/PDF/JSON generation + provenance
  storage.py               persistent archive + chain-of-custody sealing
  data_generator/           synthetic population + storylines + surprise set

scripts/
  real_dataset_paysim.py   PaySim validation (structural-mismatch finding)
  real_dataset_samld.py    SAML-D validation (--harder for the diluted pass)
  make_dirty_fixtures.py   generates the 14 dirty-data fixtures
  dirty_audit.py           runs and grades all 14

tests/                    96 tests
data/dirty_samples/       14 messy-data fixtures
data/real_datasets/       downloaded Kaggle CSVs (gitignored, ~1.9GB)
assets/                   pitch deck + chart-generation scripts
```

---

## 16. Glossary

- **Sufficiency engine** — the SUPPORTED/DEGRADED/BLOCKED honesty layer
  (§5).
- **Entity resolution** — linking the same person across sources without a
  shared ID.
- **Prevalence** (in a validation sample) — % of scored entities that are
  actually true positives. Different from SAML-D's real dataset-wide rate.
- **Lift** — how many times better than random guessing a ranking performs
  at a given k.
- **Provenance** — the legal-basis/analyst record attached to a run (§11).
- **Chain of custody / sealing** — the SHA-256 hash-and-chain integrity
  layer (§12).
- **Borda count** — the default score-fusion method; rank aggregation
  across models.
- **DEGRADED vs BLOCKED** — DEGRADED still runs (on less data, with a
  warning); BLOCKED doesn't run at all.

---

## 17. Master Numbers Cheat Sheet

*(everything in one place for quick recall)*

| Number | What it is |
|---|---|
| 100/100 | pytest suite passing |
| 14/14 | dirty-data fixture audit |
| 30 | `min_events_per_entity` threshold (population average) |
| 7 | main-pipeline model families |
| 21.7% | fraud prevalence, curated SAML-D validation |
| 8.8% | fraud prevalence, harder SAML-D validation |
| 0.104% | SAML-D's real, full-dataset fraud rate |
| 1.00 → 0.80 | precision@10, curated → harder |
| 4.6x | lift over random @ k=10, curated pass |
| 9,504,852 | SAML-D total transactions |
| 9,873 | SAML-D labeled-suspicious transactions |
| 28 | SAML-D typologies |
| 6,362,620 | PaySim total transactions (structurally unusable) |
| 4 | real bugs found and fixed via real-data validation |
| 0.190 / 0.429 | recall@5 / recall@10, synthetic baseline (main key) |
| 0.800 / 0.900 | precision@5 / precision@10, synthetic baseline (main key) |
| 0.071 / 0.100 | recall@10 / precision@10, surprise hold-out (generalization) |
| $0.8–2.0T | estimated global money laundered per year |
| <1% | fraction of laundered proceeds ever seized (FATF) |

---

## 18. Known Limitations — Say These Yourself

Better you say these first than a judge finds them:

1. **Synthetic-only in primary development** — real validation exists now
   (§8), but it's a supplementary check, not the whole evaluation.
2. **Generalization is the real gap.** Main-key recall@10 is 0.429 with every
   type recovering some signal, but the surprise hold-out — the true
   generalization test — is 0.00 on 5 of 7 types. Say this one yourself
   before a judge digs for it: `structuring` generalizes, most else doesn't yet.
3. **Entity resolution is name-based only** — no phonetic matching, no
   OSINT-style deanonymization of unrelated pseudonymous handles.
4. **Batch, not real-time.**
5. **No web/API deployment** — desktop + CLI only, currently.
6. **Chain-of-custody is tamper-evident, not tamper-proof** (§12).
7. **Legal-basis framing is not lawyer-verified** (§11).
8. **Precision numbers in §8 are prevalence-inflated** by validation-sample
   design — know exactly why (§8) if asked.

---

## 19. Team Roles & Pitch Plan

See the full run-of-show, timing, and Q&A ownership table already agreed —
summarized here for revision:

| Role | Owns |
|---|---|
| A — Engine | architecture, sufficiency engine, models, fusion; live adaptivity demo |
| B — Data & Validation | schema detection, dirty-data robustness, SAML-D validation story |
| C — Trust & Compliance | legal-basis provenance, chain-of-custody; live "Verify integrity" demo |
| D — Narrative | problem framing, open/close, timekeeping, Q&A traffic control |

~5-minute run of show: Problem (D) → Architecture (A) → Features (A/B) →
Live demo 1: adaptivity (A) → Live demo 2: verify integrity (C) →
Validation slide (B) → Compliance slide (C) → Close (D) → Q&A.

**Reminder**: pre-stage one already-tampered run in History before the
pitch — never tamper with a file live.

---

## 20. Anticipated Q&A

| Question | Owner | Core answer |
|---|---|---|
| "How is precision 1.0 possible?" | B | §8 — graceful degradation, 1.00→0.80, lift not precision |
| "What legal authority do you have?" | C | §11 — analyst triage layer, not a lawyer, don't cite statutes you haven't verified |
| "What if someone uses an unrelated handle?" | A | Known limitation (§18.3) — name-based linking only |
| "Is this real-time?" | A | No — batch/file-based, roadmap item |
| "What's your recall?" | B | Own it: 0.429@10 on known storylines, but 0.071@10 on the surprise/generalization set — lead with the gap, not just the better number; reframe around precision + lift on real data |
| "How do you know a report wasn't altered?" | C | §12 — live-demo Verify integrity |
| "Why trust an unsupervised system?" | A/C | Sufficiency engine's per-model honesty + the sealed, provenance-carrying report |
| "What's next?" | D | Real-data partnership, API layer, calibrated probabilities |
