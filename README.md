# TraceWeave — Multi-Source Anomaly Detection Engine

> A source-agnostic, modular machine-learning engine that detects suspicious financial activity across **bank transactions, Call Detail Records (CDR), and social-media activity** by fusing several specialised models over a single unified dataframe — and tells the user exactly which models the data can support.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Why Multiple Models Were Used](#why-multiple-models-were-used)
- [Model Training Process](#model-training-process)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Model Comparison](#model-comparison)
- [Real-World Validation](#real-world-validation)
- [Visualization and Graphs](#visualization-and-graphs)
- [Prediction System](#prediction-system)
- [User Interface](#user-interface)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How to Use](#how-to-use)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Technical Highlights](#technical-highlights)
- [How the Entire Project Works](#how-the-entire-project-works)

---

## Project Overview

### Simple Explanation

**TraceWeave** is a tool that reads several messy data files — a bank's transaction export, a telecom's call records, and a social-media activity dump — and automatically figures out what is in them, links the same person across all three, and then scores every person to rank **who most looks like they are doing something suspicious**.

It is built around the idea that **no single file tells the whole story**. A person might make a phone call and then transfer a large amount of money *minutes* later. A bank-only monitor would never see that link; an effective detector has to combine bank, call, and social signals.

The output is **not** a single "is suspicious / is not" answer. Instead each person gets a **risk score (0–1)** and a ranked list, plus a **plain-English explanation** of *why* they were flagged (for example: "transferred money to the same person they had just called within 25 minutes").

### Technical Explanation

TraceWeave is a **multi-model anomaly-detection pipeline** over a **canonical long-format dataframe** (`SCHEMA.md`). It performs:

1. **Dynamic schema detection** — column headers are fuzzy-matched (RapidFuzz, threshold ~80) against alias dictionaries; unlabeled/renamed/foreign-language columns are inferred from their *content* (timestamp parsing, phone regex, currency detection).
2. **Entity resolution** — blocking plus Jaro-Winkler fuzzy name matching ties the same person across bank, CDR, and social sources where no shared identifier exists (account number vs. phone number vs. handle).
3. **Parallel analysis stages** that each capture a different signal dimension:
   - **Time correlation** (`pandas.merge_asof`) — cross-source co-location.
   - **Network correlation** — graph/structural signals (burst concentration, OddBall degree-mass deviation, PageRank deviation, community motifs, scatter–gather, reciprocity, layering chains).
   - **Statistical & ML outliers** — Isolation Forest (plus candidates like EIF, LOF, OCSVM, HBOS, GMM, PCA residual, Mahalanobis, autoencoder, KDE, z-score) over a shared per-entity feature vector.
   - **Benford's-law** first-digit deviation, **structuring** threshold-proximity rule, and **behavioral regime-flip** detection (dormancy, silence-after-large-transfer).
4. **Score fusion** (Borda / top-2 weighted / rank-average / score-mean / weighted-sum, plus an optional learned **meta-learner** RandomForest/XGBoost) that produces a single ranked-risk list.
5. **Explanation & reporting** in Markdown, styled HTML, and PDF.

A **data-sufficiency engine** is the key adaptive feature: for every model it determines whether the supplied data truly supports that model, returning `SUPPORTED`, `DEGRADED`, or `BLOCKED` with a human-readable reason. This means the pipeline **never runs a model on data it cannot honestly support** and **never fails silently**.

---

## Problem Statement

Anti-money-laundering (AML) monitoring is a massive problem: the UN estimates that 2–5% of global GDP — roughly **$0.8–2.0 trillion** — is laundered every year. Existing rule-based systems (fixed thresholds) generate huge false-positive queues, so analysts drown in alerts while real risk slips through.

Criminals deliberately structure their behaviour to avoid detection:

- **Structuring / smurfing** — splitting large sums to stay just below reporting thresholds.
- **Dormant-to-active flips** — accounts that have been quiet suddenly lighting up.
- **Cross-channel co-location** — a call or social post minutes before a transfer (e.g., 19 minutes), which is invisible to any single-source monitor.

The data itself makes this hard: each financial institution, telecom, and social platform exports **different column names** (e.g. `txn_date` vs `posted_at` vs `call_time`), so no fixed parser can work. Tools that assume a fixed schema break on the first new export.

TraceWeave is designed to address these three failure modes:

| Failure mode | Concrete example |
|---|---|
| **Cross-source blindness** | A call 19 minutes before a transfer is invisible to a bank-only monitor — the link *between* sources is the signal. |
| **Schema fragility** | `txn_date` vs `posted_at` vs `call_time` — a hardcoded parser dies on a new export. |
| **Coverage dishonesty** | Tools silently run with missing attributes instead of telling the user what is absent. |

---

## Key Features

| # | Feature | What it does |
|---|---|---|
| 1 | **Dynamic schema detection** | Fuzzy header matching + content-probe inference; handles renamed, unlabeled, and non-English columns; classifies the source type (bank/CDR/social) as a byproduct. |
| 2 | **Entity resolution** | Blocking + Jaro-Winkler fuzzy matching resolves one person across sources without a shared ID. |
| 3 | **Multi-model ensemble** | Time, network, statistically/ML, Benford, structuring, behavioral, and chain (layering) models fused into a single ranking. |
| 4 | **Data-sufficiency engine** | Per-stage `SUPPORTED` / `DEGRADED` / `BLOCKED` verdicts with clear, actionable reasons — never silent failure. |
| 5 | **Explainable insights** | Template-based plain-language narration per flagged person: which models fired, the concrete evidence, the combined score. |
| 6 | **One-command evaluation** | Answer-key harness: recall@k, precision@k, per-anomaly-type recall, ablation (leave-one-model-out), surprise hold-out, and the H1 hypothesis check. |
| 7 | **Multi-format loading** | CSV, TSV, JSON, JSONL/NDJSON, TXT, LOG with delimiter sniffing, latin-1 fallback, and free-form prose token extraction. |
| 8 | **Report generation** | Self-contained Markdown, styled HTML, printable PDF, and machine-readable JSON with embedded matplotlib charts. |
| 9 | **Desktop GUI** | A PySide6 drag-and-drop application with a webview report, run history, and report export. |
| 10 | **Persistent run storage** | Every analysis is archived (reports + input copies) to the user's app-data directory and survives restarts/uninstalls. |
| 11 | **Dirty-data robustness suite** | 14 fixture files that stress unknown/ambiguous inputs, verified 14/14 pass. |
| 12 | **Dataset suite + meta-learner training** | Scripts to generate labelled synthetic datasets (5 correlation modes × 3 seeds) and train/persist a fusion meta-model. |
| 13 | **Real-world validation** | Blind pipeline runs against SAML-D, an external, independently-labeled AML dataset (Oztas et al., IEEE ICEBE 2023) never used in development — see [Real-World Validation](#real-world-validation). |
| 14 | **Legal-basis provenance** | Every analysis run records who authorized it and under what legal basis, printed directly into the exported report rather than kept as hidden metadata. |
| 15 | **Chain-of-custody sealing** | Every archived run is SHA-256 sealed file-by-file and chained to the previous run's seal; a live "Verify integrity" check detects tampering. |

---

## Technology Stack

### Programming Language

- **Python** (3.11+) — the entire pipeline, models, GUI, and tooling.

### Machine Learning & Data Science

| Library | Purpose |
|---|---|
| **pandas** | Data loading, the canonical long-format dataframe, `merge_asof` time correlation, group-by feature engineering. |
| **NumPy** | Array math for scoring, Benford histograms, burst concentration, feature matrices. |
| **SciPy** | Statistics (`chi2`, `gaussian_kde`, `linregress`), Mahalanobis distance. |
| **scikit-learn** | `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `GaussianMixture`, `PCA`, `MinCovDet`, `MLPRegressor`, `KMeans`, and the meta-learner `RandomForestClassifier`. |
| **RapidFuzz** | Fuzzy header/name matching for schema detection and entity resolution. |
| **networkx** *(imported, not in requirements.txt)* | Ego-graph rendering and graph models (PageRank, Louvain communities) — imported with `try/except`. |
| **XGBoost** *(optional)* | Meta-learner candidate, compared against RandomForest. |
| **PyTorch / torch** *(optional)* | Deep models — an LSTM autoencoder and a temporal message-passing GNN (TGN-lite). Skipped with a warning when unavailable. |
| **joblib** | Persisting and loading the trained fusion meta-model. |

### Data Visualization

- **matplotlib** — all report charts (timeline, people who stand out, sources, ego-graphs, entity drill-down, activity map, model scores).

### Desktop / GUI & Reporting

- **PySide6** — the drag-and-drop desktop application (Qt widgets + `QWebEngineView`).
- **reportlab** — PDF report generation.
- **platformdirs** — user data directory for persistent run storage.
- **pypdf** *(listed in requirements)*.

> **Why these were used:** pandas provides the dataframe-centric architecture that the whole project is designed around (every stage reads/writes the same table). scikit-learn gives a broad, battle-tested set of unsupervised outlier detectors that require no labels. RapidFuzz powers the fuzzy matching that makes schema detection and entity resolution robust to real-world naming chaos. matplotlib produces the charts that make findings interpretable, and PySide6 provides a rich desktop UI without a web backend.

---

## System Architecture

```text
sources/*.csv · *.json · *.jsonl · *.txt · *.log
        │
        ▼
[Stage A] Dynamic Schema Detection     (RapidFuzz header match + content probe)
        │
        ▼
[Stage B] Entity Resolution            (blocking + Jaro-Winkler fuzzy)
        │
        ▼
        ┌──────────────────────────────┐
        │  Unified long-format dataframe │  entity_id | event_type | timestamp |
        │                              │  source | amount | counterparty_id | …
        └──────────────────────────────┘
        │
   ┌────┴──────────┬───────────────┬───────────────────┐
   ▼               ▼               ▼                   ▼
[Stage C]       [Stage D]     [Stage E]          [Stage H]
Time           Network        Stat/ML            Benford · structuring ·
Correlation     (graph)       outliers           behavioral · chain
(merge_asof)    models        (IF + candidates)   (rules/statistical)
   │               │               │                   │
   └───────────────┴───────────────┴───────────────────┘
                        │
                 [Stage F] Score Fusion & Ranking
                 (Borda / top-2 / meta-learner)
                        │
                 [Stage G] Explanation Generator
                        │
                 Report: Markdown · HTML · PDF · JSON
                 + Desktop GUI (PySide6)
```

### Stage-by-stage workflow

1. **Stage A — Schema detection.** Loads any supported format, fuzzy-matches each column header to a canonical slot (`timestamp`, `amount`, `counterparty`, `event_type`, `location`, `direction`, `duration`), and falls back to content probing when the header is unknown. Classifies the source type.
2. **Stage B — Entity resolution.** Groups records into person entities using identifiers + fuzzy name matching, so the same person is one `entity_id` across all sources.
3. **Unified dataframe.** All sources are concatenated and deduplicated into one long-format table.
4. **Sufficiency check.** The engine decides which models the data truly supports.
5. **Models run in parallel.** Only non-`BLOCKED` models are executed.
6. **Fusion.** Scores are normalised and combined into a final ranked risk list with explanations.
7. **Report.** Markdown/HTML/PDF/JSON are written, showing top findings as plain-English insights backed by charts.

---

## Dataset

The project ships **synthetic data** generated by its own data generator (`src/aml/data_generator/`). This follows published precedent (IBM NeurIPS 2023, SAML-D) — using synthetic data with **complete ground-truth labels** so models can be evaluated honestly.

### Design: two-tier data

- **Predefined storylines** — scripted anomaly types with documented ground truth (used to verify each model).
- **Randomised surprise injector** — anomalies written only to a **write-only answer key**, so any entity the system flags from that set is genuine generalisation, not a lookup table.

### Files in `data/sources/`

| File | Format | Approx. rows | Source type |
|---|---|---|---|
| `bank_export.csv` | CSV | ~4,035 | bank |
| `bank_export.json` | JSON | same records | bank |
| `cdr_export.csv` | CSV | ~12,726 | CDR |
| `cdr_export.jsonl` | NDJSON | same records | CDR |
| `social_export.csv` | CSV | ~1,816 | social |
| `social_export.txt` | TSV | ~320 | social |

Columns are deliberately **non-uniform** across sources (matching real exports): bank rows carry `txn_type, amount, recipient_account, branch_code, account_name, account_no, timestamp, device`; CDR rows carry `caller_msisdn, callee, cell_id, length, subscriber, timestamp, notes`; social rows carry timestamps, handles, and mentions.

### Ground-truth answer keys

- `data/answer_key.json` — predefined storyline entities (per-entity type + storyline).
- `data/answer_key_surprise.json` — the surprise/anonymised set.

### Anomaly types (ground-truth labels)

| Type | Meaning |
|---|---|
| `colocation` | Cross-source temporal link (call/post shortly before transfer). |
| `dormant_flip` | Dormant entity abruptly becomes active. |
| `structuring` | Amount splitting just below the reporting threshold. |
| `fan_io` | Fan-in / fan-out structural pattern. |
| `silence` | Silence after a large transaction. |
| `post_burst` | Sudden burst of social activity. |
| `chain` | Rapid multi-hop layering transfer path. |
| `background` | Normal (the write-only answer-key class). |

### Dataset suite (`data/datasets/`)

A **15-dataset suite** generated across **5 correlation modes** (`full`, `none`, `pair_bank_cdr`, `pair_bank_social`, `pair_cdr_social`) × **3 seeds**, each with measured cross-source overlap, allowing head-to-head evaluation of how well each mode supports the cross-source signal. See `data/datasets/suite.md`.

### Dirty-data fixtures (`data/dirty_samples/`)

14 files that specifically stress the robustness of schema detection: `renamed_mixed_case.csv`, `headers_es.csv`, `unix_epoch.csv`, `mixed_dates.csv`, `currency_mess.csv`, `missing_values.csv`, `dup_columns.csv`, `no_header.csv`, `latin1.csv`, `semicolon.csv`, `sparse.csv`, `bank_only.csv`, `cdr_only.csv`, `social_only.csv`. All 14 verified to pass (`GAP_AUDIT.md`).

> Exact population statistics (total unique people, event counts per storyline) are generated at runtime and vary by seed; the generator and answer keys are the authoritative source of ground truth.

---

## Data Preprocessing

|<div style="width:170px">Step</div>|<div style="width:210px">What it does</div>|Why it matters|
|---|---|---|
|**Encoding fallback**|CSV failures with `UnicodeDecodeError` retry with `latin-1`; warnings emitted.|Real exports aren't always UTF-8; a crash here would kill the whole pipeline.|
|**Delimiter sniffing**|`,` `\t` `;` `|` detected from the first line.|Semicolon-delimited and tab-delimited files load correctly.|
|**Schema detection & normalisation**|Columns mapped to canonical slots by header + content; `amount` string→float parsing (`$`, commas, `(parens)`, comma-decimal, euro style); `direction`/`event_type` derivation.|Produces the consistent long-format dataframe every model reads.|
|**Timestamp parsing**|Tolerant datetime parsing: unix-epoch (s/ms/µs) auto-detection, mixed-format element-wise fallback, day-first vs month-first heuristic, tz-naive normalisation.|Real timestamp columns come in many formats; naive/hardcoded parsing was a source of bugs.|
|**Missing-value handling**|Many spellings of "missing" (`nan`, `None`, `N/A`, `-`, `nil`, empty) collapsed to NaN; dtype kept `float64`/`NA`.|Keeps the dataframe at the correct dtype so numeric models don't crash.|
|**Deduplication**|Exact duplicate rows dropped after ID cleaning (same event arriving as CSV + JSON).|One logical event is never counted twice by `merge_asof`, counts, or feature vectors.|
|**Entity resolution**|Identifiers + names mapped to a single `entity_id` (fuzzy name matching across sources only).|Links the same person across bank/CDR/social — the flagship cross-source capability.|
|**Feature engineering**|Per-entity feature vector: counts, amount stats, fan-in/out, temporal gaps, hour entropy, night/weekend ratios, structuring proximity, velocity.|The shared Stage-E input; engineered once and consumed by all candidate outlier models.|
|**Min-max normalisation / ranking**|Per-model raw scores min–max scaled to 0–1 (`_minmax`), sometimes power-scaled. Feature matrices standardised (`z-score`) and log1p-transformed.|Makes scores comparable and models robust to scale differences before fusion.|
|**Sufficiency gating**|Before each model runs, required vs. populated fields and event volume are checked; <30 events/entity blocks entity-scored models.|Prevents models from producing spurious scores on insufficient data.|

---

## Exploratory Data Analysis

EDA in this project is **visualisation-driven reporting** built with matplotlib and embedded in the generated report. All charts are generated in `src/aml/visuals.py`. Below are the charts and what each represents.

### Timeline of activity

**Purpose:** Answer "when did the events happen, and from which source?"
**Axes:** X-axis = date; Y-axis = number of records, stacked by source (bank/CDR/social).
**Interpretation:** Shows the volume of activity over the observation window, broken down by source, revealing periods of heavy vs. sparse activity.
**Why it matters:** Context for the whole analysis — dormant/silent periods and bursts are only meaningful relative to the overall timeline.

### People who stand out

**Purpose:** Answer "who should an analyst look at first?"
**Axes:** Y-axis = top person names; X-axis = number of distinct signals (reasons) found for each.
**Interpretation:** Bars are coloured by risk (HIGH = red, MEDIUM = amber, LOW = teal). Longer bars mean more independent pieces of evidence.
**Why it matters:** Turns raw model scores into a prioritised, human-first watchlist.

### Where the records came from

**Purpose:** Answer "what data types did we process and at what volume?"
**Axes:** X-axis = source type; Y-axis = counts; grouped bars for records vs. people.
**Interpretation:** Shows per-source row counts and how many entities each contributed.
**Why it matters:** Makes the source mix and coverage explicit before conclusions are drawn.

### Ego-graphs of flagged people

**Purpose:** Answer "who is each flagged person connected to?"
**Axes:** None (network layout). The flagged person is the red labelled hub; each neighbour is a dot coloured by source, with dot size scaling with contact volume and line thickness with flow.
**Interpretation:** Reveals fan-in/fan-out structure, dense clusters, and which sources the flagged person interacts with.
**Why it matters:** Visualises the network/structural signals that graph models detect.

### Per-person drill-down timeline

**Purpose:** Answer "what is one flagged person's complete activity pattern?"
**Axes:** X-axis = time; a scatter of event dots (colour = source, size = amount) plus a daily-activity bar.
**Interpretation:** Shows colocation, bursts, and dormancy/silence at the individual level.
**Why it matters:** Provides the evidence trail behind a single flagged entity.

### Activity map

**Purpose:** Answer "where did flagged activity happen?"
**Axes:** Pseudo-geographic layout of locations (branches/towers/geo-tags); dot size = activity, red = used by a flagged person.
**Interpretation:** Highlights the physical locations that correlate with suspicious behaviour.
**Why it matters:** Adds a location dimension to the insight story.

### Model results (peak score & flagged count per model)

**Purpose:** Answer "how much signal did each model family produce?"
**Axes:** X-axis = model; Y-axis = score/count; grouped bars = peak score vs. number of entities flagged (>0.6).
**Interpretation:** Shows which models fire and how strongly, and which are inactive for a given dataset.
**Why it matters:** Makes the model mix transparent and shows the adaptive coverage of the pipeline.

---

## Machine Learning Models

TraceWeave uses a **mixture of models**, each capturing a different signal dimension. The main pipeline runs a fixed set of model families (`time_correlation`, `network`, `statml`, `benford`, `structuring`, `behavioral`, `chain`). A large **model zoo** of additional candidates is implemented and driven by the research/evaluation scripts.

### Beginner vs. project-specific explanation format

Below, each model is explained twice: a **Beginner Explanation** (plain language) and a **Project-Specific Explanation** (how it was used here).

### 1. Time-Correlation (`merge_asof`)

- **Beginner:** If someone calls/posted a person and then transfers them money within minutes, the two events are suspiciously connected in time.
- **Project-specific:** `fit_time_correlation` uses `pandas.merge_asof` to join CDR/social events with bank transfers within a tolerance (default ~25–30 minutes), counting only **same-party** episodes (call to X then transfer to X). Each distinct matched transfer counts as one episode. The score is `min(episodes / min_episodes, 1.0)`. The `time_correlation_backward` variant catches call-after-transfer choreography, and `time_correlation_anypair` counts same-party overlaps without asof pairing. This is the signature **cross-source co-location** signal.

### 2. Network (graph) models

- **Beginner:** If a person suddenly deals with many different people in a short time, or forms an unusual hub/cluster, that is structurally suspicious.
- **Project-specific:** The primary `network` model (`fit_oddball`) is an **OddBall-style** degree-vs-weight deviation over egonets — it fits `w ~ d^k` (log-log regression of edge count on degree) and flags entities whose mass deviates from the expected density line (a classic money-mule / fan-out signature). Additional graph candidates: `fit_network` (burst concentration — max distinct counterparties in a window), `oddball_signed`, `degree_deviation`, `reciprocity`, `pagerank_deviation`, `community_motif`, `scatter_gather`, and the deep `fit_network_tgn` (temporal GNN autoencoder).

### 3. Statistical / ML Outliers (Isolation Forest + candidates)

- **Beginner:** If a person's behaviour profile (how often they move money, how many people they deal with, their amount patterns, their activity rhythm) is very different from everyone else's, they stand out statistically.
- **Project-specific:** Builds a shared per-entity feature vector (`n_events`, amount stats, fan-in/out, temporal gaps, hour entropy, night/weekend ratios) via `build_features`, then `fit_statml` runs scikit-learn's **Isolation Forest** (contamination 0.06, 400 estimators) on the standardised matrix and ranks by `-score_samples`. Candidates in the zoo: Extended Isolation Forest (`statml_eif`), LOF (`statml_lof`), robust Mahalanobis via MinCovDet (`statml_mahalanobis`), PCA reconstruction error (`statml_pca`), One-Class SVM (`statml_ocsvm`), an MLP autoencoder (`statml_autoencoder`), HBOS (`statml_hbos`), GMM (`statml_gmm`), KDE (`statml_kde`), polarising z-score (`statml_zscore`), and an LSTM autoencoder (`statml_lstm`, needs torch).

### 4. Benford's-law deviation

- **Beginner:** Real-world financial amounts have a natural leading-digit distribution (Benford's law). If a person's amounts deviate from it, the numbers may be manufactured.
- **Project-specific:** `fit_benford` builds the **population** first-digit distribution, counts each entity's observed first-digit histogram, and scores by a chi-square **p-value** (`1 − p`). Variants: `benford_ks` (Kolmogorov–Smirnov-style max deviation) and `benford_second` (second-digit test, which catches "round" cleaned amounts).

### 5. Structuring rule

- **Beginner:** Repeatedly sending amounts just below the reporting threshold (e.g., just under £10,000) is a classic money-laundering trick (smurfing).
- **Project-specific:** `fit_structuring` counts an entity's bank rows in the band `[0.88 × threshold, threshold)` and scores as `counts^1.5 / sqrt(total events)` — rewarding *persistent* small-cutters relative to overall activity. `fit_structuring_banded` is a variant that weights band-event quantity directly.

### 6. Behavioral regime-flip

- **Beginner:** A person going very quiet and then suddenly becoming active, or moving a huge amount and then going silent, is a behaviour change worth flagging.
- **Project-specific:** `fit_behavioral` implements three **person- AND population-relative** signals: (1) cold-start flip (bank activity begins very late then ≥3 credits land in the final window), (2) mid-life dormancy (a quiet gap ≥45 days that is ≥5× the person's own median gap), (3) silence tail (an extreme debit is the very last event, followed by ≥15 days of quiet).

### 7. Chain / layering detector

- **Beginner:** Money "passing through" several accounts in rapid succession (A → B → C → D) is a layering pattern.
- **Project-specific:** `fit_chain` finds directed paths of ≥3 consecutive "big" debits (person-relative extreme amount), where each hop occurs within `hop_days` of the previous, and scores every entity on a chain by depth and amount extremity. This is the model behind the `chain` storyline label.

### Deep models (optional, torch-gated)

- `fit_statml_lstm` — an LSTM autoencoder that reconstructs each entity's ordered event-feature sequence; entities it cannot reconstruct from a compressed latent are sequential outliers.
- `fit_network_tgn` — a temporal message-passing GNN (TGN-lite) graph autoencoder across time buckets; nodes whose neighbourhood cannot be rebuilt from the compressed representation deviate from the graph structure.

> Both deep models are listed in the zoo and implemented, but they **skip with a warning when `torch` is not installed** — they are not part of the default number reported in the evaluation baseline.

---

## Why Multiple Models Were Used

Different algorithms capture **different, complementary signal dimensions** of the same dataframe:

- Time models see *when* events happen relative to each other.
- Network models see *who* is connected to whom and the *shape* of those connections.
- Stat/ML models see *feature-space* outliers compared with the rest of the population.
- Benford/structuring/behavioral/chain models catch *specific laundering typologies*.

Because real anomalous behaviour rarely shows up in only one dimension — and because supervised labels are scarce in this domain — the project trains several **unsupervised** detectors and **fuses** their normalised scores. This is motivated by the research hypothesis **H1**: *no single model exceeds ~0.7 recall@10 alone, while the fused ensemble exceeds 0.9*. The evaluation (`eval.py`) includes an H1 check and a leave-one-model-out ablation to verify that each model adds value.

The fusion is chosen deliberately: the default is **Borda-count rank aggregation** (with a max-bonus so a perfect single-model alarm isn't buried), with alternatives (top-2 weighted, rank-average, score-mean, weighted-sum, and a learned RandomForest/XGBoost **meta-learner**) available.

---

## Model Training Process

```text
Raw Dataset
   ↓
Cleaning (encoding, dedup, missing-value)
   ↓
Preprocessing (schema detection, normalisation, entity resolution)
   ↓
Feature Engineering (shared per-entity feature vector)
   ↓
Sufficiency check (which models the data supports)
   ↓
Train / score (unsupervised models fit on the population)
   ↓
Predictions (per-model 0–1 scores)
   ↓
Evaluation (recall@k, precision@k, per-type, ablation, surprise)
```

Key training facts (from the code):

- The default routing **trains/scales on the full ingested population** (unsupervised, no labels needed at prediction time).
- `min_events_per_entity = 30` — the volume gate below which entity-scored models are `BLOCKED`.
- Toleration window `tol_minutes` is configurable (default 30 in config, 25 in some model defaults); the pipeline config sets `tol_minutes: 25`.
- `structuring_threshold = 10000.0`.
- Fixed seeds are used for reproducibility: e.g., `IsolationForest(random_state=0)`, `EIF(random_state=0)`.
- The **meta-learner** is trained offline by `scripts/train_fusion_meta.py` over several generated labelled datasets, comparing **RandomForest** vs **XGBoost** by cross-validated average precision; the better model is persisted to `src/aml/assets/fusion_meta.joblib` and loaded at analysis time — with automatic fallback to Borda if missing.

---

## Model Evaluation Metrics

The primary evaluation is the **ranked-top-k** framing, appropriate for a ranked watchlist where an analyst reviews the top alerts:

| Metric | Meaning (simple) | Why relevant here |
|---|---|---|
| **recall@k** | Of all true anomalies, what fraction appear in the top-k ranked list. | Measures whether the flagship anomalies surface high enough to be seen. |
| **precision@k** | Of the top-k results, what fraction are actually (known) anomalies. | Measures false-alert control — how much noise the analyst must wade through. |
| **Per-type recall@10** | For each anomaly type (colocation, dormant_flip, ...), what fraction makes the top-10. | Diagnoses which models/typologies are well or poorly covered. |
| **Ablation (leave-one-model-out)** | recall@10 when each model is removed from the fusion. | Verifies that each model contributes value (H1). |

Metrics are computed by `eval.py` by mapping answer-key identities onto pipeline entity IDs via the resolution table, evaluating the predefined-storyline set on the main key and the generalisation set on the **surprise key** separately (to avoid leakage).

> Specific numeric results are reported in the [Results](#results) section from the persisted evaluation snapshots in `results/`.

---

## Model Comparison

The persisted evaluation snapshots compare fusion behaviour and per-model contributions. Because exact per-model accuracy numbers vary with the dataset seed and the audit shows the reported numbers changed as bugs were fixed (see [Results](#results) and `GAP_AUDIT.md`), the most reliable *comparative* evidence is the per-model score coverage visible in `results/current_baseline/eval.json`:

| Model family | Entities scored (current baseline) |
|---|---|
| `time_correlation` | 86 |
| `network` | 200 |
| `statml` | 508 |
| `benford` | 100 |
| `structuring` | 7 |
| `behavioral` | 42 |

This shows how much of the population each model's signal covers — the stat/ML feature-scorer scores everyone, while the highly-specific `structuring` model fires on very few entities.

> A full head-to-head of every candidate model by accuracy is produced by the research scripts (`scripts/model_matrix_eval.py`, `scripts/model_zoo_eval.py`, `scripts/dataset_suite_eval.py`); the summary of model-vs-model conclusions is documented in the plan and the persisted results, but **exact candidate-model accuracy numbers are not statically stored in the repo's README-visible files** and so are not claimed here.

---

## Real-World Validation

Every number above is graded on the project's own synthetic generator. To get a number that isn't self-graded, the pipeline was also run **blind** against real, external, independently-labeled data it never saw during development.

### SAML-D

[SAML-D](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml) (Oztas et al., *"Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset,"* IEEE ICEBE 2023 — the same paper `PLAN.md` already cited as design precedent, before this validation ever ran) is a real, peer-reviewed AML transaction dataset: 9,504,852 transactions, 9,873 labeled suspicious (**0.104%** true fraud rate), spanning 28 typologies (Structuring, Smurfing, Layered_Fan_In/Out, Behavioural_Change, Bipartite, Cash_Withdrawal, Deposit-Send, and more). License: CC BY-NC-SA 4.0 (non-commercial).

Reproduced via `scripts/real_dataset_samld.py` (download the CSV yourself — Kaggle gates it behind login — see the script's docstring). Because the sufficiency engine's volume gate is a **population average**, not per-entity (see `sufficiency.py::_overall_volume_note`), any sample that scores at all must concentrate known-fraud accounts well above SAML-D's true base rate. Rather than report one such number, the script runs **two** validation passes on purpose:

| | Curated (`--rebuild`) | Harder (`--harder --rebuild`) |
|---|---|---|
| Scored entities | 4,625 | 7,929 |
| Ground-truth positives resolved | 1,005 | 698 |
| Fraud prevalence in sample | 21.7% | 8.8% |
| precision@10 | 1.000 | 0.800 |
| precision@100 | 0.590 | 0.270 |
| recall@10 | 0.010 | 0.011 |
| recall@1000 | 0.263 | 0.221 |

**Read this honestly**: precision is extremely sensitive to base rate, and both samples are still far more fraud-concentrated than SAML-D's real 0.104% rate (getting all the way to that rate isn't reachable without a fundamentally larger sample than this pipeline's entity-resolution step can process quickly). The finding that matters is that precision@10 **degrades gracefully** (1.00 → 0.80) as the test gets harder, rather than collapsing — evidence of real ranking signal, not a lucky number at one enriched setting. The more portable metric is **lift over random ranking**: 4.6x at k=10 in the curated pass, tapering to 1.2x by k=1000.

Validating against real data also surfaced and fixed two real bugs invisible to the synthetic test suite: a schema-detection gap where sender/originator-style headers (e.g. `nameOrig`) never resolved to `actor_id`, silently losing the entity dimension; and a `RecursionError` crash in the chain/layering detector caused by a transaction cycle (two accounts paying each other back within the hop window) that had no cycle guard.

### PaySim (structural mismatch, not a pipeline bug)

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (6,362,620 transactions, 8,213 fraud rows, CC BY-SA 4.0) was also tried, via `scripts/real_dataset_paysim.py`. It turned out to be structurally incompatible with the pipeline's entity-centric design: nearly every sender account appears **exactly once** (mean ~1.0–1.5 transactions/account) — a one-shot-per-transaction fraud-classification dataset, not an accumulating-history-per-account one the way SAML-D and this project's own synthetic generator are. No amount of resampling fixes this; `min_events_per_entity` can never be cleared. The attempt was still valuable — it is what surfaced the `nameOrig`/`actor_id` schema-detection bug fixed above.

---

## Visualization and Graphs

All report charts are generated by `src/aml/visuals.py` and embedded into Markdown/HTML/PDF. They are described in detail in [Exploratory Data Analysis](#exploratory-data-analysis). Additional model/experiment charts are produced by the research scripts under `scripts/`.

Chart list (report):
1. **Timeline of activity** — event volume over time, stacked by source.
2. **People who stand out** — horizontal bars of evidence count per top person, coloured by risk.
3. **Where the records came from** — per-source records vs. people grouped bars.
4. **Ego-graphs of flagged people** — network panel per top person (hub + neighbours by source).
5. **Per-person drill-down timeline** — event dots (colour = source, size = amount) + daily activity bars.
6. **Activity map** — pseudo-geographic map of locations, red = flagged.
7. **Model results** — peak score + flagged-entity count per model family.

---

## Prediction System

1. **What the user provides:** one or more source files (bank/CDR/social) in CSV/TSV/JSON/JSONL/TXT/LOG.
2. **How input is processed:** each file is loaded, schema-detected, and normalised into canoncial columns; the same preprocessing used during training (encoding, parsing, dedup, entity resolution, feature engineering) is applied automatically.
3. **Model selection:** the sufficiency engine decides which models run based on the resolved fields and event volume.
4. **Predictions:** each active model produces a normalised 0–1 score per entity.
5. **Fusion & ranking:** scores are fused (Borda by default) into one ranked risk list with `entity_id`, `score`, `n_events`, and the list of models that fired (>0.6).
6. **Result displayed:** the report shows the top findings as plain-English insights with charts; the full machine-readable scores go to `report.json`.

> Because the models are **unsupervised**, there is no separate "training vs. user-input" mismatch: the same ingestion + feature pipeline scores any new file set exactly as it scored the development data.

---

## User Interface

The project ships a **PySide6 desktop application** (`main_app.py`) named **TraceWeave**.

### Main window screens

| Screen | Contents |
|---|---|
| **Home** | A drag-and-drop zone, a browse button, a list of loaded files (each removable, with a "Clear all" button), **Legal basis / authorization** and **Analyst** text fields, an **Analyze** button, and a **History** button. Status text guides the user. |
| **Report** | A `QWebEngineView` preview of the colourful HTML report filling the window, plus buttons: **Download MD**, **Download PDF**, **History**, and **← Home**. |
| **History** | A dialog listing archived runs (records · people · sources · top insight) with **Open selected**, **Verify integrity**, and **Delete**. |

### Interaction flow

1. User drags-and-drops supported files into the drop zone (unsupported extensions are skipped with a toast notification).
2. Files appear as removable rows; the Analyze button enables once a file is present.
3. User optionally fills in **legal basis / authorization** (warrant #, case ref, regulatory request) and **analyst** name — left blank, the report honestly shows "not recorded" rather than a fake default.
4. Clicking **Analyze** runs the pipeline on a background thread (`QThread`) so the UI stays responsive; the button shows "Analyzing…".
5. On completion, the report screen opens in the webview, the run is **automatically archived and SHA-256 sealed** (see [Chain-of-Custody Sealing](#chain-of-custody-sealing) below), and a non-blocking toast confirms completion.
6. The user can download the report as Markdown or PDF, and revisit past analyses via **History**, where **Verify integrity** recomputes hashes and reports `OK` or exactly which file was `TAMPERED`.

The UI uses a "frosted-glass" green-and-white theme with explicit hover/pressed states on every button.

### Legal-basis provenance

Every analysis run records who authorized it and under what legal basis. This is not just metadata — it is printed directly into the exported report itself (Markdown/HTML/PDF/JSON), so the artifact carries its own record of authorization. If left blank, the report prints **"Legal basis / authorization: not recorded"** honestly, rather than implying consent that was never captured — the same never-hide-an-absence discipline as the sufficiency engine.

This is designed to answer, directly, the question a law-enforcement partner will ask first: *"under what authority was this data analyzed?"* TraceWeave does not collect data — it is an analyst-side triage layer over data already lawfully obtained (a production order, a court order, a compliance channel), and every report states that basis explicitly. (This framing has not been verified against specific statutory citations — treat it as a design principle, not legal advice.)

### Chain-of-custody sealing

Every archived run is hashed file-by-file with SHA-256 (every report file, chart, and copied source file) and the resulting hash set is **chained to the previous run's seal** — an append-only ledger, the same tamper-evidence idea a blockchain uses, without needing one.

- `storage.verify_run(run_id)` recomputes hashes and reports `OK`, or exactly which file(s) are `TAMPERED`.
- `storage.verify_chain()` walks every archived run in order and confirms each one's seal correctly chains to the previous one's — catching tampering with a past run's *recorded* hashes, not just its files.
- Live in the GUI: History → select a run → **Verify integrity**.

This is a **local, tamper-evident** check, not a cryptographically signed, tamper-proof one — it proves a report hasn't been altered since sealing and that history hasn't been quietly rewritten, but nothing here is signed with a key the analyst doesn't also control, so it does not prove authorship to a third party.

---

## Project Structure

```text
Hackathon/
│
├── main_app.py               # PySide6 desktop GUI (drag-drop analyst)
├── run_pipeline.py           # One-command CLI analysis -> report
├── run_demo.py               # One-command demo: generate data + run + report
├── eval.py                   # Evaluation harness (recall@k, ablation, surprise)
│
├── config/
│   └── config.yaml           # All pipeline numeric/dir settings + probe thresholds
│
├── requirements.txt          # Python dependencies
├── SCHEMA.md                 # Unified dataframe frozen spec
├── PLAN.md                   # Research plan, model matrix, execution timeline
├── GAP_AUDIT.md              # Audit: PLAN.md spec vs implementation + results
│
├── src/aml/                  # Core package
│   ├── __init__.py           # Windows torch/threading safety, version
│   ├── config.py             # PipelineConfig loaded from YAML
│   ├── schema.py             # Canonical slots, aliases, capability registry
│   ├── schema_detect.py      # Stage A: dynamic schema detection + normalisation
│   ├── entity_resolve.py     # Stage B: entity resolution (Jaro-Winkler)
│   ├── loaders.py            # Multi-format loaders (CSV/JSON/TXT/prose)
│   ├── sufficiency.py        # SUPPORTED/DEGRADED/BLOCKED engine
│   ├── pipeline.py           # PythonPipeline orchestration -> PipelineResult
│   ├── models.py             # All model families + fusion + dispatch
│   ├── fusion_meta.py        # Learned RandomForest/XGBoost fusion meta-learner
│   ├── insights.py           # Plain-language evidence extractors
│   ├── visuals.py            # matplotlib report charts
│   ├── report.py             # Markdown/HTML/PDF/JSON report generation + legal-basis provenance
│   ├── storage.py            # Persistent run archive + SHA-256 chain-of-custody sealing
│   ├── assets/
│   │   └── fusion_meta.joblib  # Persisted trained meta-model
│   └── data_generator/
│       ├── population.py     # Persona registry + normal behaviour
│       ├── storylines.py     # Predefined anomaly typologies
│       ├── emit.py / emit_research.py / emit_multi.py  # Dataset writers
│
├── scripts/                  # Research & tooling
│   ├── make_dirty_fixtures.py    # Dirty-data fixture generation
│   ├── dirty_audit.py            # Dirty-data robustness audit
│   ├── model_matrix_eval.py      # Per-stage candidate sweep
│   ├── model_zoo_eval.py         # Model-zoo evaluation
│   ├── dataset_suite_eval.py     # Dataset-suite evaluation
│   ├── generate_dataset_suite.py # Dataset-suite generation
│   ├── real_dataset_validate.py  # Internet-sample validation
│   ├── research_formats_validate.py
│   ├── fetch_samples.py          # Fetch internet sample data
│   ├── probe_dirty.py / stage_a_bench.py / stage_a_stress.py
│   ├── train_fusion_meta.py      # Train/persist the meta-learner
│   ├── real_dataset_paysim.py    # Real-world validation: PaySim (structural mismatch)
│   ├── real_dataset_samld.py     # Real-world validation: SAML-D (--harder for lower prevalence)
│   ├── tune.py
│   └── run_all.py
│
├── tests/                   # pytest suite (12 files; 96 pass, 0 skip)
│   ├── conftest.py           # Synthetic-dataset fixture
│   └── test_*.py
│
├── data/
│   ├── sources/              # CSV/JSON/JSONL/TXT sample exports
│   ├── answer_key.json       # Predefined-storyline ground truth
│   ├── answer_key_surprise.json  # Surprise/anonymised write-only key
│   ├── dirty_samples/        # 14 dirty-data fixtures
│   ├── internet_samples/     # Fetched real-world samples (processed .csv gitignored)
│   ├── real_datasets/        # Downloaded PaySim/SAML-D CSVs (gitignored, ~1.9GB; see scripts/real_dataset_*.py)
│   └── datasets/             # 5-mode × 3-seed dataset suite + suite.md
│
├── results/                  # Persisted eval snapshots (gitignored)
│   ├── current_baseline/eval.json · eval.md
│   ├── after_fixes/eval.json · eval.md
│   └── real_datasets/        # samld_verdict.json · samld_verdict_harder.json · paysim_verdict.json
│
└── assets/                   # Pitch deck + chart-generation scripts
    ├── AML_Pitch_Deck*.pptx
    ├── build_deck.py · make_charts.py · make_deck_charts.py
    └── charts/ · slide1.html
```

### Important files explained

- **`main_app.py`** — the desktop application entry point; can also render a UI screenshot via `--shot`.
- **`run_pipeline.py`** — the primary CLI: loads configured inputs, runs the pipeline, prints source detection + sufficiency verdicts + top-10, and writes the full report.
- **`run_demo.py`** — regenerates the synthetic dataset and runs the pipeline end-to-end for a demo.
- **`eval.py`** — reproduces the evaluation metrics and ablation against the answer keys with `--out results/<name>`.
- **`scripts/real_dataset_samld.py`** — real-world validation against SAML-D; `--harder` runs the lower-prevalence, more realistic pass. See [Real-World Validation](#real-world-validation).
- **`SCHEMA.md`** — the frozen canonical dataframe spec that every stage builds against.
- **`PLAN.md`** — the research foundation and design decisions.
- **`GAP_AUDIT.md`** — the spec-vs-implementation audit and verified results.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/dhruvnailwal/Hackathon.git
cd Hackathon
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- Windows: `venv\Scripts\activate`
- macOS / Linux: `source venv/bin/activate`

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> Optional extras (not in `requirements.txt`, used only by the optional research/deep paths): `networkx` (graph models/rendering), `xgboost` + `joblib` (meta-learner training), `torch` (LSTM/TGN deep models), `pytest` (tests), `pyyaml` (config). The pipeline runs with the core requirements; missing optional libraries degrade gracefully with warnings.

### Step 4 — Run the tests (recommended)

```bash
pytest
```

(96 passing, 0 skipped as of the latest run; `GAP_AUDIT.md` reports the earlier 92/2 baseline.)

---

## How to Run

### CLI — pipeline + report

```bash
python run_pipeline.py
```

or point it at specific files/directories:

```bash
python run_pipeline.py data/sources
python run_pipeline.py path/to/file.csv
```

### CLI — one-command demo

```bash
python run_demo.py
```

Regenerates the synthetic dataset, runs the pipeline, and prints a summary.

### Desktop GUI

```bash
python main_app.py
```

### Evaluation harness

```bash
python eval.py
python eval.py --out results/experiment
```

### Dirty-data audit (14 fixtures)

```bash
python scripts/make_dirty_fixtures.py
python scripts/dirty_audit.py
```

### Train the fusion meta-learner (optional)

```bash
python scripts/train_fusion_meta.py
```

### Real-world validation (SAML-D / PaySim)

Requires downloading the dataset CSV yourself first (Kaggle gates both behind login — see each script's docstring for the exact URL and where to place the file).

```bash
python scripts/real_dataset_samld.py              # curated pass (21.7% prevalence)
python scripts/real_dataset_samld.py --harder --rebuild   # harder pass (8.8% prevalence)
python scripts/real_dataset_paysim.py             # PaySim (structural mismatch — see Real-World Validation)
```

---

## How to Use

1. **Start the application** — run `python main_app.py` (GUI) or `python run_pipeline.py` (CLI).
2. **Add your files** — drag-and-drop (or browse for) bank/CDR/social files in the supported formats.
3. **Press Analyze** — the pipeline runs automatically and adaptively.
4. **Read the report** — the top findings are plain-English insights; each flagged person has evidence bullets and a risk level (HIGH/MEDIUM/LOW).
5. **Check which models ran** — the run summary lists source types and per-model sufficiency verdicts; a `BLOCKED` model shows *why* (e.g., "add a bank file to enable it").
6. **Export / revisit** — download the report as Markdown or PDF, or open a past run from History.
7. **For CLI users** — the printed output shows detected sources, model verdicts, and the top-10 ranked insights; the full report is written to the configured report directory (default `data/report/`).

---

## Example Prediction

Because the system is file-driven, an "example prediction" is best shown as a pipeline run over the sample set:

```text
bank_export.csv  -> Schema detect -> Entity resolve -> ...
cdr_export.csv   -> Schema detect -> Entity resolve -> ...
social_export.csv-> Schema detect -> Entity resolve -> ...
        unified long-format dataframe
        ↓
  sufficiency verdicts (which models supported)
        ↓
  models run in parallel
        ↓
  Borda fusion -> ranked insights with scores and explanations
        ↓
  report (MD/HTML/PDF/JSON)
```

In `results/current_baseline/eval.md`, the top-ranked entity is **E0192 with score 0.9163**, flagged by the `network`, `statml`, and `behavioral` models — an example of a finding where multiple complementary signals agreed.

---

## Results

> These are the synthetic-data baseline numbers. For numbers graded on real, external, independently-labeled data, see [Real-World Validation](#real-world-validation).

The most up-to-date baseline in the repo is `results/current_baseline/eval.json` (a run on the synthetic sample set). From it:

| Metric | Value |
|---|---|
| recall@5 | 0.148 |
| precision@5 | 0.800 |
| recall@10 | 0.296 |
| precision@10 | 0.800 |
| Number of entities | 508 |
| Anomalies evaluated | 27 |

Per-type recall@10 (current baseline): `structuring` = 1.00, `fan_io` = 0.667, `post_burst` = 0.667, `silence` = 0.333; `chain`, `dormant_flip`, `colocation` = 0.00.

Surprise hold-out (current baseline): recall@10 = 0.067, precision@10 = 0.100, `structuring` = 1.00.

An earlier fix-cycle snapshot (`results/after_fixes/eval.json`) reports: recall@10 = 0.429, precision@10 = 0.900.

**Important caveat (do not misread these numbers):**

- The `GAP_AUDIT.md` documents that earlier reported numbers (e.g. 0.500/0.900) were **inflated by an amount-parsing bug** (`551.5` → `5515`); after the fix the deterministic baseline is the lower 0.148/0.296 values recorded above. This honesty about known bugs is deliberate.
- recall@10 is low because the default fusion (top-2 weighted or Borda) ranks the anomalous `behavioral`-heavy entities highly, while several rare typologies (`chain`, `dormant_flip`, `colocation`) with small populations are harder to surface in the top-10.
- The **structuring** detector is consistently strong (recall 1.0 in both the main and surprise sets).
- Precision@10 is reasonable (0.8–0.9), meaning the top of the list is mostly genuine anomalies.

These are the actual persisted numbers, not idealised figures — the project treats honest, reproducible evaluation as a core value.

---

## Limitations

- **Primary development is synthetic; real-world validation is a supplementary check, not the main evaluation.** The pipeline has been run blind against SAML-D, a real external AML dataset (see [Real-World Validation](#real-world-validation)), but both validation samples are still far more fraud-concentrated than SAML-D's true 0.104% rate — the reported precision numbers are not real-world deployment claims.
- **Chain-of-custody sealing is tamper-evident, not tamper-proof** — no cryptographic signing with a key the analyst doesn't also control, so it does not prove authorship to a third party.
- **Legal-basis framing has not been verified against specific statutory citations** — treat it as a design principle, not legal advice.
- **Recall@10 is modest on the default baseline.** Several rare anomaly typologies (`chain`, `dormant_flip`, `colocation`) are not recovered in the top-10 in the current `results/current_baseline` snapshot.
- **Entity resolution is per-source with raw-ID equality for cross-source linking** — there is no phonetic/canonicalisation step (documented in `GAP_AUDIT.md`).
- **Cross-source identity linking** relies on shared identifier equality plus fuzzy *names*; name collisions across sources are avoided but the linkage is not perfect.
- **Deep models are optional.** LSTM/TGN skip with a warning when `torch` is absent.
- **Unsupervised scoring** — there is no supervised "probability of being suspicious"; scores are relative anomaly scores, not calibrated probabilities.
- **Computational cost** of some network candidates (e.g., `time_correlation_anypair` O(n²), burst concentration) can be high on large files.
- **No real-time / streaming** detection — it is batch/file-based.
- **No web deployment or API** — the shipped interface is a desktop GUI and CLI (a FastAPI dashboard was planned but is not part of the shipped code).

---

## Future Improvements

**Already implemented (current functionality):** the full Stage A–G pipeline, the model zoo, sufficiency engine, fusion (Borda/weighted/meta), evaluation harness, report generation, desktop GUI, dataset/robustness tooling, real-world validation against SAML-D/PaySim, legal-basis provenance, and chain-of-custody sealing.

**Potential future improvements (not yet implemented):**

- **Real-world validation at realistic prevalence** — SAML-D validation currently runs at 21.7%/8.8% fraud concentration, both well above SAML-D's true 0.104% rate; closing that gap needs a sample (and entity-resolution runtime) an order of magnitude larger.
- **Better entity resolution** — add phonetic/canonicalisation and stronger cross-source linking.
- **Hyperparameter tuning** — a systematic tuning pass over contamination, tolerances, and fusion weights.
- **Improved fusion** — integrate the learned meta-learner as the default after broader validation.
- **Deep-learning escalation** — train and include LSTM/TGN by default with released weights.
- **Web/API deployment** — a FastAPI/Streamlit dashboard and REST API (planned in `PLAN.md` but not shipped).
- **Real-time/streaming** ingestion for live monitoring.
- **Explainable AI** — move from template explanations to SHAP/LIME-style feature attribution.
- **Database integration and model monitoring** — persist and monitor drift over time.
- **Calibrated probabilities** — convert relative scores into calibrated risk probabilities with external validation.

---

## Technical Highlights

- **Problem statement:** Cross-source AML anomaly detection where no fixed schema or single model suffices.
- **Dataset:** Synthetic two-tier data (predefined storylines + write-only surprise set) with complete ground truth; multi-format, deliberately messy exports.
- **Preprocessing:** Dynamic schema detection (RapidFuzz + content probes) → normalisation → entity resolution → dedup → shared feature vector → sufficiency gating.
- **Feature engineering:** Per-entity counts, amount stats, fan-in/out, temporal gaps, entropy, night/weekend ratios, structuring proximity, velocity.
- **Models:** Time correlation (`merge_asof`), network (OddBall/burst/PageRank/communities/layering), stat/ML (Isolation Forest + EIF/LOF/OCSVM/HBOS/GMM/PCA/Mahalanobis/autoencoder/KDE/z-score), Benford, structuring, behavioral, chain, plus optional deep LSTM/TGN.
- **Model comparison:** A documented model matrix and model-zoo evaluation; per-model score coverage and ablation (leave-one-model-out) verify fusion value (H1).
- **Evaluation:** recall@k, precision@k, per-type recall@10, ablation, surprise hold-out, deterministic seeds.
- **Best model observed:** The `structuring` rule is the most reliable single detector (type recall 1.0); no single model dominates all types, which is why the ensemble is used.
- **Visualization:** Matplotlib charts for timeline, watchlist, sources, ego-graphs, drill-downs, activity map, and model results, inlined into self-contained reports.
- **Prediction pipeline:** A single source-agnostic `Pipeline` that assembles only the models the data supports, fuses scores, and explains findings.
- **Challenges & solutions:** Amount-parsing bugs (fixed), mixed-format datetime parsing (unix-epoch + element-wise fallback), pandas 3.x dtype discipline, Windows torch import crashes (eager import + thread caps), undisclosed missing data (sufficiency engine).
- **Key learnings:** honesty about insufficient data beats silent degradation; a mixture of complementary unsupervised models plus robust ranking beats any single detector; adaptive schema handling is essential for real exports.

---

## How the Entire Project Works

```text
User
 ↓
Application (GUI drop files / CLI args)
 ↓
Loaders (CSV/JSON/JSONL/TXT, delimiter sniffing, encoding fallback)
 ↓
Stage A — Schema detection (what is in each file)
 ↓
Stage B — Entity resolution (who is who, across sources)
 ↓
Unified long-format dataframe
 ↓
Sufficiency engine (what can we honestly do with this data?)
 ↓
Active models run (time / network / stat-ML / benford / structuring / behavioral / chain)
 ↓
Fusion & ranking (one risk score per person)
 ↓
Explanation (why each person was flagged, in plain language)
 ↓
Report (Markdown / HTML / PDF / JSON + charts)
 ↓
User reads, exports, and revisits via History
```

In plain words: you give TraceWeave a pile of financial, call, and social files. It figures out what each column means on its own, links the same person across all three, decides which of its detectors it can honestly run on the data, scores every person with those detectors, merges the scores into one ranked list, and writes a readable report that says who to look at and *why* — backed by real records and charts, not unexplained model numbers.

---

## Conclusion

TraceWeave is a research-grade, source-agnostic, multi-model anomaly-detection engine for anti-financial-crime screening. It demonstrates how dynamic schema handling, entity resolution, a sufficiency engine, and a mixture of unsupervised models — fused and explained — can surface suspicious cross-source behaviour that single-source, fixed-schema monitors miss. The project is fully tested, reproducibly evaluated against hidden answer keys, and presented through both a CLI and a drag-and-drop desktop application, with an honest accounting of its strengths and limitations.

---

**Built as a hackathon/research project.** See `PLAN.md` for the research grounding, `SCHEMA.md` for the data contract, and `GAP_AUDIT.md` for the verified spec-vs-implementation audit and deterministic results.
