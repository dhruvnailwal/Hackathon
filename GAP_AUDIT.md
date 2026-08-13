# Gap Audit — Multi-Model Anomaly Detection Engine

Audit of the implemented engine against `PLAN.md` Stage A–G (schema detection,
entity resolution, sufficiency engine, models, fusion, evaluation) plus the §3
dirty-data robustness matrix. Verified 2026-08-11: **pytest 92 passed / 2
skipped; dirty-data audit 14/14 pass (exit 0); eval deterministic.**

## 1. Gap table — `PLAN.md` spec vs. implementation

| Stage | Specced in PLAN.md | Implemented | Gap / risk | Status |
|---|---|---|---|---|
| A. Schema detection  (§2.2) | Header matching (fuzzy), content probing, per-stage detection modes | `schema_detect.detect_columns` (hybrid/header/content + `ProbeConfig`), `_best_alias` rapidfuzz, `_content_probe` 8-slot probe chain, unix-epoch + day-first + mixed-format datetime parsing, amount parser (symbols/thousands/parens/comma-decimal) | Values whose header is absent AND content is ambiguous (e.g. integer account IDs vs counterparty IDs, §2.2 noted) still resolve by magnitude heuristic; documented in `det.columns` method strings. | implemented + hardened |
| A. Normalisation | Canonical long format event stream per `SCHEMA.md` | `normalize_source`: float64 NaN dtype discipline, `na_action="ignore"` maps (pandas 3.0.5), honest per-slot warnings, tz-normalised timestamps | — | implemented |
| L. Loaders | Handle real export quirks | CSV/JSON/JSONL/TXT; latin-1 fallback after `UnicodeDecodeError`; delimiter sniffing (`;`, `|`) | — | implemented |
| B. Entity resolution | Named + ID matching (fuzzy), alias handling | `entity_resolve.resolve_entities` (ID + name similarity, per-source actor fallback chain) | Resolution is per-source; cross-source (bank vs cdr) identity linking relies on raw-ID equality, no phonetic/canonicalisation. | partial |
| C/D/E. Sufficiency engine (§1B) | Source capability registry, model activation matrix, data-sufficiency verdicts, failure-mode contract | `sufficiency.py` registry (`SOURCE_CAPABILITIES`, activation matrix `MODEL_ACTIVATION`), verdicts SUPPORTED/DEGRADED/BLOCKED with reasons; single-source → DEGRADED, social-only → BLOCKED, <30 events/entity → BLOCKED, honest empty-score warnings | — | implemented |
| Models (§2.3–2.6, §3) | time_correlation (split-window / merge_asof), network (degree/fan-in/out, chain, burst), statml (IF/EIF/LOF/OCSVM/GMM/HBOS/Mahalanobis/PCA), benford, structuring, behavioral; deep LSTM/TGN optional | `models.py` all implemented; per-model entry in `run_model`; torch-gated LSTM/TGN with warn-skip | torch not installed in demo env → deep models skipped (documented warning at runtime, not silent) | implemented; deep optional |
| F. Fusion (§2.6) | Ranking/fusion with explanation | `fusion_meta.py` Borda + learned RF/XGBoost weights (`scripts/train_fusion_meta.py`), per-entity `fired` model list | — | implemented |
| G. Evaluation harness (§4.2/4.3) | recall/precision@k on storyline + surprise hold-out | `eval.py`: main + `--surprise-key`, per-type recall@10, ablation (leave-one-model-out), H1 standalone per-model check, surprise-leak exclusion | — | implemented |
| Metrics (baseline) | — | main recall@10=0.444 / precision@10=0.800; surprise recall@10=0.167 / precision@10=0.200 | Earlier recorded 0.500/0.900 was inflated by an amount-parsing bug (`551.5` → `5515`); fixed, new numbers deterministic. | fixed |

## 2. Prioritized code-change list

Priority 1 — silent-wrong / crash fixes (done in this session):

1. `src/aml/schema_detect.py::_parse_amount_value` — single-dot branch only
   treats a fraction of exactly 3 digits as dot-thousands; float reprs
   (`130.85000000000002`) and 1-decimal values (`551.5`) no longer inflate
   ×10/×1e14. Demo metrics moved 0.500→0.444 (the bug had been inflating 78
   of 4271 bank amounts).
2. `src/aml/schema_detect.py::_parse_datetime_series/_parse_one_datetime` —
   unix-epoch (s/ms/us) auto-unit detection for numeric columns (was 1970 ns-
   epoch garbage); element-wise fallback for mixed-format columns; all
   timestamps tz-naive so `pd.to_datetime` over a mixed tz column no longer
   NaTs entire series (pandas 3.x).
3. `src/aml/schema_detect.py::normalize_source` — `astype("string")` +
   `na_action="ignore"` for amount parsing (pandas 3.0.5 StringDtype NA) and
   `fillna(np.nan).astype(float)` (currency fixture crashed on `pd.NA`).
4. `src/aml/loaders.py` — `UnicodeDecodeError` → latin-1 fallback + warning;
   sniffed-delimiter warning; previously a latin-1 file crashed the pipeline.
5. `src/aml/schema.py::ACTOR_ID_ALIASES` — ES/FR/DE/PT headers (`cuenta`,
   `compte`, `konto`, `conta`, …); Spanish export previously lost the entity
   dimension and honest-BLOCKED every stage.

Priority 2 — §1B honesty (done):

6. `src/aml/sufficiency.py` — single-source `time_correlation` DEGRADED;
   social-only `network` BLOCKED; <30 events/entity BLOCKED with explicit
   `n=<k>` reason; honest DEGRADED when a stage's core slot is absent.
7. `src/aml/pipeline.py::_run_models` — non-BLOCKED model returning empty
   scores appends a warning instead of vanishing silently.
8. `eval.py` — exclusion of surprise type from the main answer key
   (generator writes surprise persons into both keys); H1 per-model check.

Priority 3 — regression protection (done):

9. `scripts/make_dirty_fixtures.py` + `scripts/dirty_audit.py` + 14 fixtures
   in `data/dirty_samples/` — the permanent §3 matrix (see table below);
   audit exits 1 on any crash/silent-wrong.

Open items (accepted, documented):

10. Deep models (LSTM/TGN) skip when torch is absent — warn, never crash.
11. Cross-source entity linking is ID-equality based; canonicalisation not
    implemented (out of §3 scope).
12. DD/MM vs MM/DD ambiguity resolves month-first when both parse (US
    exports win, §2.2 heuristic documented in code).

## 3. Dirty-data scenario matrix (§3) — all verified (a)

| Fixture | Scenario | Classification | Notes |
|---|---|---|---|
| `renamed_mixed_case.csv` | mixed-case / spaced headers, alias coverage | (a) ok | benford SUPPORTED, network DEGRADED (single source) |
| `headers_es.csv` | Spanish headers | (a) ok | entity dimension resolves; stages SUPPORTED |
| `unix_epoch.csv` | epoch-seconds timestamp | (a) ok | timestamps 2025-01-01T00:00, not 1970 |
| `mixed_dates.csv` | ISO + DD/MM + Z-suffix + epoch-ms + junk in ONE column | (a) ok | 87.5% parsed; junk rows dropped with honest warning |
| `currency_mess.csv` | `$`, parens, comma-decimal, codes, junk | (a) ok | 10/12 formats parse; unparseable rows NaN + warning |
| `missing_values.csv` | '' / NaN / N/A / - / 0-sentinel | (a) ok | dtype stays float64 |
| `dup_columns.csv` | `amount` vs `Amount ` | (a) ok | header match resolves; no double-count |
| `no_header.csv` | headerless, content-probe only | (a) ok | bank detected, amounts parsed, 5 entities |
| `latin1.csv` | cp1252 encoded | (a) ok | decode fallback + warning |
| `semicolon.csv` | `;` delimited | (a) ok | sniffed + warning |
| `sparse.csv` | 12 events/entity | (a) ok | all entity-scored stages BLOCKED (<30) |
| `bank_only.csv` | single source | (a) ok | time_correlation/network DEGRADED, rest SUPPORTED |
| `cdr_only.csv` | tower/lac location slots | (a) ok | durations resolve; benford/structuring BLOCKED (no amount) |
| `social_only.csv` | @handles, no amounts | (a) ok | network BLOCKED; no fake projections |

Re-run: `python scripts/make_dirty_fixtures.py && python scripts/dirty_audit.py`
(exit 1 = a (b) or (c) regression).