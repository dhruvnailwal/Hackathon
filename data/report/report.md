# Anomaly Lens — analysis report
Generated 2026-08-05 20:12 UTC
Source files: `data/sources`

## Summary

| metric | value |
|---|---|
| events | 17,929 |
| entities | 418 |
| sources | 3 |
| cross-source links | 87 |
| blocked models | 0 |

## Source files

| file | source type | rows | entities | resolved slots | warnings |
|---|---|---|---|---|---|
| data\sources\bank_export.json | bank | 4,271 | 102 | actor_id, actor_name, amount, counterparty, event_type, location, timestamp |  |
| data\sources\cdr_export.jsonl | cdr | 13,338 | 87 | actor_id, actor_name, counterparty, duration, location, timestamp | cdr: no amount dimension (only call events carry it); cdr: no amount dimension (only call events carry it) |
| data\sources\social_export.txt | social | 320 | 316 | actor_id, counterparty, timestamp | social: no amount dimension (only post events carry it); social: no amount dimension (only post events carry it) |

## Models auto-activated

| model | verdict | reason |
|---|---|---|
| schema_detection | SUPPORTED | resolved 8 canonical slots |
| entity_resolution | SUPPORTED | counterparty/edge slots resolved |
| time_correlation | SUPPORTED | timestamp resolved; enables internal + cross-source correlation |
| network | SUPPORTED | counterparty/edge dimension resolved; cross-source graph available |
| statml | SUPPORTED | amount + timestamp resolved; full feature vector available |
| benford | SUPPORTED | amount resolved; first-digit deviation scoring available |
| structuring | SUPPORTED | amount resolved; threshold-proximity structuring rule available |
| behavioral | SUPPORTED | timestamp + amount resolved; regime-flip scoring available |

## Ranked insights

| rank | entity | score | n_events | models fired |
|---|---|---|---|---|
| 1 | E0095 | 1.000 | 64 | network, behavioral |
| 2 | E0034 | 1.000 | 256 | network, behavioral |
| 3 | E0075 | 0.983 | 50 | structuring, behavioral |
| 4 | E0012 | 0.982 | 196 | time_correlation, behavioral |
| 5 | E0002 | 0.978 | 388 | time_correlation, behavioral |
| 6 | E0097 | 0.926 | 64 | network, behavioral |
| 7 | E0100 | 0.896 | 74 | statml, behavioral |
| 8 | E0093 | 0.876 | 54 | structuring, behavioral |
| 9 | E0096 | 0.835 | 58 | behavioral |
| 10 | E0052 | 0.829 | 36 | behavioral |

## Explanations

- **E0124**: Fusion score 0.70. stat model: feature-space outlier vs peer population
- **E0026**: Fusion score 0.76. time model: cross-source call/post landed shortly before a transfer
- **E0095**: Fusion score 1.00. network model: unusual hub/density/role shift. stat model: feature-space outlier vs peer population. behavioral model: dormancy duration or silence-after-large-transfer regime flip
- **E0096**: Fusion score 0.84. behavioral model: dormancy duration or silence-after-large-transfer regime flip
- **E0220**: Fusion score 0.70. stat model: feature-space outlier vs peer population
- **E0089**: Fusion score 0.77. time model: cross-source call/post landed shortly before a transfer
- **E0075**: Fusion score 0.98. stat model: feature-space outlier vs peer population. structuring model: repeated amounts clustered just under the reporting threshold. behavioral model: dormancy duration or silence-after-large-transfer regime flip
- **E0023**: Fusion score 0.75. time model: cross-source call/post landed shortly before a transfer
- **E0051**: Fusion score 0.79. behavioral model: dormancy duration or silence-after-large-transfer regime flip
- **E0052**: Fusion score 0.83. behavioral model: dormancy duration or silence-after-large-transfer regime flip
