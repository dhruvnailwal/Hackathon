# SCHEMA.md — Unified Dataframe Specification (FREEZE POINT)

> **Guardrail** (plan §5): nobody changes the columns below or the canonical field
> slots after hour 2. All model owners build against this fixture.

## Canonical long-format dataframe

Every internal stage reads/writes the SAME table. Columns (all `object`/typed as noted):

| Column | Dtype/Type | Meaning | Populated by |
|---|---|---|---|
| `entity_id` | str | resolved person ID | Stage B |
| `event_type` | str (`call`/`transaction`/`post`) | kind of event | Stage A |
| `timestamp` | `datetime64[ns]` | event time | Stage A |
| `source` | str (`bank`/`cdr`/`social`) | origin file | Stage A |
| `amount` | float (nullable) | value; only bank | Stage A |
| `counterparty_id` | str (nullable) | resolved other-party ID | Stage B |
| `location` | str (nullable) | tower/branch/geo tag | Stage A |
| `direction` | str (`in`/`out`, nullable) | bank debit/credit | Stage A |
| `duration` | float (nullable) | call seconds | Stage A |

### Field slots (canonical slot -> resolved column)
- `timestamp` : datetime
- `amount` : float
- `counterparty` : str
- `event_type` : str
- `location` : str
- `direction` : str
- `duration` : float

### Source capability (which slots a source can genuinely fill)
- bank   : timestamp, amount, counterparty, location, direction
- cdr    : timestamp, counterparty, location, duration
- social : timestamp, counterparty(partial), location(text/tag)

## Anomaly types (ground-truth labels used across stages)
- `colocation`  : cross-source temporal link (call/post shortly before transfer)
- `dormant_flip`: dormant entity abruptly becomes active
- `structuring` : amount splitting just below reporting threshold
- `fan_io`      : fan-in / fan-out structural pattern
- `silence`     : silence after large transaction (behavioral regime flip)
- `background`  : normal (the write-only answer-key class)