from aml.schema import SourceResolved
from aml.sufficiency import (
    VERDICT_BLOCKED,
    VERDICT_DEGRADED,
    VERDICT_SUPPORTED,
    SufficiencyEngine,
)


def _rs(source, fields, n_rows=2000, n_entities=60):
    return SourceResolved(
        source=source, file=f"{source}.csv",
        resolved_fields=set(fields), n_rows=n_rows, n_entities=n_entities,
    )


def test_bank_only():
    engine = SufficiencyEngine([_rs("bank", ["timestamp", "amount", "counterparty"])])
    v = engine.evaluate()
    assert v["benford"].status == VERDICT_SUPPORTED
    assert v["structuring"].status == VERDICT_SUPPORTED
    assert v["statml"].status == VERDICT_SUPPORTED
    assert v["network"].status == VERDICT_DEGRADED
    assert v["time_correlation"].status == VERDICT_SUPPORTED
    assert v["behavioral"].status == VERDICT_SUPPORTED


def test_cdr_only():
    engine = SufficiencyEngine([_rs("cdr", ["timestamp", "counterparty", "duration"])])
    v = engine.evaluate()
    assert v["benford"].status == VERDICT_BLOCKED
    assert v["structuring"].status == VERDICT_BLOCKED
    assert v["statml"].status == VERDICT_DEGRADED
    assert v["network"].status == VERDICT_DEGRADED


def test_social_only():
    engine = SufficiencyEngine([_rs("social", ["timestamp", "counterparty"])])
    v = engine.evaluate()
    assert v["benford"].status == VERDICT_BLOCKED
    assert v["behavioral"].status == VERDICT_BLOCKED


def test_no_timestamp_blocks_time_correlation():
    engine = SufficiencyEngine([_rs("bank", ["amount", "counterparty"])])
    v = engine.evaluate()
    assert v["time_correlation"].status == VERDICT_BLOCKED


def test_multisource_network_supported():
    engine = SufficiencyEngine([
        _rs("bank", ["timestamp", "amount", "counterparty"]),
        _rs("cdr", ["timestamp", "counterparty", "duration"]),
    ])
    v = engine.evaluate()
    assert v["network"].status == VERDICT_SUPPORTED


def test_low_volume_degrades_time_correlation():
    engine = SufficiencyEngine(
        [_rs("bank", ["timestamp", "amount"], n_rows=40, n_entities=30)],
        min_events_per_entity=30,
    )
    v = engine.evaluate()
    assert v["time_correlation"].status == VERDICT_DEGRADED


def test_empty_inputs_block_entity_resolution():
    engine = SufficiencyEngine([])
    v = engine.evaluate()
    assert v["entity_resolution"].status == VERDICT_BLOCKED
