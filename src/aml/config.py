"""Pipeline configuration loaded from YAML (no hardcoded demo values in code)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    sources_dir: str = "data/sources"
    report_dir: str = "data/report"

    top_k: int = 10
    min_events_per_entity: int = 30
    tol_minutes: int = 30
    structuring_threshold: float = 10000.0
    with_models: bool = True
    seeds: List[int] = field(default_factory=lambda: [42, 99])

    # Stage-A content-probe thresholds (see aml.schema_detect.ProbeConfig)
    probe_date_parse_ratio: float = 0.9
    probe_phone_ratio: float = 0.7
    probe_numeric_ratio: float = 0.9
    probe_id_ratio: float = 0.7
    probe_amount_median_abs: float = 100.0
    probe_amount_frac_ratio: float = 0.3
    probe_amount_marker_ratio: float = 0.5
    probe_big_code_magnitude: float = 10000.0
    probe_amount_integer_median: float = 100.0

    def __post_init__(self) -> None:
        self._path: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        p = Path(path)
        cfg = cls()
        if not p.exists():
            return cfg
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        for key, val in raw.get("pipeline", {}).items():
            if key in cfg.__dataclass_fields__:
                setattr(cfg, key, val)
        cfg._path = str(p)
        return cfg

    def probe_config(self):
        """Build the Stage-A content-probe thresholds from this config."""
        from .schema_detect import ProbeConfig
        return ProbeConfig(
            date_parse_ratio=self.probe_date_parse_ratio,
            phone_ratio=self.probe_phone_ratio,
            numeric_ratio=self.probe_numeric_ratio,
            id_ratio=self.probe_id_ratio,
            amount_median_abs=self.probe_amount_median_abs,
            amount_frac_ratio=self.probe_amount_frac_ratio,
            amount_marker_ratio=self.probe_amount_marker_ratio,
            big_code_magnitude=self.probe_big_code_magnitude,
            amount_integer_median=self.probe_amount_integer_median,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump({"pipeline": asdict(self)}, sort_keys=False),
            encoding="utf-8",
        )