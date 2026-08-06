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
    tol_minutes: int = 25
    structuring_threshold: float = 10000.0
    with_models: bool = True
    seeds: List[int] = field(default_factory=lambda: [42, 99])

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump({"pipeline": asdict(self)}, sort_keys=False),
            encoding="utf-8",
        )