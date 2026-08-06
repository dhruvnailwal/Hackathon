import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """Generate a small mixed-format (CSV+JSON+JSONL+TXT) synthetic dataset."""
    from aml.data_generator.emit_multi import generate_all_mixed, make_social_text

    out = tmp_path_factory.mktemp("synth")
    generate_all_mixed(out, seed=11, surprise_seed=22, n_background=30,
                       n_per_storyline=2, n_surprise=4)
    make_social_text(out, n=60)
    return out
