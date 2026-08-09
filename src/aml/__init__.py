"""Multi-model anomaly detection engine.

A source-agnostic, modular pipeline that detects anomalous patterns across
CDR, bank-transaction, and social-media data via a mixture of models over a
single long-format dataframe (see SCHEMA.md).
"""

import os

# Windows: torch's bundled MKL thread pool can crash against the pools
# spawned by pandas/sklearn/numpy in the same process (intermittent access
# violations). Cap the pools before any heavy library is imported and allow
# duplicate OpenMP runtime names (xgboost/sklearn may load their own copy).
for _v in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch up front: on Windows the lazy mid-process import can crash
# (access violation inside torch's own import chain) once pandas/sklearn
# have already loaded. Eager import in a fresh process is reliable.
import sys as _sys
_TORCH_OK = False
for _attempt in range(3):
    try:
        import torch  # noqa: F401
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _TORCH_OK = True
        break
    except Exception as _e:  # pragma: no cover - environment dependent
        import time as _time
        print(f"torch import attempt {_attempt + 1} failed: {_e!r}", file=_sys.stderr)
        _time.sleep(1)
if not _TORCH_OK:
    print("warning: torch unavailable; deep models (LSTM/TGN) will be skipped",
          file=_sys.stderr)

__version__ = "0.1.0"