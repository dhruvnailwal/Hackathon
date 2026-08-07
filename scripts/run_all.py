"""Run every validation & evaluation script in dependency order.

Produces the complete evidence trail under results/:
  Stage A stress matrix (512 cases)       -> results/stage_a_stress/
  detection-mode benchmark                -> results/stage_a_bench/
  real-world dataset validation           -> results/real_datasets/
  internet-sample validation              -> data/internet_samples/ + report
  model-zoo comparison                    -> results/model_zoo/
  hyperparameter sweep                    -> results/tuning/
  baseline snapshot                       -> results/baseline/
  UI smoke screenshot                     -> results/ui_check.png

Usage:
    python scripts/run_all.py [--quick]     # --quick: skip stress/bench re-runs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(name: str, script: str, args: list[str] | None = None) -> int:
    print(f"\n===== {name} =====")
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / script)] + (args or [])
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    tag = "ok" if r.returncode == 0 else "FAILED"
    print(f"----- {name}: {tag} in {dt:.1f}s")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the expensive stress matrix and mode benchmark")
    args = ap.parse_args()

    rc = 0
    if not args.quick:
        rc |= run("Stage A stress matrix (512 cases)", "scripts/stage_a_stress.py")
        rc |= run("Detection-mode benchmark", "scripts/stage_a_bench.py")
    rc |= run("Real-world dataset validation", "scripts/real_dataset_validate.py")
    rc |= run("Internet-sample fetch + pipeline", "scripts/fetch_samples.py")
    rc |= run("Model-zoo comparison", "scripts/model_zoo_eval.py")
    rc |= run("Hyperparameter sweep", "scripts/tune.py")
    rc |= run("Baseline snapshot", "eval.py", ["--out", "results/baseline"])
    rc |= run("UI smoke screenshot", "main_app.py", ["--shot", "results/ui_check.png"])

    print("\n" + "=" * 60)
    print("ALL DONE" if rc == 0 else f"COMPLETED WITH FAILURES (rc={rc})")
    print("Evidence trail under results/:")
    for d in sorted(p for p in (ROOT / "results").iterdir() if p.is_dir()):
        print(f"  results/{d.name}/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
