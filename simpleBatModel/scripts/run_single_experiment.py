from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_experiment import run_all  # noqa


if __name__ == "__main__":
    run_all(ROOT / "cases" / "single_experiment.yaml")
