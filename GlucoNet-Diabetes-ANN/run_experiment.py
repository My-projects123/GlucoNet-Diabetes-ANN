#!/usr/bin/env python3
"""Beginner-friendly entry point. Does not download data.

Usage (from this folder, with the virtualenv active):

    python run_experiment.py
    python run_experiment.py --profile-only
    python run_experiment.py --seed 42 --splits 5 --select-k 5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diabetes_prediction.run import main

if __name__ == "__main__":
    raise SystemExit(main())
