"""Single import boundary for the frozen algorithm implementation."""

from __future__ import annotations

import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = REPOSITORY_ROOT / "algorithms"
os.environ.setdefault("MPLBACKEND", "Agg")

# The original algorithm modules use sibling imports (``import segmentation``
# and ``import rectangle_coverage``).  Keeping this directory on sys.path lets
# us relocate those files without modifying their contents or import behavior.
algorithm_path = str(ALGORITHMS_DIR)
if algorithm_path not in sys.path:
    sys.path.insert(0, algorithm_path)

import build_graph as bg  # noqa: E402
import connectivity_methods as conn_methods  # noqa: E402
import rectangle_coverage as rc  # noqa: E402
import segmentation as seg  # noqa: E402
import segmentation_methods as seg_methods  # noqa: E402


__all__ = ["bg", "conn_methods", "rc", "seg", "seg_methods"]
