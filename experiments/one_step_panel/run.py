"""Reproduce the one-step reviewer-panel experiments and paper figures.

Run from any working directory with::

    python experiments/one_step_panel/run.py

The script reads the two pinned score archives in ``data/`` and rewrites the
derived depth archive, JSON summaries, and figures. The test split is never
used to fit weights or choose regularization. Reported held-out frontiers vary
the global threshold descriptively; they are not deployable operating points
selected without test labels.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
TEMP_CACHE = tempfile.TemporaryDirectory(prefix="coalition-alignment-cache-")
TEMP_CACHE_PATH = Path(TEMP_CACHE.name)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(TEMP_CACHE_PATH / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_PATH / "xdg"))

from alignment_analysis import main as run_alignment_analysis
from approximate_analysis import main as run_approximate_analysis
from binary_analysis import main as run_binary_analysis
from cardinal_analysis import main as run_cardinal_analysis
from depth_analysis import main as run_depth_analysis
from oracle_analysis import main as run_oracle_analysis
from topic_distribution_figure import main as run_topic_distribution_figure


REQUIRED_INPUTS = {
    "rewardbench2_scores.npz": (
        "2b6f6cfa2c6cace3d85e1bc96b5a8c61390b3b91a2bb73f67bd6e89849ed0d7c"
    ),
    "strongreject_scores.npz": (
        "e462848048763dc24640535baa2ce2bf9ed9834535faf323c2b73b8ea5396d48"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_inputs() -> None:
    missing = [
        name for name in REQUIRED_INPUTS if not (HERE / "data" / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing experiment inputs: " + ", ".join(sorted(missing))
        )
    for name, expected in REQUIRED_INPUTS.items():
        observed = sha256(HERE / "data" / name)
        if observed != expected:
            raise RuntimeError(
                f"input checksum mismatch for {name}: expected {expected}, "
                f"observed {observed}"
            )


def main() -> None:
    if sys.version_info < (3, 9):
        raise RuntimeError("the experiment runner requires Python 3.9 or later")
    check_inputs()
    (HERE / "plots").mkdir(exist_ok=True)
    stages = (
        ("exact RewardBench 2 depth", run_depth_analysis),
        ("bounded-domain approximate errors", run_approximate_analysis),
        ("individual-alignment diagnostics", run_alignment_analysis),
        ("prompt-specific oracle aggregation", run_oracle_analysis),
        ("binary count-rule figures", run_binary_analysis),
        ("held-out weighted-rule analysis", run_cardinal_analysis),
        ("topic-level distribution figure", run_topic_distribution_figure),
    )
    total_started = time.perf_counter()
    for index, (label, function) in enumerate(stages, start=1):
        started = time.perf_counter()
        print(f"\n[{index}/{len(stages)}] {label}", flush=True)
        function()
        print(
            f"[{index}/{len(stages)}] completed in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    print(
        "\nReproduced and checked one-step panel experiments "
        f"({time.perf_counter() - total_started:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
