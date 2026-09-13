#!/usr/bin/env python3
"""Compare regenerated paper figures with the manuscript reference images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "paper_figures" / "reference"
GENERATED = ROOT / "paper_figures" / "generated"

# Text rasterization varies slightly across Matplotlib, FreeType, and operating
# system versions. These limits catch missing curves, shifted layouts, or stale
# data while permitting small antialiasing differences.
MAX_MEAN_ABSOLUTE_ERROR = 0.02
MAX_LARGE_DIFFERENCE_FRACTION = 0.05
LARGE_DIFFERENCE = 0.10


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    references = sorted(REFERENCE.glob("*.png"))
    if not references:
        raise SystemExit("No reference PNGs found.")

    expected_names = {path.name for path in references}
    generated_names = {path.name for path in GENERATED.glob("*.png")}
    missing = sorted(expected_names - generated_names)
    if missing:
        raise SystemExit("Missing generated figures: " + ", ".join(missing))

    failures: list[str] = []
    for reference_path in references:
        generated_path = GENERATED / reference_path.name
        reference = load_rgb(reference_path)
        generated = load_rgb(generated_path)
        if reference.shape != generated.shape:
            failures.append(
                f"{reference_path.name}: dimensions differ "
                f"({reference.shape} versus {generated.shape})"
            )
            continue

        difference = np.abs(reference - generated)
        mean_error = float(difference.mean())
        large_fraction = float((difference > LARGE_DIFFERENCE).mean())
        print(
            f"[ok] {reference_path.name}: mean error {mean_error:.4f}; "
            f"large-difference pixels {large_fraction:.2%}"
        )
        if mean_error > MAX_MEAN_ABSOLUTE_ERROR:
            failures.append(
                f"{reference_path.name}: mean error {mean_error:.4f} exceeds "
                f"{MAX_MEAN_ABSOLUTE_ERROR:.4f}"
            )
        if large_fraction > MAX_LARGE_DIFFERENCE_FRACTION:
            failures.append(
                f"{reference_path.name}: large-difference fraction "
                f"{large_fraction:.2%} exceeds "
                f"{MAX_LARGE_DIFFERENCE_FRACTION:.2%}"
            )

    if failures:
        raise SystemExit("Figure comparison failed:\n" + "\n".join(failures))
    print(f"All {len(references)} manuscript figures match within tolerance.")


if __name__ == "__main__":
    main()
