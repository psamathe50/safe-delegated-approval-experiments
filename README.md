# Safe Delegated Approval experiments

This repository contains the code and pinned score archives for the one-step
reviewer-panel experiments in *Delegating Authorization to Misaligned Agents:
Coalitional Alignment and Safe Control*. It reproduces all eight experimental
figures in the current paper (Figures 4--11).

The analysis is self-contained after dependency installation: it makes no
model, API, or network calls. A CPU is sufficient; no GPU is required.

## What this repository reproduces

| Paper figure | Output | Content |
| --- | --- | --- |
| Figure 4 | `revised_count_rules` | Exact count-rule soundness and completeness |
| Figure 5 | `revised_authorization_comparison` | Pure-response and lottery authorization frontiers |
| Figure 6 | `revised_approximate_count` | Approximate count-rule guarantees |
| Figure 7 | `revised_rewardbench_topics` | RewardBench 2 deterministic results by topic |
| Figure 8 | `revised_full_topics` | RewardBench 2 lottery results by topic |
| Figure 9 | `revised_cardinal_appendix` | Full lottery cardinal and binary comparisons |
| Figure 10 | `revised_margin_sensitivity` | Completeness-margin sensitivity |
| Figure 11 | `revised_topic_calibration` | Shared versus topic-specific calibration |

Each output is written as both PDF and PNG under
`paper_figures/generated/`. The corresponding manuscript versions are kept in
`paper_figures/reference/` for automated comparison.

## Requirements

- Python 3.9--3.11 (Python 3.11 recommended)
- `make` and a POSIX-compatible shell for the commands below
- Approximately 1 GB of free memory

Windows users can run the underlying Python commands directly; see
`experiments/one_step_panel/README.md` for PowerShell setup notes.

## Quick reproduction

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make verify PYTHON=.venv/bin/python
make figure-check PYTHON=.venv/bin/python
```

`make verify` checks the integrity of the pinned inputs, committed scientific
quantities, split policy, and headline results. `make figure-check` regenerates
the eight paper figures and compares them with the manuscript versions. Small
cross-platform font-antialiasing differences are allowed; changes to figure
dimensions or content fail the check.

The generated figures can then be opened from:

```text
paper_figures/generated/
```

## Full reproduction from pinned scores

To recompute every derived result from the archived model scores before
regenerating and checking the figures, run:

```bash
make reproduce PYTHON=.venv/bin/python
```

This is the strongest local reproduction command. It runs the complete
analysis, verifies the results, renders and compares all paper figures, and
checks tracked files for common credentials and machine-specific paths. The
CPU-only run typically takes five to ten minutes on a recent laptop. The first
exact-depth stage may be quiet for several minutes.

The manual **full reproduction** GitHub Actions workflow runs the same command
on a clean Linux worker and uploads the regenerated figures as a workflow
artifact.

## Repository structure

- `experiments/one_step_panel/` -- analysis code, pinned inputs, derived
  results, and detailed experiment documentation
- `paper_figures/` -- manuscript figure renderers, references, and generated
  outputs
- `scripts/compare_figures.py` -- cross-platform visual comparison
- `scripts/check_public_release.py` -- portability and credential scan
- `.github/workflows/` -- fast verification and full-reproduction workflows

For experiment definitions, exact lottery-domain calculations, train/
validation/test handling, input checksums, and output descriptions, see
`experiments/one_step_panel/README.md`.

## Evaluation protocol

Weights, regularization, and reviewer identities are selected without test
labels. Threshold families are constructed from non-test score values and then
evaluated descriptively on held-out prompts. The topic-calibration experiment
selects five topic thresholds jointly on validation prompts under a pooled
soundness-failure budget; the test split is used only for final evaluation.

The bundled archives contain public model scores, public model identifiers,
hashed prompt keys, and benchmark metadata. They contain no raw prompt or
response text. Input files are checked by SHA-256 before analysis.

## Reproducibility notes

The verification suite separately checks the scientific quantities and the
rendered figures. PDF files are not compared byte-for-byte because creation
timestamps and other PDF metadata can differ across platforms; PNG renderings
are used for the visual comparison instead. Core numerical and plotting
packages are pinned in `requirements.txt`.
