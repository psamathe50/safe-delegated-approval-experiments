# Safe Delegated Approval experiments

This repository is a standalone reproduction bundle for the paper's one-step
reviewer-panel experiments. It contains the pinned score archives, analysis
code, verification checks, and portable renderers for the experimental
figures. The analysis makes no model or network calls.

The repository is private while it is being audited. It is structured so it
can later be made public after the release checklist is completed.

## Quick start

Python 3.9 through 3.11 is supported; Python 3.11 is recommended. The exact
package versions below match the paper-figure rendering environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make verify PYTHON=.venv/bin/python
make figure-check PYTHON=.venv/bin/python
```

`make verify` checks the pinned inputs and the committed derived results. The
figure command writes the paper figures to `paper_figures/generated/` and
compares them with the versions currently included in the manuscript.

To recompute all derived results from the pinned score archives, regenerate
the figures, and run the release scan:

```bash
make reproduce PYTHON=.venv/bin/python
```

The full CPU-only analysis typically takes five to ten minutes on a recent
laptop.
The manual `full reproduction` GitHub Actions workflow runs the same command
on a clean Linux worker and retains the regenerated paper figures as an
artifact.

## Repository map

- `experiments/one_step_panel/`: canonical analysis package and pinned data.
- `paper_figures/`: manuscript-facing figure renderers, generated outputs,
  and the current reference PDFs and PNGs.
- `scripts/check_public_release.py`: conservative scan for local paths,
  credentials, and other release hazards.
- `PUBLIC_RELEASE_CHECKLIST.md`: remaining steps before changing visibility.

The detailed experiment definitions, split policy, exact lottery audits,
headline checks, and input checksums are documented in
`experiments/one_step_panel/README.md`.

## Reproducibility policy

Weights, regularization, and reviewer identities are selected without test
labels. Threshold families are defined from non-test score values, then
evaluated descriptively on held-out prompts. The topic-calibration comparison
jointly selects five topic thresholds on validation prompts under a pooled
validation soundness-failure budget; its test set is used only for final
evaluation.

Reference PDFs are retained to make review convenient. PDF byte hashes are not
used as the correctness criterion because plotting and PDF metadata can vary
across platforms. The verification suite checks inputs and scientific
quantities, while the pinned Matplotlib version minimizes visual drift.

## Data and licensing

The bundled NPZ files contain public model scores, public model identifiers,
hashed prompt keys, and benchmark metadata. They do not contain raw prompt or
response text. Dataset redistribution terms and the repository's source-code
license still need to be finalized before public release; see
`PUBLIC_RELEASE_CHECKLIST.md`.
