# Third-party data and attribution

This repository's MIT License applies to the original source code in this
repository. It does not replace or modify the licenses, terms, or attribution
requirements of the datasets, model outputs, or software on which the
experiments are based.

The bundled NPZ archives contain numerical scores, public model identifiers,
hashed prompt keys, and benchmark metadata. They do not contain raw prompt or
response text, model weights, or third-party source code.

## RewardBench 2

The RewardBench 2 archive was derived from:

- the [RewardBench 2 evaluation dataset](https://huggingface.co/datasets/allenai/reward-bench-2)
  at revision `7ff08853b0d5686e79b13fda8677024f566a104a`; and
- the [RewardBench 2 results archive](https://huggingface.co/datasets/allenai/reward-bench-2-results)
  at revision `b19c7033e964187d12e74a43a07f2d727a3d37e5`.

The evaluation dataset is released under the
[Open Data Commons Attribution License (ODC-By)](https://opendatacommons.org/licenses/by/1-0/)
and is intended for research and educational use in accordance with Ai2's
[Responsible Use Guidelines](https://allenai.org/responsible-use). The dataset
card notes that included third-party model outputs remain subject to their own
terms. The [RewardBench software](https://github.com/allenai/reward-bench) is
released under the Apache License 2.0. Users should consult the source dataset,
results archive, and relevant model-provider terms for their intended use.

Please cite RewardBench 2:

```bibtex
@misc{malik2025rewardbench2advancingreward,
  title         = {RewardBench 2: Advancing Reward Model Evaluation},
  author        = {Saumya Malik and Valentina Pyatkin and Sander Land and Jacob Morrison and Noah A. Smith and Hannaneh Hajishirzi and Nathan Lambert},
  year          = {2025},
  eprint        = {2506.01937},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2506.01937}
}
```

## StrongREJECT

The StrongREJECT archive was derived from the official processed
`labelbox.csv` and `labelbox_evals.csv` releases in the
[StrongREJECT repository](https://github.com/dsbowen/strong_reject). The
StrongREJECT software is released under the MIT License. Its official README
also identifies several source datasets with separate licenses and notes that
some source datasets have no stated license. Users should review those
upstream notices before redistributing or repurposing StrongREJECT-derived
material.

Please cite StrongREJECT:

```bibtex
@inproceedings{souly2024strongreject,
  title     = {A Strong{REJECT} for Empty Jailbreaks},
  author    = {Alexandra Souly and Qingyuan Lu and Dillon Bowen and Tu Trinh and Elvis Hsieh and Sana Pandey and Pieter Abbeel and Justin Svegliato and Scott Emmons and Olivia Watkins and Sam Toyer},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year      = {2024}
}
```

## Software dependencies

Third-party Python packages are installed separately from PyPI and are not
vendored in this repository. Each package remains subject to its own license;
consult the package metadata and upstream project for details.
