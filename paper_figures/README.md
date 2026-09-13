# Paper figure reproduction

The renderers in this directory regenerate the eight experimental figures
used by the current manuscript. They read only the committed artifacts under
`experiments/one_step_panel/data/` and write to `generated/`.

| Figure stem | Renderer | Principal input |
| --- | --- | --- |
| `revised_count_rules` | `render_count_figures.py` | `rewardbench2_depth.npz` |
| `revised_rewardbench_topics` | `render_count_figures.py` | `rewardbench2_depth.npz` |
| `revised_full_topics` | `render_count_figures.py` | `strongreject_scores.npz` |
| `revised_approximate_count` | `render_approximate_figure.py` | `approximate_results.npz` |
| `revised_authorization_comparison` | `render_authorization_figures.py` | `cardinal_results.json` |
| `revised_cardinal_appendix` | `render_authorization_figures.py` | `cardinal_results.json` |
| `revised_margin_sensitivity` | `render_authorization_figures.py` | `cardinal_results.json` |
| `revised_topic_calibration` | `render_topic_calibration.py` | `topic_threshold_budget_results.json` |

The files under `reference/` are the figure versions currently included in
the manuscript. Run `make figure-check` from the repository root to regenerate
all figures and compare their PNG renderings. The comparison permits small
font-antialiasing differences across platforms while failing on dimension or
substantive visual changes. Scientific quantities are checked separately by
`experiments/one_step_panel/verify.py`.
