# One-step panel experiments

This directory is the self-contained analysis bundle for the one-step panel
experiments. It uses two pinned public score archives and makes no model or
network calls during the analysis.

## Quick start

From the repository root in a POSIX shell:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r experiments/one_step_panel/requirements.txt
python3 experiments/one_step_panel/run.py
python3 experiments/one_step_panel/verify.py
```

On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` instead of the `source`
command and replace `python3` with `python` if needed. Runtime paths are
derived from this checkout, and no machine-specific data path or scheduler
configuration is required.

The complete analysis took 6 minutes 24 seconds on a 2026 Mac laptop in the
tested environment. It is CPU-bound, needs less than 1 GB of input data, and
does not benefit materially from a GPU. The first exact-depth stage can be
quiet for several minutes; `run.py` prints a stage banner before it starts.

The analysis requires Python 3.9 or later. It was last tested with Python
3.9.6, NumPy 1.24.2, SciPy 1.13.1, and Matplotlib 3.9.4.

For a one-command reproduction after installing the dependencies, run:

```bash
make -C experiments/one_step_panel all
```

## What is being evaluated

Both benchmarks are interpreted as one-step authorization problems.

- **StrongREJECT:** the proposal releases one recorded response and the
  fallback abstains. The median human harm score defines the principal's
  utility; nine released evaluators vote using their own harm scores.
- **RewardBench 2:** the proposal is a lottery over four recorded responses
  and the fallback is the uniform lottery. The principal values probability
  on the unique labeled-correct response; 48 reward models approve according
  to expected score. This interpretation treats archived scores as cardinal
  utilities extended linearly to lotteries, with review before sampling.

Ties count as approvals throughout. Exact soundness is evaluated over every
proposal in the stated domain, not only over the recorded endpoints. The
full-domain calculations enumerate the exact two-dimensional hyperplane
arrangement after normalization; reviewer coalitions and response lotteries
are not sampled.

The prompt-specific oracle analyses know the benchmark principal separately
on every instance. They measure what the released panel can certify and are
not deployable rules for unlabeled prompts. The learned weighted analyses use
a fixed 60/20/20 category-stratified split. Weights and regularization are
selected without test labels. Figures that trace the best threshold at a
specified test-set soundness rate are descriptive held-out frontiers, not
operating points selected for deployment without labels.

The topic-conditioned comparison assumes that the standard RewardBench 2
subset is observed before authorization. It fits weights, hyperparameters, and
the selected singleton separately within Factuality, Focus, Math, Precise
Instruction Following, and Safety, using only the corresponding training and
validation prompts. Each family still sweeps one scalar threshold shared by
all topics; its candidate values are pooled only after routing validation
prompts through their topic-specific parameters. It does not select a separate
threshold for each topic. The global and topic-conditioned cardinal rules use
exactly the same scores centered at the uniform fallback and normalized to
unit $\ell_2$ norm within reviewer and prompt, nonnegative softmax objective, and
regularization grid; only the training and validation scope differs by topic.

The separate topic-threshold budget experiment allows five thresholds while
keeping four score rules fixed: global cardinal weights, topic-routed cardinal
weights, the pooled training-selected global singleton, and the five
within-topic training-selected singletons. For every pooled validation
exact-soundness failure budget from 0 through 353, an exact multiple-choice
dynamic program chooses one threshold per topic to maximize pooled validation
$0.5$-completeness. The matched baseline chooses one shared threshold under
the same validation objective and budget. The global-cardinal and
global-singleton controls reuse exactly the same global weight vector or
reviewer in every topic; only their five thresholds vary, isolating calibration
from topic-specific score learning or reviewer routing.

Learned cardinal committees in this calibration audit are rounded to
eight-decimal integer units, renormalized, and aggregated in fixed reviewer
archive order with accurate summation. Weighted-binary committees use the
same projection for every shrinkage candidate in both the global and
topic-routed analyses. This is necessary for binary scores because many
responses have exactly the same approval pattern: mathematically equal subset
sums must remain one threshold checkpoint instead of being spuriously split
by last-bit optimizer or BLAS variation. The continuous learned-cardinal and
ablated-cardinal main-frontier fits remain unrounded because their validation
scores do not have this repeated-subset-sum degeneracy; supported stacks keep
the same discrete threshold path and paper checkpoints, while `verify.py`
tolerates only sub-trillionth numerical movement of the threshold values.

The projected source weights, exact integer units, totals, reviewer-order
fingerprints, and unit-vector fingerprints are serialized in
`data/cardinal_results.json` and the topic-threshold result. `verify.py` pins
all global and topic unit-vector fingerprints, so an optimizer result that
crosses a quantization boundary fails loudly rather than silently changing a
threshold path. This projection absorbs bounded numerical variation only
while fits remain inside the recorded quantization cells; it is not a claim
that arbitrary optimizer changes are platform-independent.

Each canonical topic-local grid contains every tolerance-adjusted threshold at
which either Boolean guarantee can change, the adjacent floating-point values,
one representative of every open interval, reject-all, and accept-all. The raw
grid searched for each topic is the union of that local grid and the complete
pooled shared grid. Consequently every shared threshold is literally feasible
as the constant five-threshold vector. The dynamic program losslessly
Pareto-compresses these raw grids by validation failures and completeness; if
an added pooled threshold duplicates a local validation behavior, compression
prefers the canonical topic-local representative. Thus every shared setting
has a retained validation witness that is no more costly and no less complete.
Remaining ties prefer fewer validation failures and then lexicographically
higher thresholds in fixed topic order. Singleton
reviewer identities remain training-selected, while all four families use
validation-derived calibration grids. Every validation-selected scalar or
vector is staged and hashed before held-out metrics are attached, and the hash
is checked afterward. Thus the test path is connected in validation-budget
order and is neither optimized nor filtered using test labels.
`data/topic_threshold_budget_results.json` retains every $B$-indexed
validation choice, its five topic thresholds (including repeated values for a
shared rule), the validation budget actually used, and pooled and per-topic
held-out counts.

## Code map

`run.py` is the only analysis entry point. It calls these modules in order:

| File | Role |
| --- | --- |
| `depth_analysis.py` | Exact RewardBench 2 rejection depth and beneficial-objection maximum. |
| `approximate_analysis.py` | Exact bounded-domain soundness and completeness errors. |
| `alignment_analysis.py` | Nearest-reviewer angles and individual-alignment diagnostics. |
| `oracle_analysis.py` | Prompt-specific unanimity, largest certified count rule, and maximal sound monotone rule. |
| `binary_analysis.py` | Count-rule and category figures. |
| `cardinal_analysis.py` | Global and topic-conditioned weighted-rule comparisons and train-selected singleton generalization. |
| `geometry.py` | Shared cone, split, and hyperplane-arrangement routines. |
| `verify.py` | Fast post-run audit of inputs, headline values, and required outputs. |

The exact analyses contain redundant checks. In particular, the conic
coverage LP, angular-depth calculation, bounded-simplex enumeration, and
worst-case lottery LP must agree where their conclusions overlap. The run
stops with an error if any check fails.

## Inputs and provenance

The two input archives are immutable and checked by SHA-256 before any array
with object metadata is loaded:

| File | SHA-256 | Source |
| --- | --- | --- |
| `data/rewardbench2_scores.npz` | `2b6f6cfa2c6cace3d85e1bc96b5a8c61390b3b91a2bb73f67bd6e89849ed0d7c` | RewardBench 2 dataset revision `7ff08853b0d5686e79b13fda8677024f566a104a` and results revision `b19c7033e964187d12e74a43a07f2d727a3d37e5`. |
| `data/strongreject_scores.npz` | `e462848048763dc24640535baa2ce2bf9ed9834535faf323c2b73b8ea5396d48` | Official StrongREJECT processed `labelbox.csv` and `labelbox_evals.csv` archives. |

The pinned archives are included so the experiment does not depend on changing
remote data or on external download scripts.

## Outputs

The runner rewrites the following derived data:

- `data/rewardbench2_depth.npz`
- `data/approximate_results.npz`
- `data/approximate_results.json`
- `data/alignment_results.json`
- `data/oracle_results.json`
- `data/cardinal_results.json`
- `data/topic_threshold_budget_results.json`

The canonical figure outputs include the analysis figures and standalone
verified diagnostics:

- `plots/binary_overall.pdf`
- `plots/approximate_implementation.pdf`
- `plots/rewardbench_categories.pdf`
- `plots/rewardbench_completeness_categories.pdf`
- `plots/cardinal_frontier.pdf`
- `plots/topic_conditioned_comparison.pdf`
- `plots/topic_conditioned_comparison.png`
- `plots/topic_threshold_budget.pdf`
- `plots/topic_threshold_budget.png`
- `plots/singleton_train_test_generalization.pdf`
- `plots/singleton_train_test_generalization.png`
- `plots/ranked_singleton_approximate.pdf`
- `plots/ranked_singleton_approximate.png`
- `plots/cardinal_margin_sensitivity.pdf`
- `plots/rewardbench_topic_distributions.pdf`

The runner also regenerates the standalone pure-response comparison in both
vector and raster form: `plots/cardinal_pure_response.pdf` and
`plots/cardinal_pure_response.png`.

For direct comparison, `plots/cardinal_pure_response_full_and_zoom.pdf` and
`plots/cardinal_pure_response_full_and_zoom.png` place the complete
pure-response frontier beside a high-soundness enlargement of the same curves.
The detail panel emphasizes the 99% held-out soundness requirement and marks
the descriptive checkpoints for the learned cardinal panel and validation-
selected singleton; threshold candidates remain fixed by validation data.

`rewardbench_topic_distributions.pdf` separates individual reviewer gains
from the equal-weight cardinal aggregate and reports the full per-topic
distributions of reviewer--principal angles and promptwise closest-reviewer
angles. A 300-dpi PNG is regenerated alongside the vector PDF.

`topic_conditioned_comparison.pdf` compares each global learned family with a
topic-conditioned counterpart. The solid curves use topic-specific binary or
cardinal weights, or a topic-specific validation-selected singleton; dashed
curves use the corresponding global parameters. Topic is an observed routing
variable, but each family uses one threshold across all five topics. Candidate
thresholds are pooled routed validation scores, and the curves report pooled
held-out rates over all 352 test prompts. Panel (a) shows exact soundness versus
$0.5$-completeness. Panel (b) requires at least 95% approximate-soundness
coverage at each $\epsilon_{\rm harm}$ and reports the greatest
$0.5$-completeness across the validation-defined shared thresholds. These are
descriptive held-out frontiers, not test-selected deployment rules. The
topic-specific train/validation/test counts are Factuality 285/95/95, Focus
297/99/99, Math 110/37/36, Precise Instruction Following 96/32/32, and Safety
270/90/90; comparisons involving the two smallest strata should be interpreted
cautiously.

`topic_threshold_budget.pdf` compares validation-budget paths with either one
shared threshold or one threshold per observed topic. Candidate grids are
derived from validation and are behaviorally exhaustive for exact soundness
and $0.5$-completeness, including robust mixture-score transitions rather than
only pure-response endpoints. Panel (a) shows the global and topic-routed
cardinal score rules. Panel (b) shows both the pooled training-selected global
singleton score rule and the within-topic training-selected routed singleton
score rule, each under shared- and topic-threshold calibration. At the primary
validation budget of 17 failures (at least 336/353, or 95.2%, exact soundness),
the learned topic-cardinal threshold vector freezes to a held-out result of
337/352 (95.7%) sound and
132/352 (37.5%) $0.5$-complete, versus 334/352 (94.9%) and 84/352 (23.9%) for
its matched shared-threshold baseline. The train-selected topic singletons
reach 336/352 (95.5%) and 147/352 (41.8%) with five thresholds, versus 333/352
(94.6%) and 47/352 (13.4%) with one shared threshold. A control using the
global cardinal weights with five topic thresholds reaches 334/352 (94.9%) and
144/352 (40.9%). The pooled train-selected global singleton reaches 327/352
(92.9%) sound and 151/352 (42.9%) complete with five thresholds, versus
333/352 (94.6%) and 44/352 (12.5%) with one shared threshold. Thus
topic-specific calibration drives the gain over a shared threshold, but the
global-singleton vector also exposes a larger validation-to-test soundness
shortfall. With the same five-threshold calibration, global cardinal weights
authorize 12 more prompts at the cost of three additional held-out soundness
failures; neither held-out point dominates, but separate topic weight fitting
does not explain the completeness gain. The routed singleton instead gives up
four completions relative to the global reviewer while recovering nine sound
test prompts, so reviewer routing and calibration have distinct effects.

This pooled objective is deliberately aggregate: it may spend the validation
failure budget unevenly and reject all proposals in low-yield topics. The
five-threshold rule also has more calibration capacity, and the same validation
split already selects cardinal regularization, so the path should be treated as
an exploratory held-out comparison rather than a finite-sample guarantee.

Under a requirement of at least 95% pooled exact test soundness, the
topic-conditioned cardinal rule is $0.5$-complete on 73/352 prompts (20.7%),
versus 101/352 (28.7%) for the global cardinal rule. The topic-conditioned and
global singletons reach
28/352 (8.0%) and 27/352 (7.7%), respectively, while both weighted-binary
families require reject-all. At 95% pooled approximate-soundness coverage and
$\epsilon_{\rm harm}=0.20$, topic-conditioned and global weighted binary reach
11/352 (3.1%) and 12/352 (3.4%). The topic-conditioned cardinal and singleton
remain at 20.7% and 8.0% at every displayed nonvacuous checkpoint.

The topicwise binary hyperparameter selection is sensitive to the small
validation strata. Because every topic has fewer than 100 validation prompts,
the nominal 99% within-topic validation-soundness target admits no failures and
is effectively 100%. Safe validation completeness is zero for Factuality,
Math, and Precise Instruction Following, so the locally selected binary
shrinkage should be treated as unstable and the resulting comparison as
exploratory. The figure's 95% coverage requirement is pooled across all 352
test prompts and does not imply 95% coverage within each topic.

`singleton_train_test_generalization.pdf` isolates singleton reviewer selection
from threshold evaluation. The global reviewer is selected exclusively by
pooled training choice accuracy; the topic policy selects one reviewer by
training choice accuracy within each topic. Those identities are then frozen.
Each policy sweeps one scalar threshold over its pooled training scores, and
exactly the same train-defined candidate values are evaluated in sample and on
the held-out test prompts. The threshold sweep is evaluative and does not
select reviewer identity or a deployment threshold. Dashed curves report the
in-sample training audit and solid curves the held-out test audit.

Choice accuracy is 89.0% in train and 88.9% in test for the global singleton,
and 89.2% on both splits for the topic-routed singleton. Four topics choose the
same reviewer as the pooled global selection; only Safety chooses another.
At the fixed reference threshold $\tau=0.45$, topic routing gives
(exact soundness, $0.5$-completeness) of (88.09%, 47.26%) in train and
(89.49%, 45.45%) in test; the global values are (87.90%, 45.09%) and
(89.20%, 44.32%). At $\tau=0.55$, the topic values are (93.19%, 12.19%) and
(94.89%, 11.36%), while the global values are (93.19%, 8.79%) and
(94.89%, 10.23%). These matched fixed-threshold checks show no conventional
optimistic train--test gap. The thresholds are not chosen from either frontier;
a deployment threshold must still be fixed without test labels.

`cardinal_frontier.pdf` overlays the original global binary and cardinal
families with the topic-conditioned cardinal rule and both singleton policies
whose reviewer identities are selected exclusively on training data. The
global singleton uses the pooled training-best reviewer; the topic-routed
singleton uses the training-best reviewer within each observed topic. Their
threshold candidates are the pooled training scores used by the singleton
generalization audit, whereas the learned cardinal families retain their
validation-defined threshold candidates. Under a requirement of at least 95%
exact held-out soundness, the global and topic-routed train-selected singletons are
$0.5$-complete on 30/352 (8.5%) and 34/352 (9.7%) prompts; the global and
topic-conditioned cardinal rules reach 101/352 (28.7%) and 73/352 (20.7%).

The figure also contains two validation-ranked native-binary diagnostics. The
first removes the top-ranked reviewers one at a time and uses the next
remaining reviewer alone. The second adds reviewers in the same validation
order and requires unanimous approval from each ranked prefix. These are fixed
binary-approval trajectories, not learned threshold frontiers. The
high-soundness portion of the prefix trajectory is overlaid in panel (a), using
the same held-out soundness and completeness coordinates; panel (b) shows the
full path. Its connected points vary the discrete prefix size $m$, not a score
threshold $\tau$, and do not constitute a Pareto frontier.

`cardinal_pure_response.pdf` gives the matched endpoint-only analogue for the
same six global rule families. Its horizontal coordinate is the fraction of
the 352 held-out prompts on which all three labeled-incorrect pure responses
are rejected; its vertical coordinate is the fraction on which the labeled-
correct pure response is authorized. The learned weights, selected reviewer,
and threshold candidate grids are unchanged from `cardinal_frontier.pdf`:
each curve evaluates only thresholds induced by the 353 validation prompts,
plus reject-all and accept-all. The plotted held-out frontier and its markers
are therefore descriptive evaluations of validation-defined rule families,
not test-selected deployment thresholds. The prompt-specific reconstruction
point is again an oracle rather than a deployable rule.

The descriptive checkpoint counts are:

| Rule family | Sound / complete at $\geq99\%$ pure soundness | Sound / complete at $\geq95\%$ pure soundness |
| --- | ---: | ---: |
| Binary count | 349 / 13 | 335 / 62 |
| Learned weighted binary | 349 / 31 | 335 / 171 |
| Equal-weight cardinal | 349 / 124 | 336 / 203 |
| Learned nonnegative cardinal | 349 / 213 | 336 / 282 |
| Learned cardinal without selected reviewer | 349 / 202 | 335 / 263 |
| Validation-selected singleton | 350 / 164 | 335 / 277 |

All denominators are 352. In particular, the learned cardinal rule leads the
singleton by 49 prompts (13.9 percentage points) at the 99% checkpoint, but
only by 5 prompts (1.4 points) at the 95% checkpoint. There is no separate
approximate pure-endpoint panel: each incorrect endpoint has principal loss
exactly 0.25 and the correct endpoint has gain exactly 0.75, so relaxing
$\epsilon_{\rm harm}$ below 0.25 or varying $\epsilon_{\rm good}$ anywhere
from 0 through 0.75 leaves the corresponding pure metric unchanged. At
$\epsilon_{\rm harm}=0.25$, pure-response soundness becomes vacuous.

`ranked_singleton_approximate.pdf` reports the corresponding approximate
full-domain diagnostics for every individual reviewer in that same fixed
validation order. Panel (a) is a heatmap for rank 1: each
$(\epsilon_{\rm harm},\epsilon_{\rm good})$ cell gives the held-out fraction
jointly satisfying both guarantees. Panels (b)--(d) show, for all 48 ranks,
the minimum $\epsilon_{\rm good}$ at each fixed $\epsilon_{\rm harm}$ needed
to jointly certify 10%, 20%, or 30% of test prompts. The ordering uses
validation choice accuracy without test labels; the plotted lines are
descriptive held-out diagnostics of fixed native-binary rules, not selected
operating rules. Rank 1 jointly certifies 8.8%, 17.9%, 34.1%, and 94.3% of
prompts at $\epsilon_{\rm good}=0.5$ when $\epsilon_{\rm harm}$ is 0.05,
0.10, 0.20, and 0.25, respectively. Rank is broadly informative but the test
trade-offs need not be perfectly monotone in rank. The
$\epsilon_{\rm harm}=0.25$ boundary makes soundness vacuous because it permits
the largest possible loss from the uniform fallback. A 300-dpi PNG is
regenerated alongside the vector PDF.

The three panels of the main paper's Figure 1 use the StrongREJECT and
RewardBench 2 panels from `binary_overall.pdf` together with the category
panel from `rewardbench_categories.pdf`; all three are regenerated by the
canonical runner.

PNG versions and separate benchmark panels are also regenerated for visual
inspection. The bundle omits older exploratory outputs that are not produced
by the current runner.

## Expected checks

A successful `verify.py` run confirms, among other invariants:

- 1,590 of 1,763 RewardBench 2 prompts are covered and the maximum depth is 23;
- all 190 harmful StrongREJECT responses have at least two objections;
- the independent exact constructions agree on depth and completeness;
- the paper's oracle-rule table is reproduced exactly;
- the held-out split is 1,058/353/352 and the 95% soundness-tolerance sweep
  uses the exact empirical breakpoints;
- the pure-response frontier reuses the exact validation-defined threshold
  grids, has nested soundness/completeness paths, and reproduces the 99% and
  95% held-out checkpoint counts above;
- the five topic-conditioned fits use only their routed training and validation
  strata, and every family sweeps one shared threshold drawn from pooled routed
  validation scores rather than one independently chosen threshold per topic;
- the topic-threshold calibration paths optimize every pooled validation
  failure budget exactly for all four frozen score rules, embed the shared
  grid literally in every raw topic grid, freeze and hash each validation
  selection before held-out evaluation, and reconstruct all held-out counts
  without using test labels for selection or curve filtering;
- the singleton generalization audit selects reviewer identities only by
  training choice accuracy, freezes them, and evaluates the same train-defined
  shared-threshold candidates on training and held-out test prompts;
- all 48 native single-reviewer rules have zero exact full-domain soundness on
  the test split, while ranked-prefix unanimity rises monotonically to 315/352;
- the ranked-prefix path first enters the 75--100% overlay window at $m=42$,
  with soundness 271/352 and completeness 33/352;
- the full 48-reviewer ranked prefix exactly matches the zero-objection binary
  count rule;
- every required analysis figure exists and has a valid PDF or PNG header.
