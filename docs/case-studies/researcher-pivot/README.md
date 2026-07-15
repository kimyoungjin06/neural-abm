# When Do Researcher Pivots Become Productive?

A two-study replicated case study of field-pivot decisions under different
scientific environments. Study 1 (torch-free `neural_abm.api_lite`) compares
environments under a fixed decision rule. Study 2 (torch-backed
`neural_abm.api`) replaces the fixed rule with learning agents and asks a
question a fixed-rule ABM cannot ask. This note is the reference end-to-end
research example for the package: research question, stylized mechanism,
seed-paired replication, sensitivity sweeps, figures, and limitations.

**Status**: stylized-mechanism study. Parameters are theory-motivated, not
calibrated to empirical data. Read the [Limitations](#limitations) before
citing any number outside this repository.

## TL;DR

**Study 1** (fixed decision rule, 100 seed-paired replicates of a
120-researcher population):

1. **Seed grants and hype deliver the same productive gain at very different
   cost.** Interdisciplinary seed grants raise the productive pivot rate by
   +0.190 [95% CI 0.112, 0.263] over baseline; hot-field hype delivers a
   statistically indistinguishable +0.193 [0.125, 0.263] — but does it by
   converting 2.4x more researchers (56% vs 23%) and failing ten times as
   many of them (36% vs 4% of the population).
2. **Hype changes who pivots, not just how many.** Hype recruits pivoters
   from the resource-insecure end of the population, so only 36% of
   hype-driven pivots are productive versus 84% under seed grants.
3. **Peer influence is a double-edged amplifier.** Stronger social mixing
   suppresses both spontaneous pivots and the seed-grant effect (productive
   rate 0.27 → 0.12 as social alpha goes 0 → 0.8) while consolidating
   hype-driven mass pivoting. This pattern was not designed in; it emerges
   from readiness homogenization around each scenario's population mean.

**Study 2** (learning agents, 30 seed-paired replicates, three arms):

4. **The direction of social learning matters more than learning itself.**
   Agents that imitate observed pivot success trigger an information
   cascade: because only pivoters reveal outcomes and early pivoters mostly
   succeed, one-sided positive evidence floods every environment with failed
   pivots (failed rate +0.30 over the fixed rule even at baseline).
5. **Negativity-biased learners acquire targeted hype immunity.** Agents
   that update only on observed failures cut the failed pivot rate under
   hype by −0.137 [−0.200, −0.087] while leaving the baseline world exactly
   unchanged — and the mechanism is directly auditable: their learned
   attention weight falls from 1.00 to 0.81 under hype and moves nowhere
   else.

![Productive pivot rate distributions per scenario](figures/fig1_productive_pivot_distributions.png)

## 1. Research Question

Should research institutions that want more interdisciplinary mobility
invest in structural support (small grants, bridge ties, reduced reputation
penalties), or is visible hot-field attention enough? Concretely: **under
which environments do researcher pivots become productive rather than merely
frequent?**

Hypotheses:

- **H1 (support)**: seed grants raise the productive pivot rate over
  baseline by more than 5 percentage points.
- **H2 (hype paradox)**: hype raises the pivot rate, but the productive
  share of pivots drops, because attention reaches the desire to pivot
  without reaching the capacity to pivot well.
- **H3 (interaction)**: support layered on hype recovers part of the
  productive share lost to hype.

## 2. Model

The model is a bounded-scalar Neural-ABM workflow: each step applies a local
adaptation to every researcher's `pivot_readiness`, mixes readiness through
typed peer exchange on a stage-assortative network, and commits the mixed
value through a domain-owned threshold transition. All dynamics run through
`neural_abm.api_lite`; the domain meaning lives in the study script.

### Agents

Each replicate samples 120 researchers. Career stages are drawn with
probabilities early 0.35, mid 0.30, senior 0.20, bridge 0.15, and structural
attributes (`skill_distance`, `resource_security`, `network_support`,
`reputation_risk`, `openness`) come from stage-conditioned Beta
distributions (see `STAGE_ATTRIBUTES` in the study script).

### Push-pull decision structure

The desire to pivot and the capacity to pivot well are deliberately
separated, following the push-pull framing from migration theory applied to
topic mobility:

- **Pivot pressure** (push): field opportunity, openness, resource
  *insecurity*, funding and attention signals, minus skill distance and
  reputation risk.
- **Productive fit** (pull/capacity): field opportunity, network support,
  resource security, minus skill distance and reputation risk. Attention
  never enters fit.

Hype susceptibility is heterogeneous: attention and peer-success signals are
scaled per researcher by `0.6 * (1 - resource_security) + 0.4 * openness`.
This single mechanism produces H2: hype recruits exactly the researchers
whose structural fit is weakest.

### Scenarios

| Scenario | Mechanism | Key signals |
|---|---|---|
| `baseline` | No intervention | — |
| `interdisciplinary_seed_grants` | Program-level support raises pressure *and* fit | funding 0.25, resources +0.10, bridge ties +0.10 and extra network links, reputation −0.05 |
| `hot_field_hype` | Attention raises pressure only, hardest on the insecure | attention 0.55, peer-success 0.35, small real field signal 0.06, reputation +0.10 |
| `hype_with_support` | Both signal families at once | sum of the two rows above |

### Transition

A researcher pivots when committed readiness ≥ 0.34 and the pivot is
productive when their structural fit ≥ 0.40. Outcomes are population rates
at the final step.

## 3. Experimental Design

- **Replication**: 100 replicates per scenario via
  `run_replicated_bounded_scalar_scenarios` with `base_seed=20260715`.
  Replicate *r* uses the same seed in every scenario (common random
  numbers), so scenario deltas are paired and the 95% CI is the percentile
  interval of the 100 paired deltas.
- **Primary outcome**: `productive_pivot_rate`; success criterion: mean
  paired delta > 0.05.
- **Dynamics**: 8 steps, local alpha 0.45 with N(0, 0.02) decision noise,
  social alpha 0.40, output-similarity peer rule with threshold 0.68.
- **Sensitivity**: grant-intensity sweep (0x–2x default signals) and social
  alpha sweep (0.0–0.8), 40 replicates per point.

## 4. Results

### R1 — Support works, and a placebo dose does nothing (H1 supported)

Seed grants raise the productive pivot rate by +0.190 [0.112, 0.263]; every
one of the 100 paired replicates clears the +0.05 criterion. The
dose-response is monotone, and at zero grant intensity the effect collapses
to +0.004 [−0.017, 0.017] — the placebo behaves like a placebo, which is a
basic validity check on the comparison machinery.

![Sensitivity sweeps](figures/fig3_sensitivity.png)

### R2 — Hype buys pivots, support buys productive pivots (H2 supported)

Hype matches the seed-grant productive delta (+0.193 vs +0.190) but through
an entirely different composition: 56% of the population pivots (vs 23%
under grants) and 36% of the population ends in a failed pivot (vs 4%). The
productive share of pivots is 36% under hype against 84% under grants. If an
institution measures success by pivot counts, hype looks strictly better; if
it measures productive transitions per researcher disrupted, hype is an
order of magnitude more wasteful.

![Pivot composition per scenario](figures/fig2_pivot_composition.png)

### R3 — Support partially rescues hype (H3 partially supported)

Combining both signal families yields the largest productive rate (+0.476
[0.392, 0.559]) and lifts the productive share from 36% to 51% — but the
failed pivot rate stays at 47% of the population, because in this regime
pivoting becomes near-universal (95%). Support scales productivity; it does
not cancel hype's waste. We flag this cell as the model's saturation regime
and interpret it qualitatively only.

### R4 — Peer influence suppresses interventions and consolidates hype

The social alpha sweep was run as a robustness check and produced the most
interesting emergent pattern. Stronger peer mixing homogenizes readiness
toward each scenario's population mean, which (a) eliminates spontaneous
baseline pivots (0.082 → 0.000), (b) erodes the seed-grant effect (0.273 →
0.123), and (c) mildly consolidates mass pivoting under hype+support. In
this model, strong conformity pressure makes targeted interventions weaker
and bandwagons stronger.

![Readiness trajectories](figures/fig4_readiness_trajectories.png)

The trajectory figure shows the mechanism compactly: mean readiness under
seed grants is *lower* than under hype, yet grants produce equally many and
far better pivots — who crosses the threshold matters more than how much
the population average moves.

## 5. Interpretation

Within the model's assumptions, the policy-relevant readings are:

- Counting pivots overstates hype and understates support. Composition
  metrics (productive share, failed-pivot rate) invert the ranking.
- The hype paradox needs no irrationality: it follows from heterogeneous
  susceptibility (insecure researchers respond most to attention) plus the
  fact that attention does not build capacity.
- Environments with strong conformity dynamics may need stronger targeted
  support to achieve the same productive mobility.

## 6. Study 2 — Can Learning Agents Self-Correct the Hype Paradox?

Study 1's decision rule is fixed: no researcher ever revises how they weigh
attention against resources, no matter what they observe. Study 2 removes
that assumption. It runs on the torch-backed `neural_abm.api` lifecycle
(`NABMUnit`, typed bounded-scalar channel, domain-owned commit adapter) and
asks: **when researchers learn from observed pivot outcomes, does the hype
paradox self-correct — or does hype outpace learning?**

### Design

Three arms share one sigmoid decision policy over the same nine observable
features; the only difference between arms is the learning rule, so every
divergence is attributable to learning:

| Arm | Update rule |
|---|---|
| `frozen` | Never updates (Study 1's push-pull rule in policy form) — the classical-ABM control. |
| `imitative` | Vicarious gradient learning from every observed neighbor pivot outcome (BCE toward predicting productive pivots). |
| `cautionary` | The same update, applied only to observed *failures* — negativity-biased social learning. |

Pivots are absorbing events with publicly visible outcomes; pivoted
researchers keep broadcasting readiness, so hype contagion and outcome
learning compete on the same stage-assortative network. A weak L2 anchor to
the prior rule models conservative belief updating; a 2-step burn-in keeps
initial-condition transients from being absorbed as pivots. 30 seed-paired
replicates, 14 steps, three scenarios (baseline, seed grants, hype).

The policy class is deliberately minimal (ten named parameters per agent):
each agent's training signal is single-digit observed outcomes, so larger
models would fit noise; named weights are what make the learned-parameter
trajectory usable as audit evidence; and minimal capacity keeps the
fixed-versus-learning comparison free of a capacity confound. The
system-level nonlinearity (thresholds, absorbing events, contagion) is where
both headline results come from. The lifecycle accepts arbitrary torch
modules, so this is an escalation ladder, not a cap.

Because pivots are absorbing (anyone who ever crosses the threshold stays
pivoted), the frozen arm's cumulative rates sit above Study 1's final-step
state rates; comparisons are within-design, against the frozen control.

### R5 — Imitating success triggers an information cascade

Observed outcomes are survivorship-biased: only pivoters reveal outcomes,
and early pivoters are the best-positioned, so they mostly succeed. For an
imitative learner this one-sided positive evidence raises the propensity to
pivot across the board, which produces more pivots, more success stories,
and a self-reinforcing cascade — the textbook information-cascade mechanism
emerging from local gradient updates. Failed pivot rates rise by +0.30
[0.07, 0.44] at baseline and +0.29 [0.17, 0.38] under seed grants; the
productive share of pivots collapses (0.84 → 0.42 at baseline, 0.71 → 0.54
under grants). Supportive environments are hit hardest, because support
manufactures exactly the early success stories that feed the cascade.

![Learning direction decides the failure curve](figures/fig5_learning_failed_trajectories.png)

### R6 — Cautionary learning yields targeted hype immunity

Negativity-biased learners cut the failed pivot rate under hype by −0.137
[−0.200, −0.087] relative to the frozen rule, while the baseline world is
left exactly unchanged (delta −0.000 — there are no failures to learn from).
The mechanism is directly auditable in the learned parameters: the mean
attention weight drops from 1.00 to 0.81 under hype within three steps of
the first failures and moves nowhere else. Hype immunity is acquired
exactly where, and only where, hype fails people.

![Cautionary learners discount attention only where it fails](figures/fig6_learning_attention_weights.png)

### R7 — The price of caution

Under seed grants, cautionary learners also pivot less (productive rate
−0.073 [−0.127, −0.031]): the early failures they observe make them discount
signals that, in a supportive environment, are genuinely load-bearing.
Negativity bias buys hype protection at the cost of under-reacting to real
support — a trade-off invisible to any fixed-rule model.

### Why this needed learning agents

Study 1 could have been built in any ABM framework; Study 2 could not have
been *asked* in one. Every Study 2 finding is a statement about how decision
rules change endogenously — cascades from imitation, immunity from
cautionary updates, the attention-weight trajectory as audit evidence. In a
fixed-rule ABM the modeler would have to hand-code the correction they are
trying to discover. Here the classical control and the learning variants run
in the same lifecycle, on the same seeds, with the same audit logging, and
differ in nothing but the learning rule.

## 7. Limitations

- **Stylized, not calibrated.** All parameters are theory-motivated
  constants; no empirical bibliometric data was fit. Effect sizes are
  model-internal quantities, not field estimates.
- **Thresholds are modeling choices.** The pivot (0.34) and productivity
  (0.40) thresholds were chosen so the baseline pivot rate is realistically
  low (~1%); qualitative orderings were checked under the sweeps, but a
  fuller threshold sensitivity analysis is future work.
- **The hype+support cell saturates** (95% pivot rate), so its magnitudes
  are less meaningful than its ordering.
- **`productive_fit` is static per researcher**; a pivot does not feed back
  into skills or resources, and there is no time-to-recovery dynamic.
- Baseline pivots are near zero, so ratio-style comparisons against baseline
  are unstable; all claims use absolute rate deltas.
- **Study 2's learning modes are assumptions, not discoveries.** Imitative
  and negativity-biased social learning are both empirically documented, but
  the study compares postulated update rules; it does not learn which rule
  people use. The learning rate and prior-anchor strength were fixed, not
  swept, and the policy is linear in the observed features.
- **Study 2 outcomes are immediately and perfectly observable.** Real pivot
  outcomes reveal slowly and noisily; delayed or noisy outcome disclosure
  would weaken learning and is the most interesting extension.

## 8. Reproduction

From a repository clone:

```bash
# Study 1: full study + sweeps (roughly 5-10 minutes, torch-free)
uv run --no-dev python examples/research_pivot_study.py --sweeps \
  --output docs/case-studies/researcher-pivot/data/study_results.json

# Study 2: learning agents (roughly 2-4 minutes, requires torch)
uv run python examples/research_pivot_learning_study.py \
  --output docs/case-studies/researcher-pivot/data/learning_study_results.json

# case-study figures + selected paper/figures copies (requires dev/plot deps)
uv run python scripts/plot_research_pivot_study.py

# fast smokes
uv run --no-dev python examples/research_pivot_study.py --quick
uv run python examples/research_pivot_learning_study.py --quick
```

Determinism: rerunning with the same `--base-seed` (default 20260715)
reproduces every number in this note exactly; the JSON artifact in `data/`
is the run the figures were rendered from.

## 9. Framework Surface Used

Study 1 exercises the researcher-facing torch-free path end to end:

- `ScenarioDefinition`, `BoundedScalarScenarioSpec` — question, scenarios,
  success criterion.
- `ReplicationSpec`, `ScenarioReplicateContext`,
  `run_replicated_bounded_scalar_scenarios` — seed-paired replication with
  outcome distributions and paired comparisons.
- The bounded-scalar workflow underneath contributes typed peer exchange,
  commit reports, and aggregate/micro audit rows; the first-replicate micro
  audit in the JSON artifact records per-researcher evidence
  (stage, fit, pressure, peer counts) for every scenario.

Study 2 exercises the stable torch-backed lifecycle:

- `NABMUnit`, `NABMStep`, `SocialBlock`, typed `SocialChannel`
  (bounded-scalar) — the local-update / typed-exchange / commit separation
  with learning agents implementing the NABM agent protocol.
- A domain-owned commit adapter absorbs pivot events and publishes outcomes,
  and `select_bounded_scalar_output_peers` applies the same peer rule as
  Study 1.
- Known gap surfaced by this study: the seed-paired replication helpers
  currently live in `scenario_lite` only; Study 2 reimplements the replicate
  loop by hand. Generalizing replication over the torch lifecycle is queued
  framework work.
