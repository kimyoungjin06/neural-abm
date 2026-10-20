# Information-spread policy ladder: selective labels in the D2 controls

Date: 2026-10-14

Status: `EXPLORATORY_SELECTIVE_LABELS_SUPPORTED_FOR_POOLED_LEARNERS_NOT_FOR_PER_AGENT_MLP`

## Question

The [fit-budget check](info_spread_policy_ladder_d2_fit_budget_check_20261014.summary.md)
left one question open. Before the shift, linear learners stay near the prior
(a credibility weight of 4.1–4.5 against a true 3), even with the Pivot
learner settings.

The candidate cause is selective labels. A learner sees outcomes only for
items that were shared. The hand rule shares credible items, so
low-credibility items are rarely labelled.

## Design

This diagnosis changes only what a learner may see. Everything else,
including the Pivot settings, is fixed.

- Code: `0e950f5`.
- World: the flip control.
- Cells: 10 per linear group and 5 per MLP group, each with 16 replicates.
- Runtime: 3 h 30 min on 36 CPU workers.

| Mode | What the learner sees |
|---|---|
| S | Outcomes of shared items only. This is what every registered arm sees |
| U | A random subset of all decided exposures, shared or declined, with as many rows as the run's shares. Selection is removed and volume is kept |
| F | Outcomes of every decided exposure |

At each refit, a plain logistic regression of the label on the item features
reads what the training rows themselves say. It uses no prior and almost no
penalty.

The S arm reuses the fit-budget check's seeds. It reproduced that check's
L0-full exactly, in 480 of 480 replicate records. The new code therefore
changes no registered behavior.

## Result

The table reports values before the shift, where the truth is +3. "Implied
weight" is the weight the rows themselves give, from the plain logistic
fit. "Post-shift precision gain" is the change against S, with its 95%
interval over cells.

| Group | Mode | Policy weight | Implied weight | Rows with low credibility | Post-shift precision gain |
|---|---|---:|---:|---:|---|
| A / linear | S | 4.17 | 4.13 | 4.8% | |
| A / linear | U | 3.72 | 3.39 | 12.8% | +0.012 [+0.008, +0.015] |
| A / linear | F | 3.73 | 3.40 | 12.8% | +0.011 [+0.008, +0.014] |
| A / MLP | S | 7.75 | 3.92 | 17.4% | |
| A / MLP | U | 3.90 | 3.37 | 20.3% | +0.027 [+0.019, +0.034] |
| A / MLP | F | 3.90 | 3.36 | 20.2% | +0.022 [+0.015, +0.029] |
| B / linear | S | 4.55 | 3.46 | 9.7% | |
| B / linear | U | 4.30 | 3.23 | 21.4% | +0.038 [+0.033, +0.043] |
| B / linear | F | 4.19 | 3.23 | 20.4% | +0.045 [+0.039, +0.050] |
| B / MLP | S | 13.98 | 3.58 | 17.4% | |
| B / MLP | U | 13.43 | 3.27 | 27.4% | +0.057 [+0.050, +0.062] |
| B / MLP | F | 13.40 | 3.31 | 26.7% | +0.054 [+0.049, +0.060] |

## Reading

- **Pooled learners (Regime A): supported.**
  - With shares-only labels, the rows themselves imply a weight above the
    truth: 4.13 for linear, where only 5% of rows have low credibility.
  - The linear policy at 4.17 is therefore not stuck on the prior. It
    follows what its data says.
  - Removing the selection at equal volume lowers the implied weight to 3.4.
    It lowers the policy weight to 3.7 for linear, and from 7.75 to 3.90 for
    the MLP.
  - Adding volume on top (F) changes nothing. Selection, not volume, is the
    cause.
- **Per-agent MLPs (Regime B): not explained.**
  - Their policy weight stays at 13.4 without selection, though their rows
    imply 3.3.
  - Their overconfidence is overfitting, not selection.
- **Per-agent linear (Regime B): partly explained.** The policy weight moves
  from 4.55 to 4.2–4.3, while the rows imply 3.2–3.5.
- **Precision.**
  - After the shift, removing selection raises precision in every group, by
    0.011 to 0.057. That fits better coverage of low-credibility items,
    which become the true ones after the flip.
  - Before the shift the change is neutral for A/linear, +0.011 for A/MLP,
    and slightly negative in Regime B (−0.003 to −0.019).
- **Residual.**
  - Even unselected rows imply 3.2–3.4, not 3.0.
  - Policies sit between that and the prior's 4.
  - Prior anchoring and the small-sample bias of the plain fit are not
    separated.

## The D2 investigation in one place

The adaptive arm lost to the four-donor table in the shift controls. Four
mechanisms were examined.

| Mechanism | Finding |
|---|---|
| (a) The table ensembles four donors | The main cause. At matched evidence, the closed loop alone is indistinguishable |
| (b) Learners do not forget | Real. Forgetting tracks a shift but costs evidence |
| (c) The D1 fit budget | Partly. It held learners back, but a larger budget makes MLPs more overconfident |
| (d) Selective labels | Real for pooled learners. It biases what the labelled rows say. It does not explain per-agent MLP overconfidence |

None of this scores a study cell.
