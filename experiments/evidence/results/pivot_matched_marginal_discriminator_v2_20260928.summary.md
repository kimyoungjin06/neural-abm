# Pivot matched-marginal discriminator v2 result

Date: 2026-09-28

Status: `SYNTHETIC_POLICY_LAYER_DISCRIMINATOR_REPRODUCED`

## Result

All three registered comparisons pass on the frozen generator. A post-hoc
check against the linear worlds narrows what two of them mean:

- **Feature-conditional need** is robust. Explicit linear beats scalar Delta by
  about `0.10` Brier in all 16 replicates of both linear orientations.
- **Nonlinear representation need** is real, but the registered Brier margin
  overstates it. The residual MLP beats explicit linear by almost as much on
  the linear worlds, where no nonlinearity exists, as on the interaction
  worlds. Ranking accuracy (AUC) isolates the nonlinearity cleanly.
- **The neural-specific advantage** passes as registered, but it does not come
  from the interaction. The MLP's advantage over the histogram tree is larger on
  the linear worlds than on the interaction worlds. It is a general advantage in
  fitting smooth boundaries over this frozen tree, not evidence of neural
  necessity.

The run used v2 at `4c5920c` and produced 320 rows: five models, four worlds,
and 16 replicates, each with 256 training and 256 disjoint evaluation agents.
Scalar Delta and fixed-rule predictions were byte-identical across orientations.
A second execution reproduced all four artifacts byte for byte.

## Registered decisions

Brier improvement of the model over its reference. A comparison passes when the
mean meets the threshold and the improvement is positive in at least 12 of 16
replicates, in both worlds.

| Comparison | World | Mean | Min | Positive | Threshold |
|---|---|---:|---:|---:|---:|
| Linear over scalar | linear aligned | 0.106 | 0.099 | 16/16 | 0.02 |
| Linear over scalar | linear reversed | 0.103 | 0.096 | 16/16 | 0.02 |
| MLP over linear | interaction aligned | 0.137 | 0.096 | 16/16 | 0.01 |
| MLP over linear | interaction reversed | 0.162 | 0.128 | 16/16 | 0.01 |
| MLP over tree | interaction aligned | 0.020 | -0.013 | 14/16 | 0.005 |
| MLP over tree | interaction reversed | 0.025 | -0.003 | 15/16 | 0.005 |

The linear-world guard also holds. Against explicit linear, the MLP's worst
replicate on the linear worlds is still `0.096` better, not worse.

## Held-out means

| World | Scalar | Linear | Tree | MLP |
|---|---:|---:|---:|---:|
| Brier, linear aligned | 0.251 | 0.145 | 0.072 | 0.024 |
| Brier, linear reversed | 0.284 | 0.181 | 0.072 | 0.024 |
| Brier, interaction aligned | 0.250 | 0.236 | 0.119 | 0.099 |
| Brier, interaction reversed | 0.285 | 0.256 | 0.119 | 0.094 |
| AUC, linear aligned | 0.600 | 0.978 | 0.973 | 0.999 |
| AUC, linear reversed | 0.401 | 0.850 | 0.973 | 0.999 |
| AUC, interaction aligned | 0.612 | 0.643 | 0.918 | 0.941 |
| AUC, interaction reversed | 0.388 | 0.491 | 0.918 | 0.947 |

Scalar Delta shifts calibration but cannot change ranking, so its AUC equals the
fixed rule's in every world. Explicit linear is prior-sensitive: it ranks well
when the frozen prior agrees with the world and worse when it opposes it. The
tree and MLP are nearly orientation-symmetric.

## Post-hoc check

This check was not registered. It reuses the frozen rows and runs nothing new.
It compares each registered gap on the interaction worlds with the same gap on
the linear worlds, where nonlinearity is unnecessary.

| Gap | Orientation | Linear worlds | Interaction worlds | Difference | Positive |
|---|---|---:|---:|---:|---:|
| MLP over linear, Brier | aligned | 0.121 | 0.137 | 0.016 | 13/16 |
| MLP over linear, Brier | reversed | 0.157 | 0.162 | 0.006 | 9/16 |
| MLP over tree, Brier | aligned | 0.048 | 0.020 | -0.028 | 1/16 |
| MLP over tree, Brier | reversed | 0.048 | 0.025 | -0.023 | 2/16 |
| Tree over linear, AUC | aligned | -0.005 | 0.275 | 0.279 | |
| Tree over linear, AUC | reversed | 0.123 | 0.426 | 0.303 | |

Most of the MLP's Brier gain over explicit linear appears on the linear worlds
too. On the aligned linear world, explicit linear ranks almost perfectly (AUC
0.978) but its predictions stay compressed: conditional separation is 0.263,
against 0.888 for the MLP. The linear baseline takes 300 plain gradient steps at
learning rate 0.1 from the frozen prior. The MLP's linear branch trains with
Adam. The registered nonlinearity margin therefore measures baseline fit and
calibration as much as nonlinearity.

Ranking removes that confound. On the aligned linear world, explicit linear
ranks as well as the tree. On the interaction worlds, it falls to 0.64 and 0.49
AUC while the tree reaches 0.92. The nonlinearity requirement survives.

The MLP-over-tree gap reverses the expected pattern. If the MLP's edge came from
representing the interaction, it would be larger on the interaction worlds. It
is smaller there in 15 of 16 aligned replicates and 14 of 16 reversed
replicates. Every label in this generator is a threshold on a smooth continuous
score. That suits a smooth function approximator better than axis-aligned trees
with 100 boosting rounds at learning rate 0.05.

## NABM conclusion

For Pivot, this narrows the policy-layer claim:

- scalar adaptation is insufficient once feature-outcome structure matters;
- a nonlinear conditional policy is needed for the interaction structure;
- neural representation is not shown to be necessary relative to classical
  nonlinear learners; and
- no model is selected.

Full ABM integration is not yet warranted on neural grounds. A successor should
be precommitted as result-informed and use fresh seeds. It should include a
converged linear baseline, a nonsmooth generator control that matches the
tree's inductive bias, and a registered comparison against the linear worlds.
Otherwise, a classical nonlinear policy remains the simpler baseline that a
neural policy must beat.

v2 is not retuned.

## Boundary

This scorecard uses deterministic synthetic cohorts only. It reads zero
empirical rows, performs no empirical fit, changes no generic NABM API, selects
no model, establishes no neural necessity or scientific effect, and authorizes
no package promotion.
