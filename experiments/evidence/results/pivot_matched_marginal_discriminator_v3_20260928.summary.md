# Pivot matched-marginal discriminator v3 result

Date: 2026-09-28

Status: `SYNTHETIC_POLICY_LAYER_DISCRIMINATOR_REPRODUCED`

Registered ABM route: `PAIRED_CLASSICAL_AND_NEURAL_NONLINEAR_ARMS`

## Result

With the v2 confounds removed, the synthetic policy layer gives a clean
answer:

- **Nonlinear representation is needed in both boundary geometries.** Even
  after subtracting its advantage on the linear worlds, the hidden branch
  improves Brier by 0.09 to 0.13 over the matched linear model, in all 16
  replicates of all four interaction worlds.
- **Neural and classical nonlinear families split by geometry.** The MLP wins
  every smooth-world replicate. The tree wins every axis-world replicate. The
  Brier margins are similar, about 0.045 each way.
- **The MLP's smooth-world edge is now interaction-specific.** Unlike in v2, it
  exceeds the MLP's edge on the linear worlds.
- **Neither family dominates**, so no neural-specific advantage is supported.
  The frozen routing rule sends both nonlinear families into the ABM as paired
  arms.

The run used v3 at `14f37af` with fresh seeds. It produced 480 rows: five
models, six worlds, and 16 replicates, each with 256 training and 256 disjoint
evaluation agents. Scalar Delta and fixed-rule predictions were byte-identical
across orientations. A second execution reproduced all four artifacts byte for
byte.

## Registered decisions

Brier improvement of the model over its reference. DiD subtracts the same gap on
the linear world of the same orientation and replicate.

| Decision | World | Statistic | Mean | Min | Positive | Threshold | Pass |
|---|---|---|---:|---:|---:|---:|---|
| Linear over scalar | linear aligned | raw | 0.185 | 0.170 | 16/16 | 0.02 | yes |
| Linear over scalar | linear reversed | raw | 0.209 | 0.195 | 16/16 | 0.02 | yes |
| MLP over linear | smooth aligned | DiD | 0.134 | 0.088 | 16/16 | 0.01 | yes |
| MLP over linear | smooth reversed | DiD | 0.120 | 0.072 | 16/16 | 0.01 | yes |
| MLP over linear | axis aligned | DiD | 0.094 | 0.028 | 16/16 | 0.01 | yes |
| MLP over linear | axis reversed | DiD | 0.085 | 0.038 | 16/16 | 0.01 | yes |
| MLP over tree | smooth aligned | raw | 0.045 | 0.022 | 16/16 | 0.005 | yes |
| MLP over tree | smooth reversed | raw | 0.042 | 0.012 | 16/16 | 0.005 | yes |
| MLP over tree | axis aligned | raw | -0.047 | -0.075 | 0/16 | 0.005 | no |
| MLP over tree | axis reversed | raw | -0.046 | -0.080 | 0/16 | 0.005 | no |
| MLP over tree | smooth aligned | DiD | 0.021 | -0.014 | 15/16 | 0.005 | yes |
| MLP over tree | smooth reversed | DiD | 0.017 | -0.024 | 14/16 | 0.005 | yes |

The tree-over-MLP comparison is the mirror of the raw MLP-over-tree rows. The
tree passes both axis worlds and fails both smooth worlds, so neither family
dominates.

## Held-out Brier means

| World | Scalar | Matched linear | Tree | MLP |
|---|---:|---:|---:|---:|
| Linear aligned | 0.250 | 0.066 | 0.062 | 0.038 |
| Linear reversed | 0.284 | 0.075 | 0.062 | 0.037 |
| Smooth aligned | 0.250 | 0.226 | 0.110 | 0.064 |
| Smooth reversed | 0.285 | 0.226 | 0.110 | 0.067 |
| Axis aligned | 0.248 | 0.230 | 0.060 | 0.107 |
| Axis reversed | 0.287 | 0.230 | 0.060 | 0.107 |

The matched optimizer closes most of v2's linear-world gap: the MLP's lead over
the linear model on the linear worlds fell from 0.12 to 0.16 to about 0.03 to
0.04. The rest is absorbed by the DiD.

## Post-hoc checks

These checks were not registered. They reuse the frozen rows and run nothing new.

**Log loss changes the axis-world picture.** MLP-minus-tree advantage:

| World | Brier | AUC | Log loss | MLP better in log loss |
|---|---:|---:|---:|---:|
| Smooth aligned | 0.045 | 0.041 | 0.170 | 16/16 |
| Smooth reversed | 0.042 | 0.039 | 0.159 | 16/16 |
| Axis aligned | -0.047 | -0.036 | -0.015 | 9/16 |
| Axis reversed | -0.046 | -0.037 | -0.011 | 10/16 |

On the axis worlds the tree ranks and classifies better, but its log loss is no
better than the MLP's, and the per-replicate gap ranges from -0.25 to +0.23. Some of its
probabilities are confidently wrong. Brier is the registered decision metric,
so the route stands. The difference matters downstream, because ABM agents act
on predicted probabilities, not rankings.

**Selections sit at the high-capacity edge of both grids.** The matched linear
model and the MLP chose the smallest weight decay in 85% of fits. The tree chose
the 300-round schedule in 90%. Neither grid brackets its family's optimum. Both
families are pushed toward more capacity, so the comparison stays roughly
symmetric, but the absolute margins are not tuned optima.

## NABM conclusion

For the Pivot policy layer:

- feature-conditional and nonlinear policies are both needed on these
  generators;
- which nonlinear family is better depends on the geometry of the
  success boundary, so it is an empirical question, not something synthetic
  evidence can settle;
- neural necessity is not established; and
- no model is selected.

Two preconditions for the ABM step were found while designing it. First, Pivot's
structural productive fit is linear in the features, so the current ABM gives
nonlinear policies nothing to learn. An interaction-fit world is needed, with
the current linear fit as the nested control. Second, imitative agents see
about three neighbor outcomes each over a run. That is far below the 256 rows
used here, so evidence scale must be addressed before nonlinear policy arms can
differ.

v3 is not retuned.

## Boundary

This scorecard uses deterministic synthetic cohorts only. It reads zero
empirical rows, performs no empirical fit, changes no generic NABM API, selects
no model, establishes no neural necessity or scientific effect, and authorizes
no package promotion.
