# Pivot persistence and exit prototype v1 result

Date: 2026-08-07

Status: `DESCRIPTIVE_SYNTHETIC_RESULT_REPRODUCED`

## Result

The nonabsorbing endpoint materially changes what the Pivot model reveals.
Complete failure delay produces an early imitative overshoot followed by a
large exit purge once failure evidence arrives. Selective failure
nonavailability weakens that purge and leaves substantially more failed pivots
active. A final stock alone can therefore reverse or conceal the underlying
entry, exit, and learning path.

All 192 absorbing-versus-persistence pairs retained exactly identical final
cumulative entry and complete entry trajectories. The result is thus an
observation-state extension, not a hidden change to entry behavior. The
384-row development run was repeated; raw rows, persistence map, Markdown,
and receipt were byte-identical.

## Persistence map

| Visibility | Learning | Cumulative entry | Active | Active productive | Active failed | Exited failed | Productive share active |
|---|---|---:|---:|---:|---:|---:|---:|
| Immediate | Frozen | 0.807 | 0.304 | 0.242 | 0.062 | 0.503 | 0.799 |
| Immediate | Imitative | 0.798 | 0.378 | 0.252 | 0.126 | 0.420 | 0.671 |
| Immediate | Cautionary | 0.610 | 0.191 | 0.187 | 0.005 | 0.419 | 0.980 |
| Failure lag 2 | Imitative | 0.934 | 0.284 | 0.273 | 0.011 | 0.649 | 0.963 |
| Failure lag 4 | Imitative | 0.960 | 0.285 | 0.276 | 0.009 | 0.675 | 0.967 |
| Selective failure lag 4 | Imitative | 0.961 | 0.379 | 0.276 | 0.103 | 0.582 | 0.732 |
| Failure lag 4 | Cautionary | 0.746 | 0.224 | 0.222 | 0.002 | 0.522 | 0.991 |
| Selective failure lag 4 | Cautionary | 0.759 | 0.242 | 0.227 | 0.016 | 0.516 | 0.942 |

The frozen persistence result is exactly identical across visibility regimes,
confirming that availability acts only through learning.

## Overshoot, purge, and missingness

Under four-step failure delay, the imitative arm reaches cumulative entry
`0.959` and active failed stock `0.446` by step 7. At step 8, an average of
33.7 failed outcomes becomes public and 29.4 failed agents exit. Final active
failed stock falls to `0.009`, even though cumulative entry reaches `0.960`.

When only half of failures are eligible for public release, step-8 failure
messages fall to 15.7 and exits to 19.4. The final attention weight remains
`0.841` instead of `0.767`, and active failed stock remains `0.103`. Delay and
permanent nonavailability are therefore not interchangeable: delay catches up
at a long enough horizon, while missing evidence leaves a persistent state.

## Why final composition is misleading

Relative to immediate visibility, four-step failure delay raises cumulative
imitative entry by `0.162` and failed exit by `0.255`, yet lowers final active
failed stock by `0.116` and raises the productive share among active agents by
`0.296`. Looking only at survivors would make the more disruptive system
appear better.

Immediate visibility and selective four-step delay have almost the same final
active rate (`0.378` versus `0.379`). They are not the same process. Selective
delay has `0.163` more cumulative entry, `0.162` more failed exit, a different
policy state, and a different survivor composition. This is a useful
equifinality result for the flagship case.

Cautionary learning still has a cost. Under immediate visibility it reduces
active failed stock by `0.057` relative to frozen, but also reduces productive
entry by `0.055`. When all failures eventually become public, its surviving
failed stock is nearly zero; selective nonavailability leaves a larger but
still bounded residue.

## NABM decision

The case now has a nontrivial and reproducible mechanism surface: overshoot,
delayed purge, missing-evidence persistence, survivor-selection reversal, and
stock-level observational equivalence. The Pivot-owned state adapter adds
information, but there is still no reason to move persistence semantics into
the generic NABM API without a second model family.

The next step should be a fair explicit non-neural adaptive baseline using the
same population, randomness, visibility queue, continuation rule, and entry,
stock, exit, persistence, and policy-state targets. The question is now
whether neural learning is needed to reproduce this surface, rather than
whether another Pivot state should be added.

## Boundary

This is deterministic descriptive behavior of a precommitted synthetic model.
It read zero empirical rows, fit no empirical target, selected no model,
changed no generic NABM API, established no neural necessity or scientific
effect, and authorizes no package promotion.
