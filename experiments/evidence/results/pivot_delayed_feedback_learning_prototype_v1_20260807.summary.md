# Pivot delayed-feedback learning prototype v1 result

Date: 2026-08-07

Status: `DESCRIPTIVE_SYNTHETIC_RESULT_REPRODUCED`

## Result

Delayed failure visibility created a clear, reproducible phase change in the
Pivot prototype. With immediate success and failure evidence, imitative
learning stayed close to the frozen arm. Delaying failure by two or four steps
instead produced progressively more entry and more failed pivots in every
paired replicate. Cautionary learning continued to suppress failed pivots, but
its correction weakened as failure evidence arrived later and it always
reduced some productive exploration too.

This is a useful NABM case result, not an empirical researcher-pivot effect.
The development run used 80 synthetic agents, 14 steps, 16 paired replicates,
five visibility regimes, and three learning arms. All 240 raw rows and the
phase-map receipt were regenerated in a second deterministic run; all four
artifact hashes were identical.

## Phase map

| Visibility | Learning | Pivot rate | Productive rate | Failed rate | Productive share |
|---|---|---:|---:|---:|---:|
| Immediate | Frozen | 0.790 | 0.235 | 0.555 | 0.300 |
| Immediate | Imitative | 0.795 | 0.252 | 0.543 | 0.316 |
| Immediate | Cautionary | 0.610 | 0.187 | 0.423 | 0.310 |
| Symmetric lag 2 | Imitative | 0.855 | 0.264 | 0.591 | 0.309 |
| Failure lag 2 | Imitative | 0.934 | 0.273 | 0.660 | 0.293 |
| Failure lag 4 | Imitative | 0.960 | 0.276 | 0.684 | 0.287 |
| Selective failure lag 4 | Imitative | 0.961 | 0.276 | 0.685 | 0.287 |
| Failure lag 2 | Cautionary | 0.701 | 0.209 | 0.492 | 0.299 |
| Failure lag 4 | Cautionary | 0.746 | 0.222 | 0.524 | 0.299 |
| Selective failure lag 4 | Cautionary | 0.756 | 0.225 | 0.531 | 0.300 |

The frozen arm was exactly identical in all five visibility regimes. This is
the expected negative control: changing outcome availability affects behavior
only through the registered learning path.

## Paired contrasts

Relative to immediate visibility, imitative pivot rates rose by `0.061` under
two-step symmetric delay, `0.139` under two-step failure-only delay, `0.166`
under four-step failure-only delay, and `0.166` when only half of failures were
also eligible for release. The paired failed-pivot increases were `0.048`,
`0.117`, `0.141`, and `0.142`. Failure-only delay increased both quantities in
all 16 replicates at delays two and four.

Under immediate visibility, cautionary learning versus frozen reduced failed
pivots by `0.131` and productive pivots by `0.048`. Its failed-pivot reduction
shrunk to `0.063` with failure lag two, `0.030` with failure lag four, and
`0.023` with selective failure lag four. The model therefore reproduces the
registered correction-versus-exploration trade-off and shows that delayed
evidence erodes the correction.

## Why it happens

The first open transition step produces an average of 48.5 private outcomes
among 80 agents. When success is immediately public but failure is delayed by
four steps, the imitative arm's mean attention weight climbs from `1.00` to
`1.14` before the first failure batch arrives. Failure evidence eventually
pushes it down to `0.77`, but the mean pivot rate is already about `0.96` and
the one-time absorbing transition cannot be reversed.

Making only half of failures public changes the final attention weight to
about `0.84` but barely changes the final imitative pivot and failure rates.
This is the most informative non-obvious result: two visibility mechanisms
leave different learned policies but nearly identical saturated endpoints.
The absorbing state hides late policy divergence.

Symmetric delay is less damaging than failure-only delay because it removes
the early success-only learning window. Cautionary learning is identical
between symmetric lag two and failure lag two because that policy reads only
failed outcomes; success timing is a built-in negative control for that arm.

## NABM decision

The Pivot-owned delayed queue worked through the existing NABM lifecycle and
produced a useful phase map without changing the generic API. It should remain
domain-owned for now because no second model family yet requires the same
private-to-public abstraction.

The result also changes the next priority. The largest limitation is no longer
the absence of a delayed queue; it is the absorbing one-time pivot endpoint.
The next Pivot model should add persistence, exit, or redirection so delayed
correction can change an observable outcome. That extension should remain
domain-owned and be compared against the current absorbing predecessor before
any generic framework extraction.

## Boundary

This is deterministic descriptive behavior of a precommitted synthetic model.
It read zero empirical rows, fit no empirical target, selected no model,
changed no generic NABM API, established no neural necessity or scientific
effect, and authorizes no package promotion.
