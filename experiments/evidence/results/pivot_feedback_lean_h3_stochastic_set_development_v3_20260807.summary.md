# Lean H3 stochastic set development v3 result

- Completed: `2026-08-07T02:35:51Z`
- Development precommit: `730a6d9f03c77f2eb2b471dd2ee17acf8067959c`
- Status: `COMPLETE_PUBLIC_NONBLIND_STOCHASTIC_DEVELOPMENT_SET_UNDERCOVERAGE_OVERDISPERSION_FAILURE_NO_SELECTION`
- Public scenarios and replicates: 16 x 16 = 256 rows
- Truth files read during estimation: `0`
- Opened challenge and empirical rows read: `0`
- Candidate, estimator, and support selection: `NO`
- Fresh blind authorized: `false`
- Scientific effect: `NOT_ESTABLISHED`

## Main result

The scale audit rejected a fixed absolute probability margin for peak
membership and retained normalized prominence regret. That part worked as an
analytic invariance result. The stochastic estimator did not, however, preserve
the full set-valued uncertainty that motivated the redesign.

Among 208 truth-valid rows, 191 were also estimator-valid (`91.8%`). Within
those 191 rows, direction state matched truth in every row and identification
state matched in `99.5%`, but the four-field interval vector contained the
truth vector in only `58.1%`. The direction label therefore looks much stronger
than the primary interval estimand and cannot substitute for it.

## Where it worked and failed

| Scenario family | Applicability | Field-vector containment | Interpretation |
|---|---:|---:|---|
| Clear point and mixed-field geometry | 100% in registered cases | 100% | Clear peaks and field directions were recovered. |
| Low entry, low support | 15/16 rows | 100% among valid | One panel failed closed; episode-level resolution was fragile. |
| Exact tie and regret `0.05`/`0.10` | 48/48 rows | 0% | The estimator declared valid intervals but narrowed the true sets. |
| Regret `0.20` outside the margin | 16/16 rows | 100% | Clear point geometry was recovered. |
| Adjacent and multimodal sets | 32/32 rows | 0% | Set-state labels survived, but the full interval bounds did not. |
| Flat, boundary, and missing | 48/48 invalid | Not applicable | Structural invalidity failed closed as designed. |
| Beta-binomial probability perturbation | 0/16 truth-valid rows | Not applicable | Zero entry denominators and unresolved peaks defeated the working binomial estimator. |

The near-tie result follows from the rule itself. Requiring each time point to
have at least `0.90` marginal posterior inclusion is not the same as constructing
a simultaneous credible set for the episode gap. At and near the `0.10` regret
boundary, plausible peak times are excluded often enough that averaging episode
bounds produces intervals narrower than the truth.

The beta-binomial stress is deliberately aggressive: low base entry
probabilities with concentration `30` place substantial probability near zero.
Even with 10,000 opportunities this generated many zero entry counts, leaving
conditional persistence without a registered denominator. The result does not
prove empirical overdispersion has this form. It shows that v3 has neither a
hierarchical remedy nor a defensible applicability domain for that stress.

## Why this is not a failed governance gate

No coverage floor, validity minimum, or admissibility threshold was
precommitted. The scale audit is a mathematical rule-selection criterion; the
remaining numbers are descriptive operating characteristics. Consequently,
`58.1%` containment and `0/16` overdispersed applicability are estimator
findings, not reasons to relax a gate or rescue the candidate.

The workflow was proportionate: one code precommit, deterministic public
generation, truth-free estimation, descriptive truth opening, and independent
binding and metric replay. No additional independent review or approval gate is
needed at this public development stage.

## Next methodological step

Do not open a fresh blind yet. A v3-informed successor should replace marginal
time inclusion with a joint posterior peak- or gap-set construction whose
simultaneous uncertainty meaning is explicit. It should also either implement a
hierarchical/beta-binomial working model or state a prospective minimum-support
domain. Set containment, applicability, and direction must remain separate
outputs with their own denominators.

Because that redesign would use this diagnostic, the v3 grid becomes
development evidence. A successor needs a new disjoint public stress grid
before any decision about freezing it for a fresh blind. This is a methodological
validation split, not another governance approval layer.

The canonical development report SHA-256 is
`ecaf85f3b3dc08fea1f9fb5fa6d418437889370918a4b9806a734efaec285e78`;
the independent verification SHA-256 is
`314e59d4f9370685b35e208fb300a5aa6d27a78f1a4775ba2b6477fce61baad7`.

This is public nonblind synthetic development only. It is not estimator
selection, blind confirmation, empirical H3, a mechanism or causal result,
predictive validation, NABM necessity, or a scientific effect.
