# Pivot-feedback D0 candidate-content dry-run

Status:
`PASS_INDEPENDENT_ZERO_COST_CANDIDATE_CONTENT_DRY_RUN`

The frozen 384-title evidence was aggregated into the four candidates required
by the initial adaptive frontier. The rule uses 64 recent titles per candidate
and treats ambiguous or quarantined titles as neither accept nor reject. It
separately detects an indexing anomaly only when the baseline-to-signal median
primary-topic score drops by at least 0.15 and dominant-source share rises by
at least 0.25.

| Candidate | Recent accept lower bound | Score drop | Source-share increase | Dry-run state |
|---|---:|---:|---:|---|
| F17/T14430/2003 | 0.796875 | 0.061250 | -0.125000 | `WOULD_CONTENT_PASS` |
| F17/T10579/2012 | 0.781250 | 0.000000 | -0.156250 | `WOULD_CONTENT_PASS` |
| F27/T12505/1991 | 0.218750 | 0.962600 | 0.750000 | `WOULD_CONTENT_FAIL_INDEXING_ANOMALY` |
| F27/T12421/1953 | 0.656250 | -0.055200 | 0.218750 | `WOULD_CONTENT_PASS` |

The basic result is unchanged at accept thresholds 0.50 and 0.60. At 0.70,
T12421 becomes unresolved. T12505 remains an indexing-anomaly failure across
the declared sensitivity grid; its signal window contains 27 opaque archival
titles.

These are dry-run states only. Every production state remains
`CONTENT_UNRESOLVED`; no production disposition, selection certificate, or
reselection was created.

If T12505 were later frozen as `CONTENT_FAIL`, the adaptive state machine would
next require Medicine rank 2, T10543/1993. Its recent six-year scope contains
13,080 author-linked work memberships, it has no frozen card, and the sampling
target would be 96 titles. The observed four-topic ambiguity range suggests
15–30 possible API exceptions and a historical pro-rata two-provider cost of
USD 0.009619–0.019238. This is not a current price quote or authorization.

The run read no new titles, network, credentials, API, or outcomes and cost
USD 0. Raw scanning, new API requests, production dispositions, certification,
reselection, fitting, G4, and scientific claims remain closed.
