# Pivot-feedback D0 candidate-content dry-run v1

## Why candidate-level aggregation is needed

The 384-title router classified individual titles, while the adaptive frontier
requires one fail-closed content state per field-topic-entry-year candidate.
The aggregation must preserve uncertainty and must not confuse semantic
off-topic evidence with a current-snapshot indexing discontinuity.

## Simple rule

For each candidate, the baseline and signal samples provide 64 recent titles.
The conservative accept and reject fractions use all 64 titles as the
denominator. Ambiguous and quarantined titles therefore count as neither
accept nor reject.

The precedence is:

1. flag an indexing anomaly when both the median primary-topic score drops by
   at least 0.15 and the dominant-source share rises by at least 0.25;
2. otherwise flag semantic off-topic only if the recent reject lower bound is
   at least 0.50;
3. otherwise pass only if the recent accept lower bound is at least 0.50; and
4. remain unresolved otherwise.

The earlier sample is retained as descriptive historical context. It does not
have to pass, because a genuine emerging topic can have a weak earlier
current-snapshot footprint.

## Result

T14430, T10579, and T12421 would pass the primary dry-run rule. T12505 would
fail specifically as an indexing anomaly: its median score drops from 0.99695
to 0.03435, dominant-source share rises from 0.09375 to 0.84375, and 27 of 32
signal titles are opaque archival identifiers.

The three-pass/one-indexing-fail result remains at accept thresholds 0.50 and
0.60. At 0.70, T12421 becomes unresolved. This makes the T12505 conclusion
stable over the declared grid but shows that T12421 is threshold-sensitive.

## Adaptive consequence

The result is not applied to production states. Counterfactually, freezing
T12505 as `CONTENT_FAIL` would make Medicine rank 2, T10543/1993, the next
required candidate; rank 6 T12421 would remain the first structurally
compatible later candidate under a rank-2-pass branch. T10543 has no frozen
topic card. A targeted successor would need a 96-title local sample rather
than an eager 22.4-million-work title copy.

## Claim boundary

All four production states remain `CONTENT_UNRESOLVED`. The rule is not
semantic ground truth or an accuracy estimate. No raw scan, new API request,
production disposition, selection certificate, reselection, outcome read,
model fit, G4 opening, or scientific claim occurred or is authorized.
