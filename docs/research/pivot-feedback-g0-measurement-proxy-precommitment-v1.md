# Pivot-feedback G0 measurement/proxy precommitment v1

Date: 2026-07-31

Status:
`PASS_FOUR_OUTCOME_BLIND_MEASUREMENT_PROXY_CHOICES_AND_CLOSED_FIXTURE`

## Frozen choices

This successor freezes the first four previously null G0 choices without
mutating G0 v2 or applying the rules to production discovery data.

| Choice | Frozen primary rule |
| --- | --- |
| Attention surge | Incumbent-only `t-3..t-1` versus `t-6..t-4` log-annual-mean acceleration, normalized within field and decision year by median/MAD; modified `z >= 3.0`, positive raw acceleration, and recent count greater than reference count |
| Public success | Current-snapshot bibliographic proxy using three-year citation counts and a field-by-publication-year top-decile cutoff; all boundary ties are included and the realized fraction is reported |
| Citation maturation | Three complete calendar years; a work is mature at decision year `t` only when `publication_year + 3 <= t - 1` |
| Negative-control equivalence | Average marginal binary-outcome probability-difference scale, practical-null interval `[-0.01, 0.01]`, TOST at one-sided alpha `0.05`, requiring the two-sided 90% interval to lie strictly inside the bounds |

## Why these are policy values

Median absolute deviation is used because it is robust to the extreme values
that define the very tail being screened. Leys et al. motivate MAD over
mean/standard-deviation outlier screening, while also making clear that the
choice among thresholds remains a judgment. The primary modified-z cutoff of
3.0 is therefore a preregistered policy value, with 2.5 and 3.5 retained only
as labeled sensitivity readouts; neither may replace the primary rule after a
coverage result is seen.

The public-success proxy follows the field- and publication-year normalization
logic of percentile-based bibliometric indicators. The contract uses an
inclusive boundary rather than a hidden random tie-break. This can make the
realized share exceed 10%, so the output is called top-decile-anchored rather
than an exact top-10% membership set. Reference cohorts below ten works remain
unresolved rather than receiving a fabricated percentile.

The three-year lag aligns the public-success proxy with the already frozen
three-year persistence horizon. It does not repair the source limitation:
`available_at` remains the citing work publication date in a current OpenAlex
snapshot. It is not the time OpenAlex first observed an edge, the time a
researcher saw it, or evidence of information receipt.

The equivalence gate uses the two-one-sided-tests logic rather than treating a
non-significant zero test as evidence of absence. The one-percentage-point
practical-null bound is a conservative design policy on an interpretable
absolute-probability scale. It is not supplied by the TOST literature and is
not estimated from a confirmatory effect. Low power or a wide interval fails
the gate rather than being reinterpreted as equivalence.

Method references:

- Leys et al., “Detecting outliers: Do not use standard deviation around the
  mean, use absolute deviation around the median,”
  <https://doi.org/10.1016/j.jesp.2013.03.013>.
- Waltman and Schreiber, “On the calculation of percentile-based bibliometric
  indicators,” <https://doi.org/10.1002/asi.22775>.
- Schuirmann, “A comparison of the two one-sided tests procedure and the power
  approach,” <https://doi.org/10.1007/BF01068419>.

## Closed-fixture result

The application tool and an independent implementation replayed:

- two attention normalization groups, with one primary surge;
- one MAD-zero group that produced no surge;
- one ten-work citation cohort with an exact 10% realized share;
- one twenty-work cohort whose cutoff tie raised the realized share to 15%;
- one nine-work reference cohort that remained unresolved; and
- the fixed three-year horizon and `[-0.01, 0.01]` equivalence policy.

The fixture is synthetic. No operator directory was listed or scanned, no
Parquet file or production row was opened, and no discovery-production,
confirmatory, or holdout rule application was run. Thirteen quantitative
choices and all required G0 sections still remain for later successors.
