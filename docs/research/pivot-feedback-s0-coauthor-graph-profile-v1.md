# S0 coauthor graph profile v1

Date: 2026-07-23

## Purpose

The S0 core inventory reports declared pair incidences. Those quantities count
the same author pair again whenever it appears on another work, and therefore
are not graph edges. This component constructs the missing exact graph
topology without introducing an empirical outcome, a shock threshold, field
selection, or a preferred team-size cutoff.

The executable contract is
`experiments/configs/pivot_feedback_s0_coauthor_graph_profile_v1.yaml`.

## Frozen graph definition

For each of the 26 primary fields, publication-year cell, and declared-team
cap in `{10, 50, 100}`:

1. include author events from works whose declared author count is no larger
   than the cap;
2. define nodes as distinct current-snapshot OpenAlex author tokens on those
   works, including tokens with degree zero;
3. define an undirected edge `(u, v)`, with `u < v`, when the two tokens occur
   on at least one included work in the same field-year cell;
4. deduplicate repeated pair occurrences across works;
5. report exact node and edge counts, degree order statistics, connected
   components, largest-component nodes, and the largest-component share.

The profiles are nested descriptive sensitivity surfaces. All three are
reported, none is ranked, and none may be called the selected analysis cap.
An edge or node is stored internally with the smallest cap category at which
it first appears, so the three surfaces are derived from one exact
materialization.

Degree quantiles include zero-degree isolates and use the deterministic order
statistic at `floor(q * (n - 1))`. The largest-component share is accompanied
by its exact numerator and denominator.

The receipt-bound author-event surface uses the non-null integer sentinel
`publication_year=0` for an unknown year. Graph bucketing retains that
sentinel internally and converts it back to a null publication year only in
the compact output. It is therefore a declared null-year cell, not calendar
year zero. The builder separately counts materialized nodes and distinct
edges under every cap and requires exact equality to the compact graph sums,
so a skipped bucket cannot be replaced by a zero-filled output cell.

## Why `ALL` is not expanded

The S0 core counted 47,627,187,584 declared pair incidences for the unrestricted
classified-field profile, compared with 3,105,390,831 at cap 100. Expanding
the unrestricted hyperauthorship tail would dominate the computation while
answering a different robustness question. The unrestricted quantity remains
a declared-incidence diagnostic; it is not silently treated as an exact
distinct-edge graph.

This is a pre-execution capacity boundary, not an outcome-driven choice. S0
requires exact topology under a versioned set of non-ranked cap profiles, and
v1 freezes all three finite profiles before graph results are inspected.

## Exact execution path

The builder first rehashes all 4,096 receipt-bound author-event artifacts. It
is run with the repository-only tool overlay
`uv run --with duckdb==1.5.5 python scripts/build_pivot_feedback_s0_coauthor_graph_profile.py ...`;
DuckDB is deliberately not added to the published package extras. The exact
DuckDB 1.5.5 external self-join materializes author-token pairs by work,
reduces them to one edge per field-year and token pair with the smallest cap
category, and separately materializes the node universe. The compact
field-year output is then computed with exact SciPy connected components.

Raw author tokens and edge lists are scratch artifacts. They are removed
before the output receipt and atomic directory rename, and they are never
placed in the repository. The compact output retains only aggregate topology
and hash-bound accounting.

The strengthened 2026-07-24 real-data pilot re-ran one receipt-bound event
partition after adding the null-year and materialization-closure checks. It
produced all 55,632 core-cell-by-cap rows and preserved 78 null-year profile
rows, 56 of them with nonzero node support and 419 cap-summed node tokens. It
then passed the separate verifier. Its output
receipt SHA-256 is
`46cfb6df036b4708cff19fa338c140cb57c0920763224ad5ca919f9eb51f1147`;
the independent verification report SHA-256 is
`23aa8051f2438ba7fade10ded35007e7e3fb8f560a59f5d526d1b78013cf53c1`.
This remains a permanently nonpromotable engineering pilot.

## Claim boundary

These are graphs of current-snapshot OpenAlex author tokens, not verified
people. Co-occurrence in the snapshot does not prove when OpenAlex first
observed the authorship. Passing this component does not by itself pass S0 or
authorize D0: the independently verified full ORCID and affiliation
projection must also pass, followed by an additive gate receipt. Model
fitting, empirical signatures, causal effects, and scientific effects remain
closed.
