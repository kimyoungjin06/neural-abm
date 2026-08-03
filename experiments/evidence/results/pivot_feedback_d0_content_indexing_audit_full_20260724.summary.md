# Pivot-feedback D0 content/indexing audit summary

Date: 2026-07-24

Status: `PASS_D0_CONTENT_INDEXING_EXTRACTION_HUMAN_REVIEW_REQUIRED`

The outcome-blind audit scanned all 493 current-snapshot main-work files and
15,262 row groups, recovered all 384 predeclared title/metadata samples, and
found no work-ID, publication-year, work-type, primary-topic, title-missingness,
or topic-score-missingness integrity block. It did not read entry, persistence,
realized citation, prediction, or G4 outcomes and did not use a GPU.

This is not a D0 pass. Five indexing-review flags remain, and one is decisive
for the current episode set.

## Decisive finding

The Medicine episode `F27/T12505/1991` changes sharply between its baseline and
signal samples:

| Diagnostic | Baseline | Signal |
| --- | ---: | ---: |
| Median current-snapshot primary-topic score | 0.99695 | 0.03435 |
| Dominant-source share | 0.09375 | 0.84375 |

In the signal sample, 27 of 32 works come from
`Digital Kenyon (Kenyon College)` and carry titles of the form `PVN-LOT-*`.
Those records are archival inventory objects rather than a coherent surge in
peptidase-inhibition research. The observed attention jump is therefore likely
driven at least partly by current-snapshot indexing/source influx.

Episode 3 is quarantined pending the predeclared human double review and a
versioned reselection decision. The other episodes are not promoted merely
because this episode failed.

## Boundary

The compact receipt SHA-256 is
`86be1808a3272ba7a892d8aa90f4d3687f0d253474d1b4470dad1940e4ff5280`;
the private artifact-tree SHA-256 is
`f1182738fb7c8541f425baf861ca07549b52b29fb9a1c4b133cc2807a0b44e7c`.

Human double review is incomplete. D0, G0, leave-out/network construction,
outcome opening, model fitting, G4, empirical associations, causal mechanisms,
and scientific effects remain closed, unauthorized, not run, or not
established.
