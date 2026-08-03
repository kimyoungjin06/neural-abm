# G-1A receipt and S0 core inventory

Date: 2026-07-23

## Gate result

`G-1A` now passes its deliberately narrow successor contract:

```text
PASS_G_MINUS_1A_BOUNDED_PRIMITIVE_AND_SOURCE_FIDELITY_
WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS
```

The receipt is
`experiments/evidence/results/pivot_feedback_g_minus_1a_receipt_20260723.summary.json`.
It binds the full canonical time, author-token, work-topic,
citation-chronology-proxy, cutoff-primitives, and independent source-replay
surfaces. The verifier opened no source payload and changed no production
artifact. Its only new authorization is S0.

This is not the legacy `G_MINUS_1` pass. Person identity, historical OpenAlex
observation time, candidate eligibility, work-type inclusion, maturation,
field selection, and final-panel construction remain unresolved at their
assigned later gates.

## Full S0 core scan

The S0 core builder rehashed the 4,096 normalized work-metadata partitions and
the 4,096 distinct author-work event partitions before reading them. It then
scanned:

- 492,293,206 canonical work rows;
- 779,630,258 distinct author-work event rows; and
- 776,803,175 history-usable author-work event rows.

The output contains 18,544 field-year cells for all 26 primary fields,
including one null-year cell per field, and 18,518 non-null-year attention
cells. The observed non-null publication-year range is 1000–2050. This broad
range is retained as a diagnostic; it is not an admissible analysis window.

The classified-field margins are:

| Quantity | Count |
|---|---:|
| canonical works | 354,145,248 |
| usable-date works | 329,740,704 |
| works without a primary field | 138,147,958 |
| source works without publication year | 27,565,532 |
| classified distinct author-work pairs | 675,475,922 |
| history-usable classified author-work pairs | 673,318,978 |
| zero-declared-author classified works | 30,561,425 |
| single-declared-author classified works | 150,389,821 |
| multi-declared-author classified works | 173,194,002 |

The team-cap quantities are descriptive pair incidences, not unique graph
edges and not a selected analytic threshold:

| Non-ranked profile | Pair incidences |
|---|---:|
| team size at most 10 | 1,275,177,176 |
| team size at most 50 | 2,301,207,230 |
| team size at most 100 | 3,105,390,831 |
| all declared team sizes | 47,627,187,584 |

Calendar follow-up support is likewise a descriptive availability count, not
an outcome or a chosen maturation horizon. The 1-, 3-, 5-, and 10-year
profiles contain 307,550,957; 278,577,865; 253,084,208; and 190,968,107
classified works respectively.

The independent compact-output verifier rehashed every output artifact,
rechecked all work/author margins and within-cell inequalities, and recomputed
every annual, trailing-three-year, and trailing-five-year attention value from
the field-year table. It returns:

```text
PASS_INDEPENDENT_S0_CORE_OUTPUT_CONSERVATION_AND_ATTENTION_REPLAY_
DEFERRED_COMPONENTS_STILL_PENDING
```

## Why S0 is still incomplete

Three required components remain explicitly `NOT_RUN`:

1. author-token-to-ORCID stability and split/merge diagnostics;
2. author-work affiliation and institution coverage; and
3. distinct undirected coauthor edges, degree quantiles, and connected
   components under versioned non-ranked cap profiles.

The current author-token counts are distinct within each field-year cell and
then summed; they are not global distinct people. Declared pair incidences are
not distinct graph edges. Neither quantity may be used to imply identified
persons or a measured collaboration topology.

Consequently:

```text
G-1A: PASS_BOUNDED
S0:   PARTIAL_CORE_PASS_DEFERRED_COMPONENTS_NOT_RUN
D0:   NOT_AUTHORIZED
G0:   PENDING
```

The machine-readable current ledger is
`experiments/configs/pivot_feedback_gate_state_v2.yaml`; its validator returns
`PASS_G_MINUS_1A_COMPLETE_S0_PARTIAL_D0_FAIL_CLOSED`.

The next executable task is the receipt-bound authorship source projection for
ORCID and affiliation, followed by the separately versioned coauthor graph
profile. This remains CPU/I/O work; no GPU training is authorized.

## Successor implementation status

Both successor paths now have frozen v1 contracts, producers, independent
verifiers, adversarial tests, and real-data nonpromotable pilots:

- the authorship pilot scanned 1,000,000 raw rows and independently verified
  ORCID checksum, token multiplicity, affiliation, institution, and
  field-year conservation;
- the graph pilot projected one author-event hash partition into all 55,632
  core-cell-by-cap rows and independently replayed degree sums, component
  bounds, and largest-component shares.

These pilots are engineering evidence only. They do not replace the required
1,363-file authorship run or 4,096-partition graph run, and they do not change
the ledger above. The full runs are held behind their declared exclusive
HDD/NVMe execution conditions; S0 and D0 remain closed until both full
receipts and a successor additive gate receipt pass.

## Successor completion update, 2026-07-24

The section above records the historical v2 boundary. Both required full
successors have now run and passed separate verification:

- all 1,361,551,038 raw authorship rows were projected into the
  ORCID/affiliation component; and
- all 4,096 author-event artifacts were reduced into 55,632 exact
  field-year-by-cap graph rows for caps 10, 50, and 100.

The combined completion receipt is
`experiments/evidence/results/pivot_feedback_s0_completion_full_20260724.summary.json`.
The additive v3 ledger now returns
`PASS_G_MINUS_1A_AND_S0_COMPLETE_D0_DISCOVERY_ONLY_AUTHORIZED`.
Only D0 with entry year at most 2012 is newly authorized. G0, final-panel
materialization, model fitting, G4, empirical effects, and scientific effects
remain closed or unestablished. The current result and limitations are
documented in
`docs/research/pivot-feedback-s0-completion-20260724.md`.
