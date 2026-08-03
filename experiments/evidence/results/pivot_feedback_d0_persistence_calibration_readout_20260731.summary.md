# Pivot-feedback D0 persistence calibration readout v1

Status:
`PASS_INDEPENDENT_EXACT_4301_DESCRIPTIVE_PERSISTENCE_CALIBRATION_ALL_VALUE_REPLAY`

The aggregate-only readout used the independently verified 4,301-row
`NEW_TOPIC_ENTRY` persistence source. It found 1,487 positive rows and 2,814
rows without an eligible `t+1..t+3` target-topic work, an unadjusted
descriptive rate of 34.57%.

Among positive rows, the first observed subsequent work occurs at `t+1` for
837 rows (56.29%), `t+2` for 401 (26.97%), and `t+3` for 249 (16.75%). The
three-year work-count distribution is 0: 2,814; 1: 722; 2: 325; 3: 174; 4:
90; and 5 or more: 176. A total of 190 rows have target-topic work in all
three follow-up years.

A separate row-wise verifier independently recomputed all overall, episode,
work-count-bin, first-offset, yearly-pattern, and invariant values. Every
mismatch count is zero.

No raw, confirmatory, holdout, exposure, peer, citation, impact, affiliation,
API, model, or G4 data was read. The episode rates are unadjusted descriptions
and cannot rank fields or establish association, causality, or scientific
effect. This closes the bounded D0 entry-and-persistence calibration path but
does not freeze G0.
