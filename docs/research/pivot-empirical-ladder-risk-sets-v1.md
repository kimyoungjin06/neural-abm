# Pivot empirical ladder: risk sets and pre-entry features v1 (N11b)

Date: 2026-10-12

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_empirical_ladder_risk_sets_v1.yaml`](../../experiments/configs/pivot_empirical_ladder_risk_sets_v1.yaml)

Implements stage N11b of the
[N11 design brief](pivot-empirical-model-ladder-design-v1.md). Its
precondition is the [N11a v2](pivot-empirical-ladder-recovery-v2.md)
result, `IDENTIFICATION_PASSES`.

## What N11b does

N11b builds the rows that the ladder models are later fitted to. It reads
only the private S1 extract and reads no outcome.

1. **Checks before any row.** The S1 receipt must match its recorded hash and
   bind the episode and entrant files. The extract files are bound here by
   hash, since the S1 receipt does not hash them. The G4 guard must match its
   hash and row count, and `/ssd` must hold the 20 GiB budget plus a 50 GiB
   reserve.
2. **Risk sets.** An author is at risk in response year y of an episode when
   they meet four conditions:
   - they have an extract work in the episode field in years y−3 to y−1;
   - they have no work in the episode topic before year y;
   - they are not G4-guarded;
   - in a confirmatory episode, they did not enter any discovery episode.

   This applies S1's pivot-eligibility rule to every author. Each S1 entrant
   is therefore at risk in its entry year.
3. **Events and exits.**
   - A precise entrant has event 1 in its entry quarter and leaves after it.
   - An imprecise entrant (dated 1 January) has event 2 in every quarter of
     its entry year and leaves after that year.
   - N11c registers the likelihood for event 2.
4. **Case-cohort sample.** Each episode keeps every S1 entrant, with its full
   at-risk history. It also keeps a random subcohort of at-risk authors, each
   drawn with probability min(1, max(2000, 20 × entrants) / authors at risk).
   - The draw is SplitMix64 of the author id and the episode, with a fresh
     salt. It never depends on entry or on a feature.
   - The probability and the full risk-set counts are stored, so N11c can
     weight subcohort rows by their inverse.
5. **Pre-entry features**, stored as integer counts from works dated in years
   y−3 to y−1:

   | Feature | Count |
   |---|---|
   | Productivity | Episode-field works |
   | Subfield proximity | Those works in the episode topic's subfield |
   | Career age | Years since the author's first extract work |
   | Incumbent ties | Coauthors whose first topic work predates the episode |

6. **Coauthor entry exposure.** This counts the same coauthors when their
   first topic work falls in the episode's response years.
   - A precise entry counts from the quarter after the entry quarter.
   - An imprecise entry counts from the first quarter of the following year.
7. **Shared environment**, one row per episode and quarter:
   - topic works written only by pre-episode incumbents, in the previous
     quarter and the previous year;
   - visible entries;
   - the full risk-set size;
   - field works and their date precision, counted outside every episode
     topic.

## The outcome boundary

Topic works enter in two ways only:

- as each author's earliest date in an episode topic;
- as counts of works written only by incumbents.

A topic work with an entrant author is either an entry or a follow-up, so it
never enters a count. Field-level counts also leave out every episode topic.
No follow-up of any episode can reach any column. A fixture test checks
this: adding follow-up works leaves every output unchanged.

| Outcome quantity | Computed by N11b |
|---|---:|
| Persistence labels | 0 |
| Follow-up counts | 0 |
| Visible evidence | 0 (that is N11c) |

## Invariants

Any violation stops the build. Only counts are then written, and no panel is
kept.

- The risk set reproduces the S1 entrants exactly, and their entry dates.
- Every S1 entrant has exactly one event.
- No event falls outside the S1 entrants.
- No panel author is G4-guarded.
- No discovery entrant appears in a confirmatory panel.
- No row falls after its author's exit.
- Every row is eligible in its response year.
- Every row has a field work in its window.
- The SQL clock agrees with the hash-bound N10 v2 entry clock.

## Outputs

| File | Content |
|---|---|
| `n11b_summary.json` | Counts and pooled subcohort quantiles. No identifiers and no entrant-versus-non-entrant contrast |
| `n11b_receipt.json` | Hashes of the inputs, identity artifacts, contract, builder, and authorization |
| `n11b_timing.json` | Run times; not an identity artifact |
| `private/` | Panel, environment, and per-episode sampling. Mode 0700, never committed |

A second full run must reproduce every identity artifact byte for byte. On
full-scale random synthetic inputs, the build took about 6 minutes and 57 GB
of memory.

## What the summary informs next

N11a's synthetic worlds had about 600 authors at risk per episode, about a
third of whom entered. The N11b summary reports the empirical shape:

- risk-set sizes;
- entry rates per quarter;
- pooled feature distributions.

Before N11c, a separate decision will use these counts to settle whether
the recovery test must be repeated at that shape.

## Stage status

The brief's stage table predates S1 and N11a. The brief is bound by hash
in the N11a contracts, so it is not edited. Instead, this document records
the current status.

| Stage | Status |
|---|---|
| S1 | Recorded, `S1_PASS` |
| N11a | Recorded; v2 `IDENTIFICATION_PASSES` |
| N11b | This precommit; the run needs authorization |
| N11c, N11d | Need authorization |

## Authorization

The preflight hashes files and reads no row value. A run needs a committed
authorization file that names this contract and builder by hash and quotes
the user. N11b does not authorize outcomes, visible evidence, model fitting,
N11c, or N11d.

## Boundary

N11b reads no source snapshot row, no G4 outcome, and no G1 or
model-selection row. It uses the G4 guard's author-id column only, for the
anti-join, and it fits nothing.
