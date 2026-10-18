# Information-spread policy ladder: the LLM arms (D3b)

Date: 2026-10-18 (drafted 2026-10-16)

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_llm_arms_v1.yaml`](../../experiments/configs/info_spread_policy_ladder_llm_arms_v1.yaml)

This amendment freezes the three arms that need an LLM. Their run is stage
D5.

- **Cells.** They run in the cells of
  [study v2](info-spread-policy-ladder-study-v2.md) and are paired with its
  arms, read from D4 v2's `cells.jsonl`.
- **Freeze.** It is frozen in the same commit as study v2, before D4 v2
  runs.
- **Measure.** Every arm is scored on study v2's primary measure: share
  precision on unverified items, whose fact-check is not public at the
  decision. Share precision over all items is recorded beside it.

The user chose on 2026-10-16:

- the D4 cells in full (8 worlds × 20 cells × 16 replicates), now study
  v2's;
- three primary comparisons: L1 − L0, L0 − X0, and L1 − P0;
- copying the pinned Qwen snapshot to the GPU server, with its file hashes
  verified.

The arms were drafted on 2026-10-16, before any D4 result was read. They
were moved to study v2 after D4 failed its null check. The user chose the
unverified measure on 2026-10-18.

## Questions

1. **L1 − L0.** Does a prior elicited from an LLM help the NABM learner,
   compared with the hand rule as its prior?
2. **L0 − X0.** Does the NABM learner differ from AgentTorch-style LLM
   archetypes given the same public information?
3. **L1 − P0.** Does the closed loop add to the LLM prior, as L0 − F1 asks
   of the hand rule?

## Model and readout

The model and readout are those of the Pivot pilots:

- `Qwen/Qwen3-4B-Instruct-2507` at snapshot `cdbee75`, with the file hashes
  of Pivot v4;
- transformers 5.14.1 from an overlay directory, so the project environment
  is unchanged;
- bfloat16 at batch 1, because batching in bfloat16 shifts log-odds;
- the assistant turn prefilled with `Answer: **`, and the readout is the
  logit of `Yes` minus the logit of `No` at the next token.

The softmax mass on the two answers is reported. A median below 0.5 flags a
weak format; the arms still run.

Every prompt asks: "Is this item true? Answer with Yes or No." The model
judges veracity. The decision is the shared rule of every arm: share when
the probability is at least c/(1 + c), with the same exploration floor.

## Arms

| Arm | What it is | Written |
|---|---|---|
| P0 | LLM prior: the model's answers on a design cohort, distilled to a linear rule on the world's features | Before the run |
| X0 | LLM archetypes: one answer per archetype, item class, and public aggregate | Before the run (see below) |
| L1 | The D4 learner of each family, with P0 as its prior | Before and during the run |

**P0.**
- The design cohort has 1,024 profiles per regime. Item features and sharer
  trust are uniform on [0, 1], rounded to the two decimals shown. Contacts
  who shared are uniform from 1 to 6 of 6.
- Regime B's prompt adds the sharers' track record (the trust feature).
- The log-odds are distilled by ridge regression on the features
  (λ = 0.001 per profile), as in Pivot.
- Calibration uses only the stated base rate, 0.4. It shifts the intercept
  until the mean probability over the cohort is 0.4.

**X0.**
- Archetypes are D1's: two threshold bins, times three cost levels in
  Regime B (2 and 6 archetypes).
- Item classes are D1's: two bins each of credibility and novelty (4 classes).
- The prompt describes the archetype and the item class. It also gives the
  public fact-checks of that class so far: exact counts up to 10, then the
  count to the nearest 10 and the percent true to the nearest 5.
- Calibration is one log-odds shift per regime. It makes the mean
  probability over archetypes and classes, before any fact-check, equal 0.4.

**Why X0 can be written before the run.**
- X0's aggregates are fact-checks of items, so they depend on the item
  stream alone, never on decisions.
- So a table over (archetype, item class, aggregate) holds every answer X0
  could need.
- It is elicited once: 486 aggregate states, giving 3,888 prompts in Regime
  A and 11,664 in Regime B.
- This gives the same answers as querying during the run with a prompt
  cache, as N06 did. The cost of querying without a cache is reported per
  run: one query per archetype and item class with a pending decision, per
  step.
- The largest class count in the D4 streams is 173. The grid goes to 200,
  and a run stops if any count exceeds it.

**L1.** L1 uses D4's learner and seeds, with P0 in place of the hand rule. L1
and L0 therefore differ only in the prior.

**Deviation from the brief.** The brief said that X0's agents sample from
the archetype's probability. D1 built the shared cutoff rule instead, and
D3b keeps it: sampling would add decision noise that no other arm has.

## Design

- The cells are study v2's: 2 regimes × 2 families × 4 geometries × 20
  cells, with 16 replicates each.
- P0 and X0 do not depend on the learner family. They run once per regime,
  geometry, and cell, and are paired with both families' groups.
- **Parity.** At replicate 0 of every L1 task, L0 is computed again. It must
  equal D4 v2's value exactly, and any failure blocks reading.
- **D4 v2 input.** D4 v2's record is passed to the run. It must come from
  this freeze commit and the bound study v2 contract, after a byte-identical
  reproduction. So D5 runs in a checkout of the freeze commit.

## Decision rule

The group rule applies to each comparison on the primary measure, with its
own bootstrap keys (tag 7):

- the cell-resampled 95% interval of the mean cell Δ excludes zero;
- and that mean is at least 0.01 in magnitude.

**Per study group.** Each comparison is labelled:

- `POSITIVE` (material, mean > 0);
- `NEGATIVE` (material, mean < 0);
- `INDISTINGUISHABLE`.

**Per regime (6 study groups).**

| Disposition | Condition |
|---|---|
| `MOSTLY_POSITIVE` | `POSITIVE` in at least 4 of 6 |
| `MOSTLY_NEGATIVE` | `NEGATIVE` in at least 4 of 6 |
| `MIXED_OR_NONE` | Otherwise |

**Design check.** In the null geometry, L1 − P0 on the primary measure must
not be material and positive in any group. A violation is `DESIGN_DEFECT_IN_NULL_WORLD` and blocks
reading the study groups.

**Descriptive.** P0 − F1 and X0 − F1, under the same rule but not judged.

## Measures

- **Primary.** Share precision on unverified items.
- **Recorded.** Share precision over all items, and false reach.
- **Cost.**
  - Prompts for P0 and for the X0 table.
  - X0 queries without a cache per run.
  - GPU seconds per prompt.

## Runs and record

1. **Elicitation.**
   - Two elicitations on the GPU server, which must match byte for byte.
   - If they do not, the largest log-odds difference is reported and the
     first is used.
2. **Pilot.** A full-load pilot (2 cells per group) for the time estimate.
3. **Development and reproduction runs.** These run after D4 v2 is
   recorded, in a checkout of the freeze commit. `cells.jsonl` and
   `summary.json` must match byte for byte.
4. **Record.** A summary and a result test.

**Dry run before the freeze.**
- An elicitation of the first prompts of each set, and a smoke run on stub
  log-odds.
- Only answer mass, top token, determinism, and timing were looked at; no
  log-odds and no behavior of any arm.

## Boundary

Synthetic worlds only. No empirical data and no paid API. The model's
knowledge of real information spread is not established: the worlds are
synthetic. The study makes no claim about real people.
