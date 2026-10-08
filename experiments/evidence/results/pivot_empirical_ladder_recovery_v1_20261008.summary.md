# Pivot empirical ladder recovery v1 (N11a)

Date: 2026-10-08

Status: `IDENTIFICATION_PASSES_WITH_A_REGISTERED_ANTICIPATION_LIMIT`

Precommit: `af43b16`. Design:
[`pivot-empirical-ladder-recovery-v1.md`](../../../docs/research/pivot-empirical-ladder-recovery-v1.md)

## What ran

The study ran 7 synthetic worlds × 8 replicates. Each replicate had 63
episodes, 21 of them discovery, with about 200 entrants per episode.
Development and reproduction runs were byte identical, about 4 minutes
each. No empirical row was read.

## Registered result: identification passes

| World | Role | NABM wins | Learned coefficient (true) | Median LL gain vs B2 per episode |
|---|---|---:|---:|---:|
| W1 learning | must recover | **8/8** | 1.00 (1.0) | +47.8 |
| W2 learning, growth and linkage | must recover | **8/8** | 1.00 (1.0) | +43.3 |
| W3 static | must not falsely recover | **1/8** | −0.02 (0) | −0.02 |
| W4 static, growth and linkage | must not falsely recover | **0/8** | −0.02 (0) | −0.01 |
| W5 static, independent drift | must not falsely recover | **0/8** | 0.04 (0) | −0.04 |
| W1w weak learning | reported | 8/8 | 0.50 (0.5) | +11.3 |
| W6 drift toward the success relation | reported | 8/8 | 0.34 (0) | +4.9 |

- **Learning is recovered.** The ladder wins in every learning replicate. It
  estimates the learning strength at its true value, including the weak 0.5.
- **The confounds produce no false win.** Under growth, linkage, and
  outcome-independent drift, the learned coefficient stays near zero.
- **The single W3 win is marginal.** A tree model won replicate 2 with a
  bound of +0.009 against B2. That is within the registered limit.

## The identification limit (W6)

Entrants can also be chosen from the start in the direction of the
episode's eventual success relation. This is anticipation without learning,
and the ladder still prefers NABM, with a smaller coefficient of 0.34. As
built, it cannot yet separate learning from anticipatory selection. The two
differ in timing. Learning can act only after evidence becomes visible;
anticipation acts from the first quarter. An anticipatory competitor model
would test that.

## The sign test (reported)

| | W1 learning | W2 learning, growth | W4 growth only |
|---|---|---|---|
| N10 sign-test statistic, median | +0.073 | +0.147 | +0.082 |
| Positive replicates | 0/8 | 8/8 | 0/8 |

The sign test misses pure learning. It turns positive only when growth is
added, so it responds to growth more than to learning. Growth alone came
close to the threshold.

## Next

- **N11a v2.** Add an anticipatory competitor that applies the episode's
  eventual success relation from the first quarter. Then register that
  learning must also beat it.
- **N11b.** Build risk sets and pre-entry features from the S1 extract.
  This needs authorization and reads no outcomes.
