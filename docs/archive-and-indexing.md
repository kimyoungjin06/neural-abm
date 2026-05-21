# Archive and Indexing Design

The archive system should support two needs:

1. Recovery: preserve coherent snapshots of important project states.
2. Search: make old decisions, configs, and results easy to locate.

The archive should not become the main workspace. Active material stays in
`docs/`, `paper/`, `src/`, `scripts/`, and `experiments/`.

## Snapshot Unit

Each snapshot is a folder:

```text
archive/snapshots/YYYYMMDD_HHMMSS_label/
```

Each snapshot should include:

```text
MANIFEST.md
docs/
paper/
experiments/configs/
experiments/results/
optional_run_summaries/
```

Raw runs and checkpoints should only be copied when the snapshot is tied to a
paper figure or a decision-critical result.

## Manifest Format

Each `MANIFEST.md` should contain:

```text
# Snapshot: YYYYMMDD_HHMMSS_label

## Purpose

## Included Material

## Key Changes Since Previous Snapshot

## Important Decisions

## Result Summaries

## Known Gaps
```

## Index Types

Store indexes in `archive/index/`.

Recommended indexes:

- `decision-index.md`: links to decision records and relevant snapshots.
- `experiment-index.md`: maps experiment names to configs, runs, and results.
- `paper-figure-index.md`: maps manuscript figures to source runs.
- `snapshot-index.md`: one-line summary of each snapshot.

## Snapshot Cadence

Create snapshots at these moments:

- After a major process or design change.
- Before implementing a new toy model.
- After producing a result that may enter the paper.
- Before large refactors.
- Before submitting or sharing a draft.

## Minimal First Archive Script

When implementation begins, add a script that:

1. Creates a timestamped folder under `archive/snapshots/`.
2. Copies `README.md`, `docs/`, `paper/`, `experiments/configs/`, and
   `experiments/results/`.
3. Generates a `MANIFEST.md` stub.
4. Updates `archive/index/snapshot-index.md`.

The script should not copy `experiments/runs/` by default.
