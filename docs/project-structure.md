# Project Structure

This project is expected to produce multiple artifact types: paper material,
internal notes, reusable modules, scripts, experiment logs, plots, and periodic
archives. The repository should keep these outputs separated so that research
claims, implementation, and results remain traceable.

## Directory Contract

| Path | Purpose |
| --- | --- |
| `ref/` | External or source reference material. Keep imported reports and papers here. |
| `docs/` | Internal design documents, process notes, guidelines, and decisions. |
| `docs/guidelines/` | Stable engineering/research guidelines used by implementations. |
| `docs/decisions/` | Short decision records for scope, architecture, and experiment choices. |
| `docs/toy-models/` | Human-readable specs for toy models and benchmarks. |
| `paper/` | Manuscript drafts, paper outline, figures for publication, and submission notes. |
| `src/neural_abm/` | Reusable Python package or module code for the simulator. |
| `scripts/` | Thin entry-point scripts for running experiments, exporting figures, and archiving. |
| `experiments/configs/` | Versioned experiment configuration files. |
| `experiments/runs/` | Raw run outputs, logs, metrics, and model checkpoints. |
| `experiments/results/` | Curated tables, plots, and summaries derived from runs. |
| `archive/snapshots/` | Periodic timestamped snapshots of docs/configs/results. |
| `archive/index/` | Lightweight indexes for quick navigation across archived material. |

## Artifact Flow

1. Research ideas start in `docs/`.
2. Stable choices are promoted into `docs/decisions/`.
3. Toy model specs live in `docs/toy-models/`.
4. Reusable logic is implemented in `src/neural_abm/`.
5. Execution wrappers live in `scripts/`.
6. Run parameters are saved under `experiments/configs/`.
7. Raw outputs go to `experiments/runs/`.
8. Cleaned plots and summaries go to `experiments/results/`.
9. Periodic snapshots go to `archive/snapshots/`, with indexes in `archive/index/`.

## Naming Rules

Use stable names for documents:

- Decision records: `docs/decisions/0001-short-title.md`
- Toy specs: `docs/toy-models/toy-name.md`
- Experiment configs: `experiments/configs/YYYYMMDD_short_name.yaml`
- Run folders: `experiments/runs/YYYYMMDD_HHMMSS_short_name_seedNN/`
- Result summaries: `experiments/results/YYYYMMDD_short_name_summary.md`
- Archive snapshots: `archive/snapshots/YYYYMMDD_HHMMSS_label/`

## Archive Policy

Archives are for quick recovery and navigation, not for hiding active work.
Each archive snapshot should contain:

- Key docs from `docs/`.
- Current paper outline or draft from `paper/`.
- Experiment configs used since the previous snapshot.
- Result summaries and plots, not necessarily every raw checkpoint.
- A short `MANIFEST.md` describing what changed.

The active project should remain readable without opening the archive. Archive
indexes should point back to active docs when possible.
