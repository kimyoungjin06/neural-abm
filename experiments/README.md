# Experiments

Experiment material is split into configs, raw runs, and curated results.

| Path | Purpose |
| --- | --- |
| `configs/` | Versioned experiment settings. |
| `runs/` | Raw outputs from each execution. |
| `results/` | Cleaned tables, plots, and summaries. |

Every run should be traceable to:

- A config file.
- A random seed.
- A code version or archive snapshot.
- A result summary, when promoted.

Representative validation:

```bash
uv run python scripts/run_toy_validation.py
```

The default suite runs Toy 1-10 representative scenarios for seeds 1, 2, and 3
over 50 epochs. It writes generated configs under `configs/generated/`, raw run
folders under `runs/`, and the promoted run table, metric table, and Markdown
report under `results/`.

Toy 6, Toy 7, Toy 9, and Toy 10 sweeps follow the same layout: generated
configs are written under `configs/generated/{label}/`, raw runs under `runs/`,
and summary plus grouped-summary CSV files under `results/`.
