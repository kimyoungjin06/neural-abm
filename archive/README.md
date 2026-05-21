# Archive

The archive keeps timestamped snapshots for quick recovery and navigation.

Use `archive/snapshots/` for snapshot folders and `archive/index/` for search
indexes or manifest summaries.

Suggested snapshot naming:

```text
archive/snapshots/YYYYMMDD_HHMMSS_label/
```

Each snapshot should include a `MANIFEST.md` with:

- Snapshot date and label.
- Included docs.
- Included configs.
- Included result summaries.
- Notes on what changed since the previous snapshot.

Do not use the archive as the primary working location. Active work belongs in
`docs/`, `src/`, `scripts/`, `paper/`, and `experiments/`.
