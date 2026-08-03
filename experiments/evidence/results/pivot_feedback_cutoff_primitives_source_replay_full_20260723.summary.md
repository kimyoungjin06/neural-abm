# Pivot-feedback cutoff-primitives source replay: full result

Status:
`PASS_INDEPENDENT_FULL_SOURCE_PROJECTION_REPLAY_AND_CANONICAL_TIME_VERIFICATION_WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS`

The additive v2 verifier independently replayed the frozen eight-column
main-work projection from all 493 source files. It read 492,293,207 raw rows,
quarantined the single frozen `W-1` row, and compared all 492,293,206 retained
rows with the published cutoff-metadata output in 4,096 work-ID partitions.

Every work ID, work-type code, paratext flag, retraction value and null mask,
declared author count, and primary topic, field, and domain value and null mask
matched. The independent source and output projection digests are both
`9a8f9b13250ab8b11aecd19b0d67b2a1dba9d4395e1d6e63da79f50f8989abf3`.
The partition comparison tree is
`23dab2cd3d39aeb381e977a58ac181a8ba16e88a1af5f52f38fbeb93508b9f92`.

The verifier also rehashed all 4,101 cutoff artifacts, rehashed all 4,096
canonical spine shards, replayed canonical publication-time equality, and
confirmed that the production output tree was unchanged. The external full
verification report is 4,974 bytes with SHA-256
`889f45167345187b520100a90fd5434feca1cda2b1137e44061c3d13cfedaefc`.

The run took 850.73 seconds and used CPU/I/O only. Temporary source spools were
removed after success.

The current source files were bound to the content-freeze SHA checkpoints
through exact unchanged device, inode, mode, size, mtime, and ctime identities.
This run did not rehash all 150,062,449,219 source bytes a second time, and the
summary does not claim that it did.

This pass closes the row-level source-projection gap in the normalized cutoff
foundation. It does not establish author-person identity, historical topic or
citation observability, a final G0-resolved panel, G-1, G0, empirical G1, an
NABM effect, or a scientific effect.
