# OpenAlex G-1 Sentinel Incidence Audit

## Result

The full targeted audit completed twice with the same result fingerprint,
`9a5447ac8ec6cbc2b679da104a7ecf67ff9835c4a71e73923602eda5b288a673`.
It evaluated the source `id` column in all 23 tables and 21,019 files, plus the
work-target columns in all 5,538 referenced-works and related-works files.

| Scope | `W-1` rows | `W-1` files | `W0` rows | `W0` files |
| --- | ---: | ---: | ---: | ---: |
| Source `id` | 21 | 6 | 6 | 6 |
| Reference and related-work targets | 0 | 0 | 0 | 0 |

The nonzero source incidence is:

| Table | `W-1` | `W0` |
| --- | ---: | ---: |
| Main works | 1 | 1 |
| Authorships | 1 | 1 |
| Best-OA host lineage | 1 | 1 |
| Abstract-exception payload | 1 | 1 |
| Locations | 16 | 1 |
| Primary-location host lineage | 1 | 1 |

Thus a consistent `W-1` analysis-view quarantine must remove 21 source rows
across six tables. The declared reference and related-work target columns have
no `W-1` incident edge. `W0` remains an unresolved diagnostic and is not part
of the `W-1` quarantine.

## Provenance

| Commitment | Value |
| --- | --- |
| Parent manifest SHA-256 | `6b5929a0155c033e818a816aad57095380a3907ab558c8d2cd58a89055f22b46` |
| Inventory fingerprint | `1bee200c61e746ee774715de2521550cb2c7a6f2bf08e0f8b06f43a39c9ec473` |
| Config SHA-256 | `0d7fc98d604258be7243725f60518fea478870d9b8c29d2cd58c5745fd6deb52` |
| Tool SHA-256 | `6f7cb43e435da997dca13bc256cee9e0ff2855227d3e5466f2090c207fcc9a30` |
| Raw JSON SHA-256 | `6e8c125c17145f680ffa1dc4d699ec817f37c6e6c1656e55cb3c0995146d018d` |
| Raw Markdown SHA-256 | `ccf9e5702644c8248a1740331b7b503bb447c093967ac9023b123c39b1f498e8` |

The primary run took 14.13 seconds with eight workers. The parent path, size,
and mtime inventory was unchanged at the end. This is a targeted incidence
audit only: it does not apply the quarantine, prove retained-row conservation,
validate every other identifier, establish join coverage, or run G1.
