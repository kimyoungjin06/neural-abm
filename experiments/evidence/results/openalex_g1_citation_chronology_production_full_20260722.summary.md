# OpenAlex G-1 Citation Chronology Full Production Summary

Status: `PASS_CURRENT_SNAPSHOT_CITATION_EDGE_CHRONOLOGY_ACCOUNTING_WITH_CANONICAL_TEMPORAL_V3_PARENT_AND_BIBLIOGRAPHIC_EVENT_TIME_CONVENTION_LIMITATION`

The full production run scanned all 3,044 reference Parquet files: 3,042,536,160
rows and 37,119,669,321 source bytes. Exact pair reduction found no duplicate
directed pairs, so the unique-pair count is also 3,042,536,160. The primary
zero-day chronology rule retained 2,858,679,472 internal, non-self pairs whose
cited publication date is no later than the citing publication date.

| Exclusive bucket | Unique pairs |
| --- | ---: |
| self-loop | 1,072,993 |
| source temporal unusable | 903,389 |
| external open-world target | 137,937,261 |
| target temporal unusable | 4,472,749 |
| cited date after citing date | 39,470,296 |
| same day | 5,820,320 |
| positive lag | 2,852,859,152 |

The receipt binds 19,438 artifacts and 49,118,338,001 bytes. Its SHA-256 is
`fbe753c84ceb3858a06c627f318eb756e997f73577498b06a5e88f72cacabe76`.
The independent verifier rehashed every artifact and decoded every row in both
4,096-partition products. It independently reproduced classification, lag,
cross-tab, sensitivity, sortedness, partition, and row-count closures. The
verification report SHA-256 is
`41f33c032282e465294baf2865584c82cf0e277cc0f56f987eb0122260c60f04`.

`available_at` remains a bibliographic convention equal to the citing work's
publication date. It is not the date OpenAlex first observed the edge and does
not prove historical researcher visibility. This component does not choose an
outcome horizon, materialize a cutoff panel, establish G-1 or G0, run empirical
G1, fit an NABM, or establish a scientific effect.
