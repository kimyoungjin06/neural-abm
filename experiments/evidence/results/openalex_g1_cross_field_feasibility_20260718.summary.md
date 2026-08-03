# OpenAlex G-1 Cross-Field Profile Feasibility

## Status and claim boundary

- Component status: `PASS_CROSS_FIELD_PROFILE_FEASIBILITY_ONLY`
- Work Parquet files read: `0`
- Work Parquet rows rescanned: `0`
- This is an exact reprojection of the frozen parent profile plus a
  validation of the small taxonomy graph. It is not field-entity
  validation, candidate selection, G-1, G0, empirical G1, or a
  scientific result.

## Taxonomy graph

| Entity | Rows | Distinct display names |
| --- | ---: | ---: |
| domains | 4 | 4 |
| fields | 26 | 26 |
| subfields | 252 | 244 |
| topics | 4,516 | 4,516 |

All taxonomy IDs are unique, all parent IDs resolve, all embedded
parent names and chains agree, and all declared child IDs and names
agree with unique, exact child sets.
Display names remain labels rather than join keys; 8 subfield display names are
shared across 16 distinct subfield IDs.

## Global profile diagnostics

| Diagnostic | Exact value |
| --- | ---: |
| Retained master works | 492,293,206 |
| Non-null primary field | 354,145,248 |
| Non-null primary-field rate | 71.937870% |
| Null primary field | 138,147,958 |
| Null primary-field rate | 28.062130% |
| Null publication year | 27,565,532 |
| Snapshot-year 2026 | 11,516,735 |
| Publication year after 2026 | 4,010 |
| Publication year before 1900 | 6,692,869 |
| Minimum non-null year | 1000 |
| Maximum non-null year | 2050 |

## Domain totals

| Domain ID | Domain | Fields | Subfields | Topics | Works | Complete | Partial | None usable | Zero authorship |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Life Sciences | 5 | 42 | 614 | 50,073,228 | 28,534,344 | 244,909 | 17,947,182 | 3,346,793 |
| 2 | Social Sciences | 6 | 58 | 1487 | 106,588,124 | 66,589,497 | 681,217 | 29,001,436 | 10,315,974 |
| 3 | Physical Sciences | 10 | 89 | 1571 | 136,069,643 | 77,978,134 | 600,260 | 44,870,437 | 12,620,812 |
| 4 | Health Sciences | 5 | 63 | 844 | 61,414,253 | 45,410,159 | 384,953 | 11,341,295 | 4,277,846 |

## All 26 fields

| Domain ID | Domain | Field ID | Field | Subfields | Topics | Works | Complete | Partial | None usable | Zero authorship | Through 2012 | 2013-16 | 2017-19 | 2020-22 | 2023-25 | Null year | After 2026 |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Life Sciences | 11 | Agricultural and Biological Sciences | 11 | 235 | 21,730,474 | 10,607,510 | 127,049 | 9,692,772 | 1,303,143 | 8,597,872 | 3,185,258 | 2,598,846 | 2,791,404 | 3,448,627 | 1,007,197 | 142 |
| 1 | Life Sciences | 13 | Biochemistry, Genetics and Molecular Biology | 14 | 248 | 20,179,986 | 11,750,369 | 79,168 | 6,679,048 | 1,671,401 | 9,097,914 | 2,407,617 | 1,843,425 | 2,084,500 | 3,414,085 | 1,216,773 | 186 |
| 1 | Life Sciences | 24 | Immunology and Microbiology | 5 | 44 | 2,794,190 | 2,064,771 | 15,033 | 589,807 | 124,579 | 1,384,092 | 337,561 | 253,669 | 291,146 | 455,655 | 51,229 | 25 |
| 1 | Life Sciences | 28 | Neuroscience | 8 | 62 | 4,260,655 | 3,271,722 | 18,453 | 789,977 | 180,503 | 2,099,075 | 525,035 | 412,051 | 449,535 | 638,228 | 89,163 | 39 |
| 1 | Life Sciences | 30 | Pharmacology, Toxicology and Pharmaceutics | 4 | 25 | 1,107,923 | 839,972 | 5,206 | 195,578 | 67,167 | 599,239 | 139,864 | 94,142 | 102,518 | 141,429 | 21,367 | 5 |
| 2 | Social Sciences | 12 | Arts and Humanities | 13 | 266 | 25,903,644 | 13,933,341 | 308,656 | 8,683,448 | 2,978,199 | 13,384,243 | 3,298,260 | 2,456,832 | 2,295,629 | 2,239,726 | 2,119,947 | 201 |
| 2 | Social Sciences | 14 | Business, Management and Accounting | 9 | 146 | 8,835,953 | 6,405,118 | 55,070 | 1,761,657 | 614,108 | 3,596,557 | 1,417,720 | 1,145,249 | 1,048,710 | 1,238,968 | 309,962 | 139 |
| 2 | Social Sciences | 18 | Decision Sciences | 4 | 60 | 3,338,622 | 2,021,421 | 16,437 | 995,869 | 304,895 | 1,283,813 | 484,422 | 383,759 | 413,271 | 561,156 | 179,110 | 21 |
| 2 | Social Sciences | 20 | Economics, Econometrics and Finance | 3 | 107 | 10,838,111 | 5,710,819 | 66,799 | 3,763,476 | 1,297,017 | 4,525,127 | 1,502,831 | 1,193,458 | 1,047,120 | 1,693,498 | 809,908 | 126 |
| 2 | Social Sciences | 32 | Psychology | 7 | 144 | 8,411,909 | 6,025,597 | 37,041 | 1,827,953 | 521,318 | 3,562,477 | 1,180,306 | 962,898 | 1,066,173 | 1,310,952 | 230,355 | 66 |
| 2 | Social Sciences | 33 | Social Sciences | 22 | 764 | 49,259,885 | 32,493,201 | 197,214 | 11,969,033 | 4,600,437 | 21,576,764 | 7,385,832 | 5,954,276 | 5,685,768 | 6,075,027 | 2,220,936 | 358 |
| 3 | Physical Sciences | 15 | Chemical Engineering | 6 | 13 | 886,466 | 693,255 | 3,908 | 134,790 | 54,513 | 491,707 | 107,094 | 78,819 | 80,873 | 99,814 | 21,487 | 8 |
| 3 | Physical Sciences | 16 | Chemistry | 6 | 101 | 6,826,372 | 4,956,784 | 21,210 | 1,235,710 | 612,668 | 4,179,131 | 757,418 | 457,609 | 484,447 | 792,064 | 130,087 | 47 |
| 3 | Physical Sciences | 17 | Computer Science | 11 | 302 | 22,363,553 | 13,146,556 | 97,970 | 7,996,521 | 1,122,506 | 7,926,947 | 2,790,648 | 2,141,567 | 2,376,200 | 6,150,093 | 733,856 | 263 |
| 3 | Physical Sciences | 19 | Earth and Planetary Sciences | 8 | 57 | 7,623,114 | 4,177,442 | 62,453 | 2,842,076 | 541,143 | 3,274,413 | 858,923 | 678,137 | 909,419 | 1,466,234 | 390,669 | 62 |
| 3 | Physical Sciences | 21 | Energy | 5 | 27 | 1,829,058 | 1,280,719 | 10,677 | 339,001 | 198,661 | 620,245 | 291,122 | 244,321 | 281,262 | 343,753 | 29,310 | 11 |
| 3 | Physical Sciences | 22 | Engineering | 16 | 560 | 41,398,802 | 26,848,023 | 205,418 | 11,022,781 | 3,322,580 | 17,949,535 | 5,305,719 | 4,256,952 | 4,218,234 | 6,835,382 | 2,498,410 | 221 |
| 3 | Physical Sciences | 23 | Environmental Science | 11 | 202 | 16,804,016 | 9,702,780 | 81,525 | 5,552,700 | 1,467,011 | 6,829,422 | 2,125,563 | 1,714,203 | 2,159,600 | 2,977,097 | 864,716 | 114 |
| 3 | Physical Sciences | 25 | Materials Science | 8 | 123 | 9,438,387 | 6,971,004 | 41,465 | 1,891,308 | 534,610 | 4,367,806 | 1,313,090 | 1,031,085 | 1,069,855 | 1,278,712 | 306,958 | 56 |
| 3 | Physical Sciences | 26 | Mathematics | 10 | 82 | 4,335,373 | 3,021,470 | 17,147 | 1,052,940 | 243,816 | 2,063,433 | 554,632 | 409,051 | 489,144 | 582,267 | 181,671 | 22 |
| 3 | Physical Sciences | 31 | Physics and Astronomy | 8 | 104 | 24,564,502 | 7,180,101 | 58,487 | 12,802,610 | 4,523,304 | 11,079,104 | 2,494,333 | 1,899,805 | 1,878,623 | 3,022,080 | 4,050,582 | 130 |
| 4 | Health Sciences | 27 | Medicine | 42 | 692 | 51,046,633 | 37,645,562 | 326,657 | 9,646,556 | 3,427,858 | 24,169,918 | 6,393,170 | 5,050,414 | 5,913,322 | 7,701,345 | 1,397,873 | 250 |
| 4 | Health Sciences | 29 | Nursing | 4 | 21 | 1,312,125 | 1,038,359 | 9,447 | 189,155 | 75,164 | 658,997 | 167,722 | 131,451 | 139,395 | 182,184 | 21,870 | 10 |
| 4 | Health Sciences | 34 | Veterinary | 2 | 11 | 496,686 | 366,969 | 2,232 | 103,560 | 23,925 | 267,844 | 59,149 | 45,032 | 46,401 | 64,502 | 9,955 | 5 |
| 4 | Health Sciences | 35 | Dentistry | 4 | 13 | 982,361 | 793,058 | 6,442 | 145,045 | 37,816 | 406,783 | 138,334 | 115,268 | 138,147 | 160,605 | 12,478 | 5 |
| 4 | Health Sciences | 36 | Health Professions | 11 | 107 | 7,576,448 | 5,566,211 | 40,175 | 1,256,979 | 713,083 | 3,470,803 | 1,088,933 | 899,979 | 891,054 | 1,000,407 | 171,817 | 53 |

## Null primary-field bucket

| Works | Complete | Partial | None usable | Zero authorship | Null year | After 2026 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 138,147,958 | 45,238,864 | 197,703 | 74,056,719 | 18,654,672 | 8,487,846 | 1,445 |

## Gate status

| Gate | Status |
| --- | --- |
| `parent_work_source_profile` | `PASS_REVALIDATED` |
| `taxonomy_internal_graph` | `PASS` |
| `cross_field_profile_projection` | `PASS` |
| `field_entity_validity` | `UNVERIFIED` |
| `candidate_eligibility` | `NOT_ESTABLISHED` |
| `final_field_selection` | `PENDING_G0` |
| `g_minus_1_data_readiness` | `PENDING` |
| `g0_measurement_freeze` | `PENDING` |
| `empirical_g1_signature` | `NOT_RUN` |
| `pass_g1` | `false` |

## Taxonomy-reference audit extension

- Declared status: `PENDING`
- Workload class: `local_cpu_io`

The declared extension covers exactly 493 main files plus 930 topic files (1,423 total). It must validate
work-to-taxonomy references, hierarchy labels, duplicate work-topic
edges, primary-topic inclusion, topic scores, and publication dates.

No final field or domain has been selected. Field selection remains
`PENDING_G0`, G-1 remains `PENDING`, G0 remains `PENDING`, empirical G1 remains
`NOT_RUN`, and `pass_g1=false`.

## Provenance

- Result fingerprint SHA-256: `d7dd61e46f2556fe7d97f9ac3d5d5e595bbaa10360bd0bddcf039867b8aa84fe`
- Config SHA-256: `a551207d662cc49e12d4d9c1e7e5abb08b82f39d5e8a3e71055a24f1a9f268b2`
- Tool SHA-256: `8aaa4f58bea7bfa0345b805e781c38501b65cb37b29977d740f21d330871fd63`
- Parent compact summary SHA-256: `df626b897bcb53c1bbc3209dbfbec8f69b84361367e09e1a9ffdf39e293d7b83`
- Parent full report SHA-256: `1c385d42595fc34794116189ce940526e1095ccd2a5da9eccd3da902e8f152ce`
- Parent result fingerprint SHA-256: `8b7d0273a04eb7a231a9d8b466151c562ac80eac6557ff7444508cb174862242`
- Taxonomy canonical SHA-256: `8b9f4e2f492eb6939e1dbb6a8187f06d4ea8fdebb9c1bafce017cd39e54401b4`
- Operator-local input paths are redacted.
