# Pivot-feedback D0 API-only topic-routing dry-run

Status:
`PASS_INDEPENDENTLY_VERIFIED_ZERO_COST_API_ONLY_TOPIC_ROUTING_DRY_RUN`

The frozen four-card router replayed all 384 diagnostic titles independently.
It would avoid 290 API requests (75.5208%) and send 94 ambiguous titles
(24.4792%) to an API exception queue:

| Route | Rows |
|---|---:|
| `NO_API_RULE_ACCEPT` | 229 |
| `NO_API_RULE_REJECT` | 24 |
| `API_REQUIRED_AMBIGUOUS` | 94 |
| `QUARANTINE_UNRESOLVED` | 37 |

| Topic | Avoided | API required | Avoidance |
|---|---:|---:|---:|
| Cognitive radio | 66 | 30 | 68.7500% |
| Adrenal hormones | 71 | 25 | 73.9583% |
| Peptidase | 72 | 24 | 75.0000% |
| Higher education | 81 | 15 | 84.3750% |

The run accessed neither the network nor credentials, submitted zero requests,
and cost USD 0. Prior OpenAI and Claude labels were joined only after routing
for descriptive comparison. Among the 229 rule-accept rows, 174 had prior
dual-high API agreement and zero had either provider label the title
`OFF_TOPIC` or `UNREVIEWABLE`; all 24 rule-reject rows had both providers
label them `OFF_TOPIC`. These observations do not estimate accuracy, error
rate, or calibration because the API labels are not ground truth.

The cards and the current OpenAlex topic snapshot are engineering references,
not historical semantic truth. The 384 titles are not the complete
candidate-work universe. Production dispositions, paid API submission,
episode reselection, pivot classification, outcomes, model fitting, and G4
remain closed. Human review remains `NOT_RUN_BY_DESIGN`.

The next authorized work should materialize and independently verify the
complete title-bearing candidate-work manifest without submitting it to either
API. Only then can the local router determine the actual exception-queue size
and support an explicit cost authorization decision.
