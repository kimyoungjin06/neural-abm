# NABM Unit Mini Examples

These examples are lightweight reuse demos for the `NABMAgent` lifecycle and
the reusable social update unit. They are product-facing smoke examples, not
claim-bearing research validation runs.

`first_run.py` is the clone-first entry check. It imports only from
`neural_abm.api_lite` and prints a compact torch-free summary for users who
just cloned the repository.

`minimal_api_nabm.py` is the stable API release-smoke example. It imports only
from `neural_abm.api` and should stay small enough to catch public facade drift.
`toy_catalog.py` is the torch-free package-profile catalog example. It imports
only from `neural_abm.api_lite` and prints the full capability taxonomy used to
map stable model identifiers to user-facing families.

Each script follows the same shape:

1. Define a small domain config and agent class.
2. Expose `ObservationSpec` and `SocialMessageSpec`.
3. Run local domain rules first.
4. Emit a scalar social channel from each agent.
5. Use `SocialBlock`, `SocialChannel`, and `NABMStep` to mix peer values.
6. Commit the mixed scalar back into the domain policy attribute.
7. Return a compact summary with domain metrics and social diagnostics.

The examples cover different entry points and domain rule surfaces around the
same NABM unit boundary:

- `first_run.py`: compact clone-first check for `api_lite`, the default
  torch-free profile, recommended first toy families, and the next catalog
  example.
- `minimal_api_nabm.py`: compact belief-probability social mixing through the
  stable `neural_abm.api` facade. This is an API smoke, not a domain claim.
- `toy_catalog.py`: no-torch toy catalog lookup through the lightweight
  `neural_abm.api_lite` facade. This is a package-profile smoke.
- `schelling_nabm.py`: residential relocation uses a social move-probability
  channel. Metrics include satisfaction, segregation, and move rate.
- `epidemic_compliance_nabm.py`: infection dynamics use a social compliance
  channel. Metrics include infection rate, compliance rate, and contact
  reduction.
- `market_pricing_nabm.py`: seller clearing uses a social pricing-aggressiveness
  channel. Metrics include mean price, trade rate, inventory dispersion, and
  mean profit.

Run any demo directly from the repo with `uv run python examples/<name>.py`.
