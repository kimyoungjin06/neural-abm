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
`research_pivot_scenario_lite.py` is the torch-free general-researcher scenario
example. It imports only from `neural_abm.api_lite` and uses `scenario_lite`
helpers to compare a science-of-science PIVOT question across baseline,
interdisciplinary-support, and hot-field-hype environments.
`research_pivot_study.py` is the replicated research-study version of the same
question: stochastic populations, seed-paired replicates, outcome
distributions, and sensitivity sweeps.
`research_pivot_learning_study.py` is the torch-backed Study 2: the same
environment with learning agents on the `neural_abm.api` lifecycle, comparing
frozen, imitative, and cautionary (failure-only) social learning rules. The
full two-study writeup with figures is
[docs/case-studies/researcher-pivot/README.md](../docs/case-studies/researcher-pivot/README.md).
`classical_reductions.py` is a deterministic torch-free regression example. It
contains exact DeGroot and Granovetter special cases plus explicitly labeled
FJ-like pre-mix anchored and self-excluding HK variants; the corresponding
claim boundaries are documented in
[docs/classical-reductions.md](../docs/classical-reductions.md).

Each script follows the same shape:

1. Define a small domain config and agent class.
2. Expose `ObservationSpec` and `SocialMessageSpec`.
3. Run local domain rules first.
4. Emit a scalar social channel from each agent.
5. Use `SocialBlock`, `SocialChannel`, and `NABMStep` to mix peer values.
6. Commit the mixed scalar back into the domain policy attribute.
7. Return a compact summary with domain metrics and social diagnostics.

The torch-free scenario example uses the same responsibility boundary without
requiring `NABMUnit`, `SocialBlock`, or torch-backed agent protocols:

1. Declare the research question and outcome field.
2. Define baseline and counterfactual scenarios.
3. Apply local decision pressure and typed peer exchange.
4. Commit the mixed value through a domain-owned transition.
5. Compare scenario outcomes and return aggregate/micro audit rows.

The examples cover different entry points and domain rule surfaces around the
same NABM unit boundary:

- `first_run.py`: compact clone-first check for `api_lite`, the default
  torch-free profile, recommended first toy families, and the next catalog
  example.
- `minimal_api_nabm.py`: compact belief-probability social mixing through the
  stable `neural_abm.api` facade. This is an API smoke, not a domain claim.
- `toy_catalog.py`: no-torch toy catalog lookup through the lightweight
  `neural_abm.api_lite` facade. This is a package-profile smoke.
- `classical_reductions.py`: deterministic bounded-scalar examples covering
  exact DeGroot and Granovetter cases and labeled FJ-like/HK variants.
- `research_pivot_scenario_lite.py`: no-torch science-of-science PIVOT scenario
  through `neural_abm.api_lite`. This is the first general-researcher scenario
  example and shows baseline/counterfactual comparison through the
  question/scenario/state/exchange/transition/outcome sequence.
- `schelling_nabm.py`: residential relocation uses a social move-probability
  channel. Metrics include satisfaction, segregation, and move rate.
- `epidemic_compliance_nabm.py`: infection dynamics use a social compliance
  channel. Metrics include infection rate, compliance rate, and contact
  reduction.
- `market_pricing_nabm.py`: seller clearing uses a social pricing-aggressiveness
  channel. Metrics include mean price, trade rate, inventory dispersion, and
  mean profit.

Run any demo directly from the repo with `uv run python examples/<name>.py`.
