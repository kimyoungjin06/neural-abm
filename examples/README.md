# NABM Unit Mini Examples

These examples are lightweight reuse demos for the `NABMAgent` lifecycle and
the reusable social update unit. They are not part of the Toy1-Toy5 validation
suite.

Each script follows the same shape:

1. Define a small domain config and agent class.
2. Expose `ObservationSpec` and `SocialMessageSpec`.
3. Run local domain rules first.
4. Emit a scalar social channel from each agent.
5. Use `SocialBlock`, `SocialChannel`, and `NABMStep` to mix peer values.
6. Commit the mixed scalar back into the domain policy attribute.
7. Return a compact summary with domain metrics and social diagnostics.

The three included domains show different rule surfaces around the same NABM
unit boundary:

- `schelling_nabm.py`: residential relocation uses a social move-probability
  channel. Metrics include satisfaction, segregation, and move rate.
- `epidemic_compliance_nabm.py`: infection dynamics use a social compliance
  channel. Metrics include infection rate, compliance rate, and contact
  reduction.
- `market_pricing_nabm.py`: seller clearing uses a social pricing-aggressiveness
  channel. Metrics include mean price, trade rate, inventory dispersion, and
  mean profit.

Run any demo directly from the repo with `uv run python examples/<name>.py`.
