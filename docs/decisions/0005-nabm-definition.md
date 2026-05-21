# Decision 0005: NABM Definition and Claim Boundary

Status: Accepted

Date: 2026-05-12

## Context

The project now contains Toy1-10 plus shared config, result, social, sweep, and
binary runner contracts. The acronym NABM is useful, but it needs a minimal
definition so the project does not overclaim beyond the implemented evidence.

Toy1-5 contain the strongest current NABM paths. Toy6-10 broaden ABM family
coverage and reuse the shared public contracts, but their neural path is more
limited or optional. Toy2, Toy4, and Toy5 also contain valuable non-neural
reference policies that should stay in the suite without becoming the center of
the NABM claim.

## Decision

Use NABM in this repository to mean a Neural Agent-Based Model research toolkit
or prototype suite, not a universal ABM engine.

A model qualifies for the core NABM claim when it has these minimum properties:

- neural agent or policy state that changes the simulated behavior through
  local learning or local inference;
- an ABM time loop with explicit scheduling, state transition, and logging;
- separated local update and social update paths;
- social message, peer selection, typed mixing, and commit stages;
- ABM-style aggregate and micro-state outputs for inspection.

The public claim boundary is:

- This project is not a general-purpose ABM framework.
- This project is not a universal simulator.
- This project is not a Transformer replacement.
- Current results should be described as toy-model and prototype-suite evidence.

## Toy Classification

Use `full`, `compatible`, and `reference` as the only NABM status labels in
documentation and machine-readable metadata.

`full` means neural local or social update is the core experiment path for that
toy. Toy1-5 currently use this label:

- Toy1: neural HK classification with output, latent, and parameter social
  mixing.
- Toy2: neural spatial prisoner's dilemma policy with social distillation.
  `rd_well_mixed`, `fermi_imitation`, and `reputation_imitation` remain
  reference policies.
- Toy3: neural opinion and social output mixing with opinion rewiring.
- Toy4: neural public-goods contribution policy with social distillation.
  `imitation` and `reputation_imitation` remain reference policies.
- Toy5: neural contagion/adoption policy with social distillation.
  `simple_contagion`, `complex_threshold`, and `reputation_imitation` remain
  reference-compatible baselines.

`compatible` means the toy follows the common NABM config, result, social, and
sweep contracts, but the neural path is limited, optional, or not yet the main
mechanism. Toy6-10 currently use this label.

`reference` means a policy or model is intentionally kept as a comparison
baseline rather than as the center of the NABM claim. No Toy1-10 top-level toy
currently uses `reference` as its toy-level status, but Toy2, Toy4, and Toy5
advertise reference policies.

## Consequences

- Capability metadata must expose the NABM status, neural role, social channels,
  and reference policies for each toy.
- Run artifacts must carry the same fields in `metadata.json`, `summary.json`,
  and sweep summary outputs so later tables and paper claims can be audited
  against the machine-readable capability contract.
- Documentation should call the repository a NABM research toolkit or prototype
  suite unless a narrower result justifies stronger wording.
- Non-neural reference policies remain first-class experimental baselines, but
  they should not be described as the core NABM mechanism.
- Toy6-10 should stay `compatible` until their neural local or social update
  becomes the main validated experiment path.
