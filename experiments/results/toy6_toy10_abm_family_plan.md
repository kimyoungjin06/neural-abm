# Toy6-Toy10 ABM Family Plan

## Purpose

Toy1-Toy5 now cover classification, binary spatial games, continuous opinions,
public goods, and contagion. Toy6-Toy10 should add structurally different ABM
families that stress the common coordination, validation, logging, and benchmark
contracts without forcing every domain into one runner.

## Implementation Target

- Toy6: implemented first as a multi-action categorical spatial game.
- Toy7: implemented first as a continuous extraction-intensity resource ABM.
- Toy8: implemented as an asynchronous event-driven adoption/failure/recovery
  ABM.
- Toy9: implemented as a heterogeneous-agent binary adoption ABM with
  group-specific local rules and coordination gating.
- Toy10: implemented as a dynamic-network market/ecology ABM with multi-channel
  social messages and topology churn.

## Planned Toys

| Toy | Family | Main Gap It Tests | Coordination Surface |
| --- | --- | --- | --- |
| Toy6 | Multi-action categorical spatial game | Non-binary actions and distribution-valued policy state | `none`, `output_average`; `none`, `output_similarity` |
| Toy7 | Continuous resource/intensity ABM | Continuous bounded scalar actions and resource feedback | `none`, `output_average`; `none`, `output_similarity` |
| Toy8 | Asynchronous event-driven ABM | Partial-agent/event updates instead of synchronous epochs | Event peer selection plus optional output mixing |
| Toy9 | Heterogeneous-agent ABM | Mixed policies, optimizers, and local rules in one population | Capability-gated coordination by agent group |
| Toy10 | Dynamic-network market/ecology ABM | Stateful multi-channel messages and topology churn | Multi-channel social messages and dynamic peer graphs |

## Design Rule

Keep domain runners separate where lifecycle semantics differ. Generalize shared
coordination, config validation, logging metadata, and benchmark matrix first.
Only introduce a generic runner when two or more new toys independently require
the same lifecycle hooks.
