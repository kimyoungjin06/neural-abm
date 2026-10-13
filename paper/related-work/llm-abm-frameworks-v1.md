# LLM-based ABM frameworks: related-work contrast v1

Date: 2026-10-13

Status: literature note for Paper-1. It is not a claim of this repository.

## Purpose

The Paper-1 central claim is that NABM separates policy learning from social
interaction, so fixed policies, adaptive policies, and external priors can be
compared and traced within one simulation lifecycle
([claim matrix](../claim-matrix.md)). Related work must place that claim
against the LLM-based agent-based modeling (LLM-ABM) frameworks that now
dominate "AI agents in social simulation". This note records what each
framework documents, from primary sources, on the axes that matter for the
claim.

## Method

- **Sources.** Primary only: papers (arXiv or venue), official READMEs and
  docs, and code at the commit or version named. Facts were gathered on
  2026-10-13.
- **Spot checks.** Several facts were checked twice:
  - MiroFish's dependence on OASIS, its license, and its star count;
  - MiroBench's OASIS result;
  - Concordia's scripted component and seed parameter;
  - the AgentTorch archetype abstract.
- **"Not documented"** means the primary source does not state it. It does
  not mean the feature is absent.
- **Third-party claims** are labelled as such. Blogs are not peer-reviewed.

## Contrast table

| Framework | What decides an action | How behavior changes over time | Learning across agents | Rule agents in the same environment | Reproducibility controls | Largest reported scale | Validation against real data |
|---|---|---|---|---|---|---|---|
| MiroFish (2025–) | An LLM call per active agent per round, on OASIS | Memory written to a Zep graph | Not documented | Not documented | No seed set in the code; temperature not passed to agents | "Thousands of agents" claimed; no measured run | Not documented |
| OASIS (2024) | An LLM call per activated agent, 21–23 actions | Memory; follow and unfollow | No (proposed as future work) | Predefined `ManualAction` allowed; no comparison reported | Model-assignment seed only; 10 repeated runs, one topic diverging | 1M regular plus 196 core agents, 24 A100s for a week | Twitter15/16 spread, normalized RMSE about 30%; MiroBench: uncalibrated, 0/12 core metrics indistinguishable from Reddit |
| AgentSociety (2025) | LLM calls plus rule parts (gravity model, driving and interest-rate rules) | Memory stream, needs, emotions | Not documented | "Custom agents" supported; no rule-agent comparison | Seed and temperature not documented; hosted DeepSeek-V3 | 10k agents, 5M interactions | Qualitative: Texas UBI, Hurricane Dorian (SafeGraph) |
| Concordia (2023–) | LLM-conditioned components; a Game Master resolves outcomes | Associative memory; self-description | Not documented | `ScriptedAct` and programmed components; no quantitative comparison | API seed parameter; default temperature 1.0 | Not documented | None ("future work") |
| Generative Agents (2023) | LLM perceive, retrieve, reflect, plan | Memory, reflection, planning | No | No (ablations of architecture only) | Per-prompt temperatures in code; no seed | 25 agents, 2 days, "thousands of dollars" | None (believability ratings) |
| 1,000 People (2024–26) | GPT-4o per individual, interview-grounded; not interacting | None within a run | Pooled fine-tuning tested: no gain (0.79 vs 0.84) | Not applicable | Rerun across model versions close (0.64 vs 0.66) | 1,052 individuals | GSS normalized accuracy 0.85 |
| Project Sid (2024) | PIANO: LLM controller plus small non-LLM reflex networks | Memory, action awareness, goals | Not documented | No | Repeats 4–6; seeds not documented; no code | 500 agents in one run | None |
| S3 (2023–25) | LLM per user; emotion and attitude as LLM-driven states | Memory with a forgetting rule | Not documented | Aggregate curves compared with IC, LT, Voter, DeGroot | Not documented | Sub-graphs of 8.6k- and 17.9k-user networks | Individual accuracy 66–84%; propagation patterns |
| YuLan-OneSim (2025) | LLM-generated handler code; LLM calls per action | Memory, reflection; evolving profiles | VR²T fine-tunes the shared backbone (described, not evaluated) | Not documented | Example config temperature 0; no seed | About 100k agents, 8 A100s, about 6,000 s per round | Brazilian rents, visual only |
| SocioVerse (2025) | LLM per agent, single-round questionnaire | None in the experiments | Not documented | Rule and math models listed, not used | Temperature 0.7; no seed | 331,836 agents | Election state-level accuracy 0.92; China survey KL 0.016–0.066 |
| AgentTorch archetypes (2024–25) | LLM queried per archetype each step; agents sample from it | Through prompt state only | No; gradients calibrate simulator parameters | Heuristic agents compared, but at a different population size from the LLM arm | Not documented; 5 runs averaged | 8.4M agents, about 400 queries per weekly decision | BLS participation (3 of 5 boroughs), CDC cases |
| GradABM (2023) | Mechanistic rules, differentiable | Time-varying parameters from a calibration network | No | The rules are the model | Not documented | 800k agents, 4 s on GPU | Influenza and COVID-19 forecasts, 10 counties |
| GABM tutorial (2024) | GPT-3.5 per agent per step | Peers' choices in the prompt | No | No | Rerun 20 days later changed results by at most 8% | 20 agents, 7 steps | None |
| NABM, this repository | A small learned policy (linear, tree, or MLP) with a fixed rule or a once-elicited LLM judgment as its prior | Local updates and pooled outcome learning | Yes: a pooled outcome record (repository-only) | Yes: each step reproduces its rule predecessor exactly | Paired seeds; byte-identical reruns on one host | 256 agents × 16 replicates per cell | None; the empirical test closed without an answer |

## Patterns

These are readings of the sources above, not of every LLM-ABM study.

1. **An LLM call decides the action in every LLM-ABM framework reviewed.**
   - Most query once per agent per step.
   - AgentTorch queries once per archetype per step, which cuts the number of
     calls by orders of magnitude.
   - Some frameworks add rule components for cost or realism: AgentSociety's
     gravity model and Project Sid's reflex networks.
2. **Behavior changes through memory and prompt context, not through learned
   policies.**
   - YuLan-OneSim describes population-level fine-tuning but reports no test
     of it.
   - The one direct test of pooled fine-tuning, the 1,000-people study, found
     no gain.
   - In AgentTorch and GradABM, gradients calibrate simulator parameters, not
     agent policies.
3. **Controlled comparison with rule agents in the same environment is rare.**
   - AgentTorch compares heuristic, archetype, and LLM-as-agent policies, but
     its LLM arm runs at a smaller population.
   - S3 compares aggregate curves with classical diffusion and opinion models.
   - Concordia and OASIS can host scripted agents but report no such
     comparison.
4. **Reproducibility controls are thin.**
   - Seeds are documented only in Concordia's API.
   - Repeated runs, where reported, diverge on some topics (OASIS) or shift by
     up to 8% (GABM).
   - Hosted models drift over time (Chen, Zaharia, and Zou 2024).
5. **Validation is mostly qualitative.**
   - A review of 35 LLM-ABM papers found 15 relying only on subjective
     assessment (Larooij and Törnberg 2026).
   - Quantitative checks exist (S3, SocioVerse, the 1,000-people study, and
     AgentTorch), but an independent benchmark found uncalibrated OASIS
     distinguishable from real Reddit discussions on all 12 core metrics
     (MiroBench).
6. **Scale is bought with hardware or with aggregation.** Runs range from 25
   agents costing thousands of dollars, to 1M agents on 24 A100s for a week,
   to 8.4M agents through archetypes.

## Where NABM stands

**What NABM adds relative to these sources.**

- **Policy learning separated from interaction.** Agents carry learned
  policies that change from pooled outcome records. In the frameworks above,
  behavior changes through memory and prompts.
- **A fixed rule as an exact special case.** The rule ABM is reproduced exactly
  before a learned or LLM arm is read. Fixed, adaptive, and prior-informed
  policies therefore run in one design. Within the sources reviewed, no
  framework reports this nesting.
- **An LLM as a prior rather than as the policy.** The LLM is consulted once,
  calibrated to a base rate, and then corrected by outcomes.
  - AgentTorch's archetypes are the closest relative: they also avoid
    per-agent queries.
  - The difference is that archetypes query every step and never learn from
    outcomes.
- **Byte-identical reruns** on one host, with paired seeds.

**What NABM does not do, to be stated in the paper.**

- It does not produce language behavior, rich personas, or open-ended actions.
  The policy-ladder decisions are binary.
- It has no validation against real data. The empirical test of learning was
  closed before confirmatory scoring.
- Its largest design is 256 agents × 16 replicates per cell, far below the
  scales above.
- Its LLM comparisons use one local 4B model in two agent designs.

## Implications for Paper-1

1. **Related work.** Section 2 of the outline should use the six patterns as
   its contrast axes and cite the critiques below for validation, cost, and
   drift.
2. **A comparison arm the paper lacks: an AgentTorch-style archetype
   baseline (N13 candidate).**
   - Arm design: an LLM queried per archetype at each step, without outcome
     learning.
   - It answers the strongest objection to P5, that per-agent LLM agents are a
     weak comparator, because archetypes already beat them at scale.
   - It would run on the GPU server.
3. **A second domain that touches this literature (N13 and N14 candidate).**
   Most LLM-ABM applications are information and opinion spread: OASIS, S3,
   MiroFish, and SocioVerse.
   - The repository already reduces exactly to DeGroot and Granovetter
     (`docs/classical-reductions.md`).
   - S3 compares its outputs with DeGroot, Voter, IC, and LT.
   - A contagion or opinion domain as the second consumer would therefore link
     the framework claim, the classical reductions, and this literature in one
     design.
   - The alternative, a commons domain, is further from it.
4. **Reporting.** Report repeated runs, seeds, cost, and validation status for
   every comparison, since the critiques single out their absence.

## Framework notes and sources

### MiroFish

- **Identity.** "A Simple and Universal Swarm Intelligence Engine, Predicting
  Anything."
  - Built by 666ghj ("MiroFish Team").
  - License AGPL-3.0. Repository created 2025-11-26; v0.1.2 released
    2026-03-07; 78,588 stars on 2026-10-13.
  - No paper or technical report was found.
  - <https://github.com/666ghj/MiroFish>
- **Engine.** "MiroFish's simulation engine is powered by OASIS"; the backend
  pins `camel-oasis==0.2.5`.
- **Workflow.**
  - Seed text: an LLM extracts entity and relation types, and Zep builds a
    graph.
  - Each entity becomes an OASIS agent profile.
  - An LLM writes the simulation configuration: timing, initial posts, and
    activity levels.
  - Agents interact on simulated Twitter and Reddit.
  - A ReportAgent summarizes, and users can interview agents.
- **README claims not found in the code.**
  - The README claims "thousands of intelligent agents" and dynamic variable
    injection "from a 'God's-eye view'".
  - At commit `7657031` our reading of the code found no mid-run injection:
    `scheduled_events` is always empty.
- **Runs, logging, and evidence.**
  - One LLM call per active agent per round; any OpenAI-format API.
  - No `random.seed` in the repository. Actions are logged to `actions.jsonl`
    without prompts.
  - No baseline or accuracy evaluation is documented.
  - A non-peer-reviewed practitioner test found the swarm's one-day S&P 500
    call matched a single call to the same model
    (<https://cms.research.wpp.com/?p=1700>).

### OASIS

- **Identity.** Yang, Zhang, et al., "OASIS: Open Agent Social Interaction
  Simulations with One Million Agents", arXiv:2411.11581 (2024), NeurIPS 2024
  workshop. License Apache-2.0. <https://github.com/camel-ai/oasis>
- **Agents.**
  - One LLM call per activated agent. Llama3-8B-Instruct in the paper; local
    models supported via vLLM.
  - A time engine and a recommender system decide who acts and what each agent
    sees.
- **Scale.** "196 core agents and 1 million regular agents", 60 steps, on 24
  A100 GPUs within a week.
- **Validation.**
  - Twitter15/16 propagation: normalized RMSE about 30%.
  - Herd effect stronger than in humans.
  - Ten repeated runs: one topic diverged.
  - MiroBench (Yu, Yu, Luo, Wang, arXiv:2606.14715, 2026): for uncalibrated
    OASIS, "both MWU and KS counts are 0/12" across five domains.

### AgentSociety

- **Identity.** Piao et al., "AgentSociety: Large-Scale Simulation of
  LLM-Driven Generative Agents Advances Understanding of Human Behaviors and
  Society", arXiv:2502.08691 (2025).
  - License Apache-2.0 except a commercial package.
  - Successor AgentSociety 2, arXiv:2607.11895 (2026).
  - <https://github.com/tsinghua-fib-lab/AgentSociety>
- **Agents and scale.**
  - "over 10k agents, simulating their 5 million interactions", DeepSeek-V3 via
    API.
  - Rule components: a gravity model for place choice, IDM and MOBIL for
    driving, and a Taylor rule for interest rates.
  - About 54,000 LLM calls per round at 10⁴ agents; "the LLM time remains the
    primary bottleneck."
- **Validation.** UBI effects described as similar to Texas's program, and
  Hurricane Dorian visits following SafeGraph trends. No error metrics are
  given.

### Concordia

- **Identity.** Vezhnevets et al., "Generative agent-based modeling with
  actions grounded in physical, social, or digital space using Concordia",
  arXiv:2312.03664 (2023). License Apache-2.0; `gdm-concordia` 2.4.0 (2026).
  <https://github.com/google-deepmind/concordia>
- **Agents.** Agents assemble text from components to condition LLM calls, and
  a Game Master resolves outcomes.
- **Rule agents.** "Components can also have internal logic programmed using
  classic programming." `ScriptedAct` gives "a predefined sequence of actions"
  "for situations where you need deterministic behavior."
- **Reproducibility.** The sampling API takes an optional seed; default
  temperature 1.0.
- **Validation.** "Validating that the resulting agents indeed exhibit those
  traits is part of the future work."

### Generative Agents and the 1,000-people follow-up

- **Generative Agents.** Park et al., "Generative Agents: Interactive
  Simulacra of Human Behavior", UIST 2023, arXiv:2304.03442.
  - 25 agents for two days cost "thousands of dollars in token credits."
  - Ablations hold the environment fixed. Evaluation is by believability
    ratings.
- **The 1,000-people follow-up.** Park et al., "Generative Agent Simulations
  of 1,000 People", arXiv:2411.10109 (2024; retitled in v3, 2026).
  - Interview-grounded GPT-4o agents reach normalized GSS accuracy of 0.85.
  - Fine-tuning on 500 agents' answers did not improve held-out accuracy (0.79
    vs 0.84).

### Project Sid

- **Identity.** Altera.AL, "Project Sid: Many-agent simulations toward AI
  civilization", arXiv:2411.00114 (2024). No code is released.
- **Agents.** The PIANO architecture runs concurrent modules: an LLM cognitive
  controller plus "small, fast non-LLM neural networks."
- **Scale.** Typical societies have 50–100 agents, and one run has 500. Runs of
  more than 1,000 agents exceeded the server.

### S3

- **Identity.** Gao et al., "S³: Social-network Simulation System with Large
  Language Model-Empowered Agents", arXiv:2307.14984 (v3, 2025).
- **Agents.** GPT-3.5 or ChatGLM-6B, with an LLM-driven emotion and attitude
  process.
- **Comparison.** v3 compares S3 with IC and LT for information spread, and
  with Voter, DeGroot, and neural models for opinion and emotion.

### YuLan-OneSim

- **Identity.** Wang et al., "YuLan-OneSim: Towards the Next Generation of
  Social Simulator with Large Language Models", arXiv:2505.07581 (2025).
  License Apache-2.0. <https://github.com/RUC-GSAI/YuLan-OneSim>
- **Agents.** An LLM generates the scenario specification and handler code.
  "Agents of the same type share identical behavior logic."
- **Scale.** About 100k agents on 8 A100s; about 6,026 s per round.
- **Learning.** VR²T fine-tunes the shared backbone, but its effect is not
  evaluated.

### SocioVerse

- **Identity.** Zhang et al., "SocioVerse: A World Model for Social Simulation
  Powered by LLM Agents and A Pool of 10 Million Real-World Users",
  arXiv:2504.10157 (2025). Code partly released.
- **Agents.** LLM agents answer questionnaires in a single round.
- **Results.** Up to 331,836 agents. Election state-level accuracy reaches
  0.922 with Qwen2.5-72B.

### AgentTorch and LLM archetypes

- **Identity.** Chopra et al., "On the Limits of Agency in Agent-based
  Models", AAMAS 2025, arXiv:2409.10568. License AGPL-3.0.
  <https://github.com/AgentTorch/AgentTorch>
- **Agents.**
  - Agents in an archetype "make identical decisions." The behavior of all
    agents costs "K × A queries."
  - In the New York City case, about 400 queries sample one weekly decision
    for 8.4M agents.
- **Comparison.**
  - Mean squared error: unemployment 24.6 for archetypes, 41.1 for heuristic
    agents, and 57.0 for LLM-as-agent.
  - The LLM-as-agent arm ran 300 agents rescaled to the full population.
- **Calibration.** Gradients calibrate simulator parameters to infection and
  unemployment series.
- **Umbrella paper.** Chopra, "Large Population Models", arXiv:2507.09901
  (2025).

### GradABM

- **Identity.** Chopra et al., "Differentiable Agent-based Epidemiology",
  AAMAS 2023, arXiv:2207.09714.
- **Results.** 800k agents over 133 steps in 4 s on a GPU. A calibration
  network maps data to simulator parameters, trained through the
  differentiable simulator.

### Generative agent-based modeling tutorial

- **Identity.** Ghaffarzadegan, Majumdar, Williams, and Hosseinichimeh,
  "Generative agent-based modeling: an introduction and tutorial", System
  Dynamics Review 40(1), e1761 (2024), arXiv:2309.11456.
- **Design.** 20 agents for 7 steps, using GPT-3.5 per agent per step, at
  US$0.10 per run.
- **Reproducibility.** A rerun 20 days later changed results "by not more than
  8%."

## Critiques and surveys

- **Validation and cost.** Larooij and Törnberg, "Validation is the central
  challenge for generative social simulation: a critical review of LLMs in
  agent-based modeling", Artificial Intelligence Review 59, art. 15 (2026);
  preprint arXiv:2504.03274.
  - 15 of 35 studies rely only on subjective assessment.
  - "Calibration" is reduced to prompt engineering.
  - A modest sweep can cost thousands of dollars.
- **Survey.** Gao et al., "Large language models empowered agent-based
  modeling and simulation: a survey and perspectives", Humanities and Social
  Sciences Communications 11 (2024), arXiv:2312.11970.
- **Position paper.** Anthis et al., "Position: LLM Social Simulations Are a
  Promising Research Method", ICML 2025, PMLR 267:81005–81034. It names five
  challenges: diversity, bias, sycophancy, alienness, and generalization.
- **Synthetic survey data.** Bisbee et al., "Synthetic Replacements for Human
  Survey Data? The Perils of Large Language Models", Political Analysis 32
  (2024). Responses vary less than in real surveys, and results drift over
  three months.
- **Identity groups.** Wang, Morgenstern, and Dickerson, "Large language
  models that replace human participants can harmfully misportray and flatten
  identity groups", Nature Machine Intelligence 7 (2025).
- **Information asymmetry.** Zhou et al., "Is this the real life? Is this just
  fantasy? The Misleading Success of Simulating Social Interactions With
  LLMs", EMNLP 2024, arXiv:2403.05020.
- **Model drift.** Chen, Zaharia, and Zou, "How Is ChatGPT's Behavior Changing
  over Time?", Harvard Data Science Review 6(2) (2024).
- **Realism benchmark.** Yu, Yu, Luo, and Wang, "MiroBench: Benchmarking
  Realism in Agentic Simulation of Real-world Discussions", arXiv:2606.14715
  (2026).

## Open checks

- AgentTorch: absolute runtime and cost are not reported, and the per-borough
  correlations appear only in figures.
- S3: whether code is released, and how many agents were actually simulated.
- Project Sid: hardware and cost.
- Some bibliographic details (Gao et al.'s article number, Zhou et al.'s page
  range) came from secondary indexes and should be confirmed before
  submission.
