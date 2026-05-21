from __future__ import annotations

import torch

from neural_abm import (
    PROBABILITY_DISTRIBUTION_CHANNEL,
    SCALAR_PROBABILITY_CHANNEL,
    STATE_DICT_CHANNEL,
    BinaryReadinessPropagationReport,
    BinaryReadinessPropagationUnit,
    CommitReport,
    LocalUpdateReport,
    NABMAgent,
    NABMLocalStep,
    NABMStep,
    NABMUnit,
    SocialBlock,
    SocialChannel,
    SocialDiagnostics,
    StateDictLoadAdapter,
    binary_peer_mean_values,
    tensor_message_values,
)
from neural_abm.core import ClassificationMLP, NeuralClassificationAgent, clone_state_dict


def make_agent(agent_id: int) -> NeuralClassificationAgent:
    model = ClassificationMLP(input_dim=2, hidden_dim=4, output_dim=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return NeuralClassificationAgent(
        agent_id=agent_id,
        shard_group=f"group_{agent_id}",
        model=model,
        optimizer=optimizer,
        train_x=torch.zeros((4, 2), dtype=torch.float32),
        train_y=torch.zeros(4, dtype=torch.long),
    )


def test_public_api_exports_nabm_unit_contract() -> None:
    channel = SocialChannel(
        name="cooperation_probability",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="scalar_probability_sample",
    )
    step = NABMStep(social_block=SocialBlock(alpha=0.25), channel=channel)

    assert step.channel is channel
    assert CommitReport is not None
    assert NABMUnit is not None
    assert SocialDiagnostics is not None
    assert PROBABILITY_DISTRIBUTION_CHANNEL == "probability_distribution"
    assert NABMLocalStep is not None
    assert LocalUpdateReport is not None
    assert BinaryReadinessPropagationReport is not None
    assert BinaryReadinessPropagationUnit is not None
    assert binary_peer_mean_values is not None


def test_nabm_local_step_delegates_to_update_adapter() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.called = False

        def update(self, scale: float) -> LocalUpdateReport:
            self.called = True
            return LocalUpdateReport(
                losses=[scale, scale + 1.0],
                active_agent_ids=[0, 1],
                diagnostics={"scale": scale},
            )

    adapter = Adapter()

    report = NABMLocalStep(adapter).run(0.5)

    assert adapter.called
    assert report.losses == [0.5, 1.5]
    assert report.active_agent_ids == [0, 1]
    assert report.diagnostics == {"scale": 0.5}


def test_classification_agent_satisfies_nabm_agent_protocol() -> None:
    agent = make_agent(agent_id=0)

    assert isinstance(agent, NABMAgent)


def test_nabm_step_without_commit_returns_mix_and_diagnostics() -> None:
    channel = SocialChannel(
        name="cooperation_probability",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="scalar_probability_sample",
    )
    step = NABMStep(social_block=SocialBlock(alpha=0.5), channel=channel)

    result = step.run(
        values=torch.as_tensor([0.1, 0.9, 0.4]).numpy(),
        peer_ids=[[1], [0], []],
    )

    assert result.mix.channel == "cooperation_probability"
    assert result.commit.committed_agent_ids == []
    assert result.diagnostics.active_agent_count == 2
    assert result.diagnostics.mean_peer_count == 2 / 3
    assert result.diagnostics.micro_row(0)["social_update_norm"] > 0.0


def test_state_dict_commit_adapter_loads_mixed_states() -> None:
    torch.manual_seed(113)
    agents = [make_agent(agent_id) for agent_id in range(2)]
    previous_states = [clone_state_dict(agent.model) for agent in agents]
    step = NABMStep(
        social_block=SocialBlock(alpha=0.5),
        channel=SocialChannel(
            name="parameter_state",
            kind=STATE_DICT_CHANNEL,
            commit_mode="state_dict_load",
        ),
        commit_adapter=StateDictLoadAdapter(agents=agents),
    )

    result = step.run(values=previous_states, peer_ids=[[1], []])

    assert result.commit.committed_agent_ids == [0]
    for key, value in agents[0].model.state_dict().items():
        expected = 0.5 * previous_states[0][key] + 0.5 * previous_states[1][key]
        assert torch.allclose(value, expected)
    for key, value in agents[1].model.state_dict().items():
        assert torch.allclose(value, previous_states[1][key])


def test_nabm_unit_runs_local_social_and_logging_lifecycle() -> None:
    torch.manual_seed(17)
    agents = [make_agent(agent_id) for agent_id in range(3)]
    probe_x = torch.eye(2, dtype=torch.float32)

    def select_neighbors(messages: list[dict[str, object]]) -> list[list[int]]:
        assert len(messages) == 3
        return [[1], [0, 2], []]

    unit = NABMUnit(
        agents=agents,
        step=NABMStep(
            social_block=SocialBlock(alpha=0.5),
            channel=SocialChannel(
                name="probe_probs",
                kind=PROBABILITY_DISTRIBUTION_CHANNEL,
                commit_mode="diagnostic_mix",
            ),
        ),
        peer_selector=select_neighbors,
        social_value_builder=tensor_message_values("probe_probs"),
    )

    report = unit.run(
        local_update_kwargs={"batch_size": 2, "steps": 1},
        message_args=(probe_x,),
        log_args=(probe_x,),
    )

    assert len(report.local_losses) == 3
    assert all(loss >= 0.0 for loss in report.local_losses)
    assert report.peer_ids == [[1], [0, 2], []]
    assert report.social_step.mix.mixed_values.shape == (3, 2, 2)
    assert report.social_step.diagnostics.active_agent_count == 2
    aggregate = report.aggregate_row()
    assert aggregate["agent_count"] == 3
    assert aggregate["mean_local_loss"] == report.mean_local_loss
    micro_rows = report.micro_rows()
    assert [row["agent_id"] for row in micro_rows] == [0, 1, 2]
    assert micro_rows[0]["social_channel"] == "probe_probs"
    assert micro_rows[2]["social_update_norm"] == 0.0
