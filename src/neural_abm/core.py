"""Core neural agent components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from neural_abm.unit import ObservationSpec, SocialMessageSpec


class ClassificationMLP(nn.Module):
    """Small MLP used by the first toy model."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.activation(self.fc1(x))
        logits = self.fc2(hidden)
        if return_hidden:
            return logits, hidden
        return logits


@dataclass
class NeuralClassificationAgent:
    agent_id: int
    shard_group: str
    model: ClassificationMLP
    optimizer: torch.optim.Optimizer
    train_x: torch.Tensor
    train_y: torch.Tensor

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="classification_probe",
            tensor_shape=(None, self.model.fc1.in_features),
            dtype=torch.float32,
        )

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec(
            required_keys=(
                "agent_id",
                "shard_group",
                "probe_probs",
                "latent_summary",
                "confidence",
                "param_norm",
            ),
            tensor_keys=("probe_probs", "latent_summary"),
            probability_keys=("probe_probs",),
        )

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """Return the observation tensor for this simple supervised agent."""

        return x

    def act_or_predict(self, observation: torch.Tensor) -> torch.Tensor:
        """Produce class probabilities for an observation batch."""

        return self.predict_proba(observation)

    def local_update(self, batch_size: int, steps: int) -> float:
        """Contract-friendly alias for local supervised learning."""

        return self.train_local(batch_size=batch_size, steps=steps)

    def train_local(self, batch_size: int, steps: int) -> float:
        """Run local supervised updates and return the last loss."""

        if len(self.train_x) == 0:
            return 0.0
        loss_value = 0.0
        for _ in range(steps):
            idx = torch.randint(0, len(self.train_x), (batch_size,))
            batch_x = self.train_x[idx]
            batch_y = self.train_y[idx]
            logits = self.model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().cpu())
        return loss_value

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=-1)

    def hidden_on(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.model(x, return_hidden=True)
        return hidden

    @torch.no_grad()
    def social_message(self, probe_x: torch.Tensor) -> dict[str, Any]:
        """Emit a bounded message summary for social comparison and logging."""

        logits, hidden = self.model(probe_x, return_hidden=True)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        latent_summary = hidden.mean(dim=0)
        params = flatten_parameters(self.model)
        return {
            "agent_id": self.agent_id,
            "shard_group": self.shard_group,
            "probe_probs": probs.detach().clone(),
            "latent_summary": latent_summary.detach().clone(),
            "confidence": float(1.0 - entropy.cpu() / torch.log(torch.tensor(2.0))),
            "param_norm": float(torch.linalg.vector_norm(params).cpu()),
        }

    @torch.no_grad()
    def log_state(self, probe_x: torch.Tensor) -> dict[str, Any]:
        """Return a compact, flat diagnostic state for tests and logs."""

        message = self.social_message(probe_x)
        return {
            "agent_id": self.agent_id,
            "shard_group": self.shard_group,
            "confidence": message["confidence"],
            "param_norm": message["param_norm"],
            "latent_norm": float(
                torch.linalg.vector_norm(message["latent_summary"]).cpu()
            ),
        }


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a model state dict for synchronous updates."""

    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    """Return a detached flat parameter vector."""

    parts = [param.detach().flatten() for param in model.parameters()]
    return torch.cat(parts) if parts else torch.empty(0)


def parameter_delta_norm(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> float:
    """Compute an L2 norm between two state dicts."""

    total = 0.0
    for key, before_value in before.items():
        delta = after[key].detach() - before_value
        total += float(torch.sum(delta * delta).cpu())
    return total**0.5
