from __future__ import annotations

import torch
import torch.nn as nn
from config import Config


class QNetwork(nn.Module):
    """
    Feed-forward MLP for Deep Q-Learning.

    Modes:
    ──────
    • Standard DQN  (config.DUELING = False)
        Input → Linear(H1) → ReLU → Linear(H2) → ReLU → Linear(n_actions)

    • Dueling DQN   (config.DUELING = True)
        Shared backbone:
            Input → Linear(H1) → ReLU → Linear(H2) → ReLU
        Two separate streams:
            Value stream:     Linear(H2, 128) → ReLU → Linear(128, 1)
            Advantage stream: Linear(H2, 128) → ReLU → Linear(128, n_actions)
        Output Q:
            Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]

    Parameters
    ──────────
    state_dim  : dimension of the observation vector (3 + N_RAYS + N_ACTIONS)
    n_actions  : number of discrete actions
    config     : instance of Config (reads HIDDEN1, HIDDEN2, DUELING)
    """

    def __init__(self, state_dim: int, n_actions: int, config: Config):
        super().__init__()
        h1 = config.HIDDEN1
        h2 = config.HIDDEN2
        self.dueling = config.DUELING

        # ── Shared backbone ───────────────────────────────────────────────────
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        if self.dueling:
            # ── Value stream ──────────────────────────────────────────────────
            self.value_stream = nn.Sequential(
                nn.Linear(h2, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            # ── Advantage stream ──────────────────────────────────────────────
            self.adv_stream = nn.Sequential(
                nn.Linear(h2, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
            )
        else:
            # ── Standard Q-head ───────────────────────────────────────────────
            self.q_head = nn.Linear(h2, n_actions)

        self._init_weights()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """He initialization for ReLU layers, bias=0."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, state_dim) or (state_dim,)
        returns: (batch, n_actions) Q-values
        """
        features = self.backbone(x)

        if self.dueling:
            v   = self.value_stream(features)          # (B, 1)
            adv = self.adv_stream(features)            # (B, n_actions)
            # Aggregation: subtract the advantage mean for stability
            q   = v + adv - adv.mean(dim=-1, keepdim=True)
        else:
            q = self.q_head(features)

        return q