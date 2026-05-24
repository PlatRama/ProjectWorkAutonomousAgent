from __future__ import annotations

import os
import random
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from network import QNetwork


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    FIFO buffer with uniform random sampling.

    Stores transitions (s, a, r, s', done) as pre-allocated numpy arrays
    for efficiency.
    """

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity  = capacity
        self.state_dim = state_dim
        self.device    = device
        self._ptr      = 0      # Circular pointer
        self._size     = 0      # Current number of elements

        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity,               dtype=np.int64)
        self.rewards     = np.zeros(capacity,               dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity,               dtype=np.float32)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        i = self._ptr
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = float(done)

        self._ptr  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor
    ]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self._size


# ── Double DQN Agent ──────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN with optional Dueling architecture.

    Double DQN:
        y = r + γ · Q_target(s', argmax_a Q_online(s', a)) · (1 - done)
    Reduces the systematic overestimation of standard DQN.

    Target network update:
        Soft:  θ_target ← τ·θ_online + (1-τ)·θ_target   every train step
        Hard:  θ_target ← θ_online                        every TARGET_UPDATE_FREQ steps
    """

    def __init__(self, config: Config, device: Optional[torch.device] = None):
        self.cfg = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        state_dim = config.STATE_DIM

        # ── Networks ──────────────────────────────────────────────────────────
        self.online_net = QNetwork(state_dim, config.N_ACTIONS, config).to(self.device)
        self.target_net = QNetwork(state_dim, config.N_ACTIONS, config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # ── Optimizer and loss ───────────────────────────────────────────────
        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=config.LR)
        self.loss_fn    = nn.SmoothL1Loss()   # Huber loss

        # ── Replay buffer ──────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE, state_dim, self.device)

        # ── Agent state ──────────────────────────────────────────────────────
        self.epsilon      = config.EPS_START
        self.train_steps  = 0      # Total number of network updates
        self.episodes_done = 0

        # Accumulators for intra-episode metrics
        self._ep_losses: list = []
        self._ep_q_vals: list = []

        self._in_phase2 = False


    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.cfg.N_ACTIONS)
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q = self.online_net(t)
            return int(q.argmax(dim=1).item())


    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Tuple[float, float]]:
        """
        Performs a single online network update.
        Returns (loss, q_mean) or None if the buffer is too small.
        """
        if len(self.buffer) < self.cfg.MIN_REPLAY_SIZE:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.cfg.BATCH_SIZE)

        # ── Double DQN target ──────────────────────────────────────────────────
        with torch.no_grad():
            # Online net chooses the best action in the next state
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # Target net evaluates that same action
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.cfg.GAMMA * next_q * (1.0 - dones)

        # ── Current Q-values ──────────────────────────────────────────────────
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Loss and backprop ──────────────────────────────────────────────────
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        clip = self.cfg.GRAD_CLIP_PHASE2 if self._in_phase2 else self.cfg.GRAD_CLIP
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), clip)
        self.optimizer.step()

        # ── Target network update ──────────────────────────────────────────────
        self.train_steps += 1
        self._update_target()

        # ── Metrics ──────────────────────────────────────────────────────────
        loss_val  = float(loss.item())
        q_mean    = float(current_q.detach().mean().item())
        self._ep_losses.append(loss_val)
        self._ep_q_vals.append(q_mean)

        return loss_val, q_mean

    # ── Target network update ─────────────────────────────────────────────────

    def _update_target(self) -> None:
        cfg = self.cfg
        if cfg.TARGET_UPDATE_MODE == "soft":
            tau = cfg.TAU
            for p_o, p_t in zip(
                self.online_net.parameters(), self.target_net.parameters()
            ):
                p_t.data.copy_(tau * p_o.data + (1.0 - tau) * p_t.data)
        else:  # hard update
            if self.train_steps % cfg.TARGET_UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        """Called ONCE per episode (not per step)."""
        self.epsilon = max(self.cfg.EPS_END, self.epsilon * self.cfg.EPS_DECAY)
        self.episodes_done += 1

    def set_epsilon(self, value: float) -> None:
        self.epsilon = float(np.clip(value, self.cfg.EPS_END, 1.0))

    # ── Episode metrics ─────────────────────────────────────────────────────

    def episode_metrics(self) -> Tuple[float, float]:
        """Returns (avg_loss, avg_q) for the current episode and clears accumulators."""
        avg_loss = float(np.mean(self._ep_losses)) if self._ep_losses else 0.0
        avg_q    = float(np.mean(self._ep_q_vals)) if self._ep_q_vals else 0.0
        self._ep_losses.clear()
        self._ep_q_vals.clear()
        return avg_loss, avg_q

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save(self, path: str | Path, extra: Optional[dict] = None) -> None:
        """
        Saves a full checkpoint:
          - online and target network weights
          - optimizer state
          - complete configuration
          - agent state (epsilon, train_steps, episodes_done)
          - optional extra metadata (phase, episode, metrics, ...)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "online_state_dict":  self.online_net.state_dict(),
            "target_state_dict":  self.target_net.state_dict(),
            "optimizer_state":    self.optimizer.state_dict(),
            "config":             self.cfg.to_dict(),
            "agent_state": {
                "epsilon":       self.epsilon,
                "train_steps":   self.train_steps,
                "episodes_done": self.episodes_done,
            },
        }
        if extra:
            ckpt["extra"] = extra
        torch.save(ckpt, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Optional[torch.device] = None,
        config_override: Optional[Config] = None,
    ) -> "DQNAgent":
        """
        Loads a checkpoint and reconstructs the agent.

        config_override: if provided, overwrites the config saved in the checkpoint
                         (useful for evaluate.py to change N_RAYS at runtime).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(path, map_location=device, weights_only=False)

        # Reconstructs config from the dictionary embedded in the checkpoint
        if "config" in ckpt:
            valid = set(Config.__dataclass_fields__)
            cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in valid})
        else:
            raise ValueError(f"Checkpoint {path} does not contain a saved config.")

        if config_override is not None:
            cfg = config_override

        agent = cls(cfg, device=device)
        agent.online_net.load_state_dict(ckpt["online_state_dict"])
        agent.target_net.load_state_dict(ckpt["target_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state"])

        state = ckpt.get("agent_state", {})
        agent.epsilon       = state.get("epsilon",       cfg.EPS_END)
        agent.train_steps   = state.get("train_steps",   0)
        agent.episodes_done = state.get("episodes_done", 0)

        return agent

    @staticmethod
    def load_config_from_checkpoint(path: str | Path) -> Config:
        """Reads only the Config from a checkpoint without loading the entire network."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        d = ckpt.get("config", {})
        valid = set(Config.__dataclass_fields__)
        return Config(**{k: v for k, v in d.items() if k in valid})

    def clear_buffer(self) -> None:
        """Clears the replay buffer while maintaining the pre-allocated structure."""
        self.buffer._ptr = 0
        self.buffer._size = 0
        print(f"[agent] Buffer cleared.")

    def reset_optimizer(self, lr: float = None) -> None:
        """Recreates the optimizer, resetting Adam's momentum."""
        lr = lr or self.cfg.LR
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=lr
        )
        print(f"[agent] Optimizer reset with LR={lr}")

    def sync_target(self) -> None:
        """Hard copy online → target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        print(f"[agent] Target network synchronized.")