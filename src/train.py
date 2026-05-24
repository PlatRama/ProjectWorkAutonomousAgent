from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from typing import Optional

from config import Config
from agent import DQNAgent
from environment import DQNNavEnv
from metrics_logger import MetricsLogger


def _cleanup_checkpoints(ckpt_dir: Path, keep: int = 3) -> None:
    checkpoints = sorted(ckpt_dir.glob("ep?????.pt"))
    to_delete   = checkpoints[:-keep]
    for f in to_delete:
        f.unlink()

# ── Curriculum helper ─────────────────────────────────────────────────────────

def _phase1_obstacles(episode_in_phase: int, total_phase1: int) -> int:
    """
    Number of obstacles to use in Phase 1: grows linearly from 2 to 8
    as the phase progresses.
    """
    progress = episode_in_phase / max(total_phase1 - 1, 1)
    return int(2 + math.floor(progress * 6))   # 2 → 8


# ── Single episode ────────────────────────────────────────────────────────────

def _greedy_eval(env: "DQNNavEnv", agent: "DQNAgent", cfg: Config,
                 n_episodes: int = 50, seed: int = 0) -> float:
    """Executes n_episodes greedy episodes (ε=0) and returns the success rate."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    agent.online_net.eval()

    successes = 0
    for i in range(n_episodes):
        obs = env.reset(spawn_mode="random", seed=seed + i)
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, _, done, info = env.step(action)
        if info.get("reason") == "goal":
            successes += 1

    agent.epsilon = saved_eps
    agent.online_net.train()
    return successes / n_episodes


def run_episode(
    env:        DQNNavEnv,
    agent:      DQNAgent,
    spawn_mode: str,
    n_obs:      int,
    train:      bool = True,
) -> dict:
    """
    Executes a complete episode and returns a dictionary of metrics.

    train=False → no train_step (evaluation mode during training).
    """
    obs = env.reset(spawn_mode=spawn_mode, n_obstacles=n_obs)
    total_reward = 0.0
    done         = False
    reason       = "timeout"

    while not done:
        action          = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        if train:
            agent.store(obs, action, reward, next_obs, done)
            agent.train_step()

        obs           = next_obs
        total_reward += reward
        if done:
            reason = info.get("reason", "timeout")

    avg_loss, avg_q = agent.episode_metrics()

    return {
        "total_reward": total_reward,
        "steps":        env.steps,
        "success":      reason == "goal",
        "collision":    reason == "collision",
        "timeout":      reason == "timeout",
        "avg_loss":     avg_loss,
        "avg_q":        avg_q,
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: Config, resume_path: Optional[str] = None) -> None:
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    # ── Output directories ────────────────────────────────────────────────────
    exp_dir  = Path(cfg.RESULTS_DIR) / cfg.exp_name
    ckpt_dir = Path(cfg.CHECKPOINT_DIR) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save the used config
    cfg.save(exp_dir / "config.json")

    # ── Environment and agent ─────────────────────────────────────────────────
    env   = DQNNavEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if resume_path:
        print(f"[train] Resuming from checkpoint: {resume_path}")
        agent = DQNAgent.load(resume_path, device=device)
        start_episode = agent.episodes_done
    else:
        agent = DQNAgent(cfg, device=device)
        start_episode = 0

    print(f"[train] Device: {device}")
    print(f"[train] State dim: {cfg.STATE_DIM}  |  N_RAYS: {cfg.N_RAYS}")
    print(f"[train] Dueling DQN: {cfg.DUELING}  |  Target update: {cfg.TARGET_UPDATE_MODE}")
    print(f"[train] Total episodes: {cfg.TOTAL_EPISODES}")
    print(f"[train] Exp dir: {exp_dir}")

    # ── MetricsLogger ─────────────────────────────────────────────────────────
    logger = MetricsLogger(exp_dir, exp_name=cfg.exp_name)

    # ── Curriculum schedule ───────────────────────────────────────────────────
    p0_end  = cfg.PHASE0_EPISODES
    p1_end  = p0_end + cfg.PHASE1_EPISODES
    p15_end = p1_end + cfg.PHASE15_EPISODES
    p2_end  = p15_end + cfg.PHASE2_EPISODES

    best_greedy_sr    = 0.0
    best_success_rate = 0.0   # used only for periodic printing
    recent_successes  = []    # rolling window 100 episodes
    t_start           = time.time()

    for ep in range(start_episode, cfg.TOTAL_EPISODES):
        # ── Phase and parameter determination ─────────────────────────────────
        if ep < p0_end:
            phase      = 0
            spawn_mode = "fixed"
            n_obs      = 0
        elif ep < p1_end:
            phase      = 1
            spawn_mode = "fixed"
            ep_in_p1   = ep - p0_end
            n_obs      = _phase1_obstacles(ep_in_p1, cfg.PHASE1_EPISODES)
        elif ep < p15_end:
            phase      = 15
            spawn_mode = "agent_random"
            n_obs      = cfg.N_OBSTACLES
        else:
            phase      = 2
            spawn_mode = "random"
            n_obs      = cfg.N_OBSTACLES

        # ── Phase transitions ─────────────────────────────────────────────────
        if ep == p0_end:
            agent.set_epsilon(0.40)
            print(f"\n[train] ▶ PHASE 1 — ε reset to 0.40")

        if ep == p1_end and cfg.PHASE15_EPISODES > 0:
            agent.set_epsilon(0.40)
            print(f"\n[train] ▶ PHASE 1.5 — agent_random, ε reset to 0.40")

        if ep == p15_end:
            best_greedy_sr    = 0.0
            best_success_rate = 0.0
            agent._in_phase2 = True
            print(f"\n[train] ▶ PHASE 2 — transitioning...")
            agent.set_epsilon(cfg.EPS_RESTART_PHASE2)
            agent.clear_buffer()
            agent.reset_optimizer(lr=cfg.LR_PHASE2)
            agent.sync_target()
            print(f"[train] ▶ PHASE 2 ready — ε={cfg.EPS_RESTART_PHASE2}  LR={cfg.LR_PHASE2}")

        # ── Episode ───────────────────────────────────────────────────────────
        metrics = run_episode(env, agent, spawn_mode, n_obs, train=True)

        if phase == 2:
            agent.epsilon = max(cfg.EPS_END, agent.epsilon * cfg.EPS_DECAY_PHASE2)
            agent.episodes_done += 1
        else:
            agent.decay_epsilon()

        # ── Logging ───────────────────────────────────────────────────────────
        logger.log(
            episode             = ep,
            phase               = phase,
            reward              = metrics["total_reward"],
            steps               = metrics["steps"],
            success             = metrics["success"],
            collision           = metrics["collision"],
            timeout             = metrics["timeout"],
            epsilon             = agent.epsilon,
            loss                = metrics["avg_loss"],
            q_mean              = metrics["avg_q"],
            global_train_steps  = agent.train_steps,
        )

        recent_successes.append(int(metrics["success"]))
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        rolling_sr = np.mean(recent_successes) if recent_successes else 0.0

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (ep + 1) % cfg.SAVE_EVERY == 0:
            ckpt_path = ckpt_dir / f"ep{ep+1:05d}.pt"
            agent.save(ckpt_path, extra={"episode": ep, "phase": phase,
                                         "rolling_sr": rolling_sr})

        # ── Best model ────────────────────────────────────────────────────────
        if phase == 2 and (ep + 1) % cfg.SAVE_EVERY == 0:
            greedy_sr = _greedy_eval(env, agent, cfg, n_episodes=50, seed=999)
            if greedy_sr > best_greedy_sr:
                best_greedy_sr = greedy_sr
                agent.save(ckpt_dir / "best.pt",
                           extra={"episode": ep, "phase": phase,
                                  "greedy_sr": greedy_sr,
                                  "rolling_sr": rolling_sr})


        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(
                f"Ep {ep+1:5d}/{cfg.TOTAL_EPISODES} | "
                f"Ph {phase} | "
                f"obs {n_obs} | "
                f"SR(100) {rolling_sr:.2%} | "
                f"ε {agent.epsilon:.3f} | "
                f"loss {metrics['avg_loss']:.4f} | "
                f"Q {metrics['avg_q']:.3f} | "
                f"t {elapsed:.0f}s"
            )

    # ── End training ──────────────────────────────────────────────────────────
    print(f"\n[train] Training finished — best greedy SR: {best_greedy_sr:.2%}")
    agent.save(ckpt_dir / "final.pt",
               extra={"episode": cfg.TOTAL_EPISODES - 1, "phase": 2,
                      "rolling_sr": rolling_sr})
    logger.save_summary(last_n=200)
    logger.close()
    env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN Navigation — Training")
    p.add_argument("--exp_name",         default="run_default")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--n_rays",           type=int,   default=16)
    p.add_argument("--n_obstacles",      type=int,   default=8)
    p.add_argument("--dueling",          action="store_true", default=True)
    p.add_argument("--no_dueling",       dest="dueling", action="store_false")
    p.add_argument("--target_mode",      default="soft", choices=["soft", "hard"])
    p.add_argument("--phase0_episodes",  type=int,   default=500)
    p.add_argument("--phase1_episodes",  type=int,   default=1500)
    p.add_argument("--phase2_episodes",  type=int,   default=3000)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--results_dir",      default="results")
    p.add_argument("--checkpoint_dir",   default="checkpoints")
    p.add_argument("--resume",           default=None,
                   help="Checkpoint path to resume from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        exp_name          = args.exp_name,
        SEED              = args.seed,
        N_RAYS            = args.n_rays,
        N_OBSTACLES       = args.n_obstacles,
        DUELING           = args.dueling,
        TARGET_UPDATE_MODE= args.target_mode,
        PHASE0_EPISODES   = args.phase0_episodes,
        PHASE1_EPISODES   = args.phase1_episodes,
        PHASE2_EPISODES   = args.phase2_episodes,
        LR                = args.lr,
        RESULTS_DIR       = args.results_dir,
        CHECKPOINT_DIR    = args.checkpoint_dir,
    )

    train(cfg, resume_path=args.resume)