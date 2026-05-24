from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from agent import DQNAgent
from config import Config
from environment import DQNNavEnv


# ── Protocollo di valutazione standard ───────────────────────────────────────

def evaluate(
    agent:        DQNAgent,
    cfg:          Config,
    n_episodes:   int            = 200,
    seed:         int            = 42,
    render:       bool           = False,
    spawn_mode:   str            = "random",
    extra_label:  Optional[str]  = None,
) -> dict:
    # Modalità greedy
    agent.online_net.eval()
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    env = DQNNavEnv(cfg, render=render)

    successes   = []
    collisions  = []
    timeouts    = []
    rewards     = []
    steps_list  = []
    per_episode = []

    for i in range(n_episodes):
        ep_seed = seed + i
        obs     = env.reset(spawn_mode=spawn_mode, seed=ep_seed)
        done    = False
        total_r = 0.0
        reason  = "timeout"

        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            total_r += reward
            if done:
                reason = info.get("reason", "timeout")

            if render:
                if not env.render(fps=30):
                    break

        successes.append(int(reason == "goal"))
        collisions.append(int(reason == "collision"))
        timeouts.append(int(reason == "timeout"))
        rewards.append(total_r)
        steps_list.append(env.steps)
        per_episode.append({
            "episode": i,
            "reward":  round(total_r, 4),
            "steps":   env.steps,
            "outcome": reason,
        })

    env.close()
    agent.epsilon = saved_eps
    agent.online_net.train()

    results = {
        "label":          extra_label or "eval",
        "n_episodes":     n_episodes,
        "seed":           seed,
        "n_rays":         cfg.N_RAYS,
        "n_obstacles":    cfg.N_OBSTACLES,
        "lidar_noise":    cfg.LIDAR_NOISE_STD,
        "step_size":      cfg.STEP_SIZE,
        "success_rate":   round(float(np.mean(successes)),   4),
        "collision_rate": round(float(np.mean(collisions)),  4),
        "timeout_rate":   round(float(np.mean(timeouts)),    4),
        "avg_reward":     round(float(np.mean(rewards)),     4),
        "std_reward":     round(float(np.std(rewards)),      4),
        "avg_steps":      round(float(np.mean(steps_list)),  2),
        "std_steps":      round(float(np.std(steps_list)),   2),
        "per_episode":    per_episode,
    }
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN Navigation — Evaluate")
    p.add_argument("--checkpoint",     required=True,
                   help="Percorso al file .pt del checkpoint")
    p.add_argument("--n_episodes",     type=int,   default=200)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--n_obstacles",    type=int,   default=None,
                   help="Override numero ostacoli (default: usa quello del checkpoint)")
    p.add_argument("--n_rays",         type=int,   default=None,
                   help="Override numero raggi LiDAR")
    p.add_argument("--lidar_noise",    type=float, default=None,
                   help="Override rumore LiDAR (std normalizzato)")
    p.add_argument("--step_size",      type=float, default=None,
                   help="Override dimensione passo agente")
    p.add_argument("--spawn_mode",     default="random", choices=["random", "fixed"])
    p.add_argument("--render",         action="store_true")
    p.add_argument("--output",         default=None,
                   help="File JSON dove salvare i risultati")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # load config from checkpoint
    cfg = DQNAgent.load_config_from_checkpoint(args.checkpoint)

    # Override runtime
    overrides = {}
    if args.n_rays      is not None: overrides["N_RAYS"]          = args.n_rays
    if args.n_obstacles is not None: overrides["N_OBSTACLES"]     = args.n_obstacles
    if args.lidar_noise is not None: overrides["LIDAR_NOISE_STD"] = args.lidar_noise
    if args.step_size   is not None: overrides["STEP_SIZE"]       = args.step_size
    if overrides:
        cfg = cfg.override(**overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = DQNAgent.load(args.checkpoint, device=device, config_override=cfg)

    print(f"[evaluate] Checkpoint : {args.checkpoint}")
    print(f"[evaluate] N_RAYS     : {cfg.N_RAYS}")
    print(f"[evaluate] N_OBSTACLES: {cfg.N_OBSTACLES}")
    print(f"[evaluate] LIDAR noise: {cfg.LIDAR_NOISE_STD}")
    print(f"[evaluate] Episodi    : {args.n_episodes}")

    results = evaluate(
        agent,
        cfg,
        n_episodes  = args.n_episodes,
        seed        = args.seed,
        render      = args.render,
        spawn_mode  = args.spawn_mode,
    )

    print("\n── Results ──────────────────────────────────────")
    for k, v in results.items():
        if k != "per_episode":
            print(f"  {k:<18} {v}")

    # Salva JSON
    out_path = args.output
    if out_path is None:
        ckpt_dir = Path(args.checkpoint).parent
        out_path = str(ckpt_dir / "eval_results.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate] Results saved in: {out_path}")
