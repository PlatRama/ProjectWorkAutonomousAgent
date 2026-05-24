from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import Config
from agent import DQNAgent
from environment import DQNNavEnv
from obstacle_shapes import make_random_obstacles

def eval_shape(
    agent:      DQNAgent,
    cfg:        Config,
    shape_type: str,
    n_episodes: int = 200,
    seed:       int = 42,
) -> dict:
    agent.online_net.eval()
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    env = DQNNavEnv(cfg, render=False)

    successes, collisions, timeouts = [], [], []
    rewards, steps_list = [], []
    per_episode = []

    for i in range(n_episodes):
        ep_seed = seed + i
        obstacles = make_random_obstacles(
            shape_type=shape_type,
            n=cfg.N_OBSTACLES,
            canvas=cfg.CANVAS_SIZE,
            seed=ep_seed,
        )
        obs = env.reset(spawn_mode="random", obstacles=obstacles)
        done     = False
        total_r  = 0.0
        reason   = "timeout"

        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            total_r += reward
            if done:
                reason = info.get("reason", "timeout")

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

    return {
        "shape":          shape_type,
        "n_episodes":     n_episodes,
        "success_rate":   round(float(np.mean(successes)),   4),
        "collision_rate": round(float(np.mean(collisions)),  4),
        "timeout_rate":   round(float(np.mean(timeouts)),    4),
        "avg_reward":     round(float(np.mean(rewards)),     4),
        "std_reward":     round(float(np.std(rewards)),      4),
        "avg_steps":      round(float(np.mean(steps_list)),  2),
        "std_steps":      round(float(np.std(steps_list)),   2),
        "per_episode":    per_episode,
    }


SHAPES = ["circle", "rect", "rotated_rect", "triangle", "l_shape"]


def run_all_shapes(
    checkpoint_path: str,
    n_episodes:      int = 200,
    seed:            int = 42,
    output_path:     Optional[str] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = DQNAgent.load_config_from_checkpoint(checkpoint_path)
    agent  = DQNAgent.load(checkpoint_path, device=device)

    print(f"[evaluate_shapes] Checkpoint: {checkpoint_path}")
    print(f"[evaluate_shapes] N_RAYS={cfg.N_RAYS}  N_OBSTACLES={cfg.N_OBSTACLES}")
    print(f"[evaluate_shapes] Episodi per forma: {n_episodes}\n")

    all_results = {}

    for shape in SHAPES:
        print(f"  ► Valutazione forma: {shape} ...", end=" ", flush=True)

        if shape == "circle":
            from evaluate import evaluate
            result = evaluate(
                agent, cfg,
                n_episodes=n_episodes,
                seed=seed,
                spawn_mode="random",
                extra_label="circle",
            )
            result["shape"] = "circle"
        else:
            result = eval_shape(agent, cfg, shape, n_episodes=n_episodes, seed=seed)

        all_results[shape] = result
        print(f"SR={result['success_rate']:.2%}  "
              f"CR={result['collision_rate']:.2%}  "
              f"TR={result['timeout_rate']:.2%}")

    # Tabella riassuntiva
    print("\n── Summary ──────────────────────────────────────")
    print(f"{'Shape':<16} {'SR':>6} {'CR':>6} {'TR':>6} {'Avg steps':>10}")
    print("─" * 46)
    for shape in SHAPES:
        r = all_results[shape]
        print(f"{shape:<16} {r['success_rate']:>6.2%} {r['collision_rate']:>6.2%} "
              f"{r['timeout_rate']:>6.2%} {r['avg_steps']:>10.1f}")

    output = {
        "checkpoint":  checkpoint_path,
        "n_episodes":  n_episodes,
        "seed":        seed,
        "n_rays":      cfg.N_RAYS,
        "n_obstacles": cfg.N_OBSTACLES,
        "shapes":      all_results,
    }

    if output_path is None:
        ckpt_dir    = Path(checkpoint_path).parent
        output_path = str(ckpt_dir / "shapes_eval.json")

    with open(output_path, "w") as f:
        light = json.loads(json.dumps(output))
        for s in light["shapes"].values():
            s.pop("per_episode", None)
        json.dump(light, f, indent=2)

    print(f"\n[evaluate_shapes] Results saved in: {output_path}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zero-shot evaluation for different shapes")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--n_episodes",  type=int, default=200)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output",      default=None)
    p.add_argument("--shapes",      nargs="+", default=SHAPES,
                   choices=SHAPES)
    args = p.parse_args()

    run_all_shapes(
        checkpoint_path=args.checkpoint,
        n_episodes=args.n_episodes,
        seed=args.seed,
        output_path=args.output,
    )
