from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import torch

from config import Config
from agent import DQNAgent
from evaluate import evaluate
from evaluate_shapes import run_all_shapes
from train import train


# ── Helper: quick training ────────────────────────────────────────────────────

def quick_train(cfg: Config) -> str:
    """
    Trains an agent with the provided config and returns the path to the
    best checkpoint (best.pt if exists, otherwise final.pt as fallback).
    """
    ckpt_dir = Path(cfg.CHECKPOINT_DIR) / cfg.exp_name
    best_pt  = str(ckpt_dir / "best.pt")
    final_pt = str(ckpt_dir / "final.pt")

    if Path(best_pt).exists():
        print(f"  [skip] {cfg.exp_name} — checkpoint already exists")
        return best_pt

    print(f"  [train] {cfg.exp_name} ...")
    train(cfg)

    if Path(best_pt).exists():
        return best_pt
    if Path(final_pt).exists():
        print(f"  [warn] {cfg.exp_name} — best.pt not saved (SR=0?), using final.pt")
        return final_pt
    raise FileNotFoundError(
        f"No checkpoint found for {cfg.exp_name} after training.\n"
        f"Searched: {best_pt}, {final_pt}"
    )


# ── Ablation: number of rays ──────────────────────────────────────────────────

def ablation_rays(
    base_cfg:          Config,
    ray_counts:        List[int]      = [4, 8, 16, 32],
    eval_episodes:     int            = 200,
    do_train:          bool           = True,
    base_checkpoint:   Optional[str]  = None,
) -> dict:
    """
    For each number of rays:
      1. Train an agent with config scaled for N_RAYS
      2. Evaluate on its own test set (random circles)

    Per-ray configuration:
      - Lower LR for more rays (larger gradients)
      - Longer Phase 0 for more rays (larger state, slower learning)
      - Phase 1.5 transition (agent_random): reduces distribution shift
      - Lower TAU for more rays (more stable target network)
    """
    # Parameters scaled by state dimensionality
    LR_BY_RAYS   = {4: 5e-4, 8: 3e-4, 16: 2e-4, 32: 1e-4}
    P0_BY_RAYS   = {4: 500,  8: 600,  16: 800,  32: 1000}
    P15_BY_RAYS  = {4: 400,  8: 500,  16: 600,  32: 700}
    TAU_BY_RAYS  = {4: 0.005, 8: 0.003, 16: 0.002, 32: 0.001}

    results = {}

    for n in ray_counts:
        exp_name = f"ablation_rays_{n}r"

        # Create config specific for this ray count
        override_kwargs = dict(
            exp_name         = exp_name,
            N_RAYS           = n,
            LR               = LR_BY_RAYS.get(n, 5e-4),
            PHASE0_EPISODES  = P0_BY_RAYS.get(n, 500),
            PHASE15_EPISODES  = P15_BY_RAYS.get(n, 500),
            TAU              = TAU_BY_RAYS.get(n, 0.005),
            DUELING          = base_cfg.DUELING
        )

        cfg = base_cfg.override(**override_kwargs)

        if do_train:
            ckpt = quick_train(cfg)
        else:
            assert base_checkpoint, "Need --base_checkpoint if --no_train"
            ckpt = base_checkpoint

        print(f"  [eval] {exp_name} ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_cfg = DQNAgent.load_config_from_checkpoint(ckpt)
        agent    = DQNAgent.load(ckpt, device=device)

        r = evaluate(agent, eval_cfg, n_episodes=eval_episodes,
                     seed=base_cfg.EVAL_SEED, extra_label=exp_name)
        results[str(n)] = r
        print(f"    N_RAYS={n}: SR={r['success_rate']:.2%}")

    return results


# ── Ablation: obstacle density ────────────────────────────────────────────────

def ablation_obstacles(
    base_checkpoint:  str,
    obstacle_counts:  List[int]  = [2, 4, 6, 8, 10, 12],
    eval_episodes:    int        = 200,
) -> dict:
    """
    Evaluates a pre-trained agent with an increasing number of obstacles.
    No re-training — measures robustness to unseen densities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = DQNAgent.load_config_from_checkpoint(base_checkpoint)
    agent  = DQNAgent.load(base_checkpoint, device=device)

    results = {}
    for n in obstacle_counts:
        cfg_n = cfg.override(N_OBSTACLES=n)
        r = evaluate(agent, cfg_n, n_episodes=eval_episodes,
                     seed=cfg.EVAL_SEED, extra_label=f"obs_{n}")
        results[str(n)] = r
        print(f"    N_OBS={n:2d}: SR={r['success_rate']:.2%}  "
              f"CR={r['collision_rate']:.2%}")

    return results


# ── Ablation: LiDAR noise ─────────────────────────────────────────────────────

def ablation_noise(
    base_checkpoint:  str,
    noise_levels:     List[float] = [0.0, 0.02, 0.05, 0.10, 0.20],
    eval_episodes:    int         = 200,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = DQNAgent.load_config_from_checkpoint(base_checkpoint)
    agent  = DQNAgent.load(base_checkpoint, device=device)

    results = {}
    for sigma in noise_levels:
        cfg_n = cfg.override(LIDAR_NOISE_STD=sigma)
        r = evaluate(agent, cfg_n, n_episodes=eval_episodes,
                     seed=cfg.EVAL_SEED, extra_label=f"noise_{sigma}")
        results[str(sigma)] = r
        print(f"    noise σ={sigma:.2f}: SR={r['success_rate']:.2%}")

    return results


# ── Ablation: step size ───────────────────────────────────────────────────────

def ablation_stepsize(
    base_checkpoint:  str,
    step_sizes:       List[float] = [4.0, 6.0, 8.0, 12.0, 16.0],
    eval_episodes:    int         = 200,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = DQNAgent.load_config_from_checkpoint(base_checkpoint)
    agent  = DQNAgent.load(base_checkpoint, device=device)

    results = {}
    for s in step_sizes:
        cfg_s = cfg.override(STEP_SIZE=s)
        r = evaluate(agent, cfg_s, n_episodes=eval_episodes,
                     seed=cfg.EVAL_SEED, extra_label=f"step_{s}")
        results[str(s)] = r
        print(f"    step_size={s:.1f}: SR={r['success_rate']:.2%}")

    return results


# ── Ablation: Dueling DQN vs Standard DQN ────────────────────────────────────

def ablation_dueling(
    base_cfg:       Config,
    eval_episodes:  int  = 200,
    do_train:       bool = True,
) -> dict:
    results = {}
    for dueling in [True, False]:
        label = "dueling" if dueling else "standard_dqn"
        cfg   = base_cfg.override(exp_name=f"ablation_{label}", DUELING=dueling)
        if do_train:
            ckpt = quick_train(cfg)
        else:
            ckpt = str(Path(cfg.CHECKPOINT_DIR) / cfg.exp_name / "best.pt")

        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_cfg = DQNAgent.load_config_from_checkpoint(ckpt)
        agent    = DQNAgent.load(ckpt, device=device)
        r = evaluate(agent, eval_cfg, n_episodes=eval_episodes,
                     seed=base_cfg.EVAL_SEED, extra_label=label)
        results[label] = r
        print(f"    {label}: SR={r['success_rate']:.2%}  Q_avg≈{r.get('avg_q_eval', 'N/A')}")

    return results


# ── Main Runner ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    base_cfg = Config(
        SEED             = args.seed,
        N_RAYS           = 16,
        N_OBSTACLES      = 8,
        DUELING          = False,
        PHASE0_EPISODES  = args.p0,
        PHASE1_EPISODES  = args.p1,
        PHASE2_EPISODES  = args.p2,
        RESULTS_DIR      = args.results_dir,
        CHECKPOINT_DIR   = args.checkpoint_dir,
        EVAL_SEED        = args.eval_seed,
        exp_name         = "base",
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    do_train = not args.no_train

    exp = args.exp

    # ── 1. Rays ablation ──────────────────────────────────────────────────────
    if exp in ("ablation_rays", "all"):
        print("\n═══ ABLATION: N_RAYS ═══")
        r = ablation_rays(
            base_cfg         = base_cfg,
            ray_counts       = [4, 8, 16, 32],
            eval_episodes    = args.eval_episodes,
            do_train         = do_train,
            base_checkpoint  = args.base_checkpoint,
        )
        all_results["ablation_rays"] = r
        _save(results_dir / "ablation_rays.json", r)

        # Auto-detect base_checkpoint for subsequent experiments.
        # Use the 16-ray model as default (most representative).
        # If it doesn't exist, try 8r, then 4r.
        if args.base_checkpoint is None:
            for n_fallback in [16, 8, 4]:
                ckpt_base = Path(args.checkpoint_dir) / f"ablation_rays_{n_fallback}r"
                for fname in ("best.pt", "final.pt"):
                    candidate = ckpt_base / fname
                    if candidate.exists():
                        args.base_checkpoint = str(candidate)
                        print(f"\n[run_experiments] Base checkpoint auto-detected: {args.base_checkpoint}")
                        break
                if args.base_checkpoint is not None:
                    break
            else:
                print(f"\n[run_experiments] ⚠️  No checkpoint found after ablation_rays.")
                print(f"   Pass --base_checkpoint for future experiments.")

    # ── 2. Obstacle density ───────────────────────────────────────────────────
    if exp in ("ablation_obstacles", "all"):
        print("\n═══ ABLATION: N_OBSTACLES ═══")
        ckpt = _get_base_ckpt(args, base_cfg, "ablation_obstacles")
        if ckpt:
            r = ablation_obstacles(ckpt, eval_episodes=args.eval_episodes)
            all_results["ablation_obstacles"] = r
            _save(results_dir / "ablation_obstacles.json", r)

    # ── 3. LiDAR noise ────────────────────────────────────────────────────────
    if exp in ("ablation_noise", "all"):
        print("\n═══ ABLATION: LIDAR NOISE ═══")
        ckpt = _get_base_ckpt(args, base_cfg, "ablation_noise")
        if ckpt:
            r = ablation_noise(ckpt, eval_episodes=args.eval_episodes)
            all_results["ablation_noise"] = r
            _save(results_dir / "ablation_noise.json", r)

    # ── 4. Step size ──────────────────────────────────────────────────────────
    if exp in ("ablation_stepsize", "all"):
        print("\n═══ ABLATION: STEP SIZE ═══")
        ckpt = _get_base_ckpt(args, base_cfg, "ablation_stepsize")
        if ckpt:
            r = ablation_stepsize(ckpt, eval_episodes=args.eval_episodes)
            all_results["ablation_stepsize"] = r
            _save(results_dir / "ablation_stepsize.json", r)

    # ── 5. Obstacle shapes ────────────────────────────────────────────────────
    if exp in ("shapes", "all"):
        print("\n═══ SHAPE EVALUATION ═══")
        ckpt = _get_base_ckpt(args, base_cfg, "shapes")
        if ckpt:
            r = run_all_shapes(ckpt, n_episodes=args.eval_episodes,
                               seed=args.eval_seed,
                               output_path=str(results_dir / "shapes_eval.json"))
            all_results["shapes"] = r

    # ── 6. Dueling vs Standard ────────────────────────────────────────────────
    if exp in ("dueling_vs_dqn", "all"):
        print("\n═══ DUELING DQN vs STANDARD DQN ═══")
        r = ablation_dueling(base_cfg, eval_episodes=args.eval_episodes,
                             do_train=do_train)
        all_results["dueling_vs_dqn"] = r
        _save(results_dir / "dueling_vs_dqn.json", r)

    # ── Save everything ───────────────────────────────────────────────────────
    _save(results_dir / "all_experiments.json", all_results)
    print(f"\n[run_experiments] All results saved in: {results_dir}")


# ── Utility ───────────────────────────────────────────────────────────────────

def _save(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        # Strip per_episode data for brevity
        import copy
        light = copy.deepcopy(data)
        _strip_per_episode(light)
        json.dump(light, f, indent=2)


def _strip_per_episode(obj):
    if isinstance(obj, dict):
        obj.pop("per_episode", None)
        for v in obj.values():
            _strip_per_episode(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_per_episode(item)


def _best_pt(cfg: Config) -> str:
    return str(Path(cfg.CHECKPOINT_DIR) / cfg.exp_name / "best.pt")


def _best_exists(cfg: Config) -> bool:
    return Path(_best_pt(cfg)).exists()


def _resolve_checkpoint(path: str) -> Optional[str]:
    """
    Resolves a checkpoint path that can be:
      - a direct .pt file          → returns it
      - a directory                → looks for best.pt, then final.pt inside
    Returns None if not found.
    """
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return str(p)
    # It's a directory: search best.pt then final.pt
    for fname in ("best.pt", "final.pt"):
        candidate = p / fname
        if candidate.exists():
            print(f"  [checkpoint] '{path}' is a directory → using {candidate}")
            return str(candidate)
    return None


def _get_base_ckpt(
    args: argparse.Namespace,
    base_cfg: Config,
    label: str,
) -> Optional[str]:
    """
    Returns the available base checkpoint.

    Priority:
      1. --base_checkpoint (file .pt or directory)
      2. checkpoints/base/best.pt  (default run)
      3. Auto-discovery: ablation_rays_16r → 8r → 4r → 32r
    """
    # 1. Explicit checkpoint from CLI
    if args.base_checkpoint:
        resolved = _resolve_checkpoint(args.base_checkpoint)
        if resolved:
            return resolved
        print(f"\n[warn] --base_checkpoint '{args.base_checkpoint}' not found.")

    # 2. Fallback to default run
    fallback = _best_pt(base_cfg)
    if Path(fallback).exists():
        return fallback

    # 3. Auto-discovery in already available ablation_rays runs
    ckpt_dir = Path(base_cfg.CHECKPOINT_DIR)
    for n_rays in (16, 8, 4, 32):
        candidate_dir = ckpt_dir / f"ablation_rays_{n_rays}r"
        resolved = _resolve_checkpoint(str(candidate_dir))
        if resolved:
            print(f"\n[auto] Base checkpoint automatically detected: {resolved}")
            return resolved

    print(f"\n[skip] {label} — no base checkpoint found.")
    print(f"  Train first with:  python src/run_experiments.py --exp ablation_rays")
    print(f"  Or specify:        --base_checkpoint <folder or file.pt>")
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DQN Navigation — Experiment Suite")
    p.add_argument("--exp", default="all",
                   choices=["ablation_rays", "ablation_obstacles",
                            "ablation_noise", "ablation_stepsize",
                            "shapes", "dueling_vs_dqn", "all"])
    p.add_argument("--base_checkpoint", default=None,
                   help="Already trained base checkpoint (avoids re-training)")
    p.add_argument("--no_train",        action="store_true",
                   help="Do not train, use only existing checkpoints")
    p.add_argument("--eval_episodes",   type=int, default=200)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--eval_seed",       type=int, default=42)
    p.add_argument("--p0",              type=int, default=500)
    p.add_argument("--p1",              type=int, default=1500)
    p.add_argument("--p2",              type=int, default=3000)
    p.add_argument("--results_dir",     default="results")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    args = p.parse_args()

    run(args)