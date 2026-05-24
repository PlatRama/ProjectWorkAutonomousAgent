from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       180,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "lines.linewidth":   1.8,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
})

COLORS = plt.cm.tab10.colors

PHASE_COLORS = {
    0:   "#d4e6f1",
    1:   "#d5f5e3",
    1.5: "#fef9e7",
    2:   "#fdebd0",
}
PHASE_LABELS = {
    0:   "Phase 0 (no obstacles, fixed)",
    1:   "Phase 1 (growing obstacles, fixed)",
    1.5: "Phase 1.5 (random agent, fixed goal)",
    2:   "Phase 2 (fully random)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv(path: Path) -> List[dict]:
    import csv
    rows = []
    int_keys = {"episode", "steps", "success", "collision", "timeout",
                "global_train_steps"}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = int(v) if k in int_keys else float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _rolling(values: np.ndarray, w: int) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        s = max(0, i - w + 1)
        out[i] = values[s:i+1].mean()
    return out


def _phase_spans(rows: List[dict]) -> List[Tuple]:
    if not rows:
        return []
    spans, cur_phase, start = [], rows[0]["phase"], rows[0]["episode"]
    for r in rows[1:]:
        if r["phase"] != cur_phase:
            spans.append((cur_phase, start, r["episode"] - 1))
            cur_phase, start = r["phase"], r["episode"]
    spans.append((cur_phase, start, rows[-1]["episode"]))
    return spans


def _add_phase_shading(ax, spans, alpha: float = 0.13) -> list:
    handles = []
    seen = set()
    for phase, ep_s, ep_e in spans:
        color = PHASE_COLORS.get(phase, PHASE_COLORS.get(int(phase), "#eeeeee"))
        ax.axvspan(ep_s, ep_e, alpha=alpha, color=color, zorder=0)
        if phase not in seen:
            handles.append(mpatches.Patch(
                color=color, alpha=0.6,
                label=PHASE_LABELS.get(phase,
                      PHASE_LABELS.get(int(phase), f"Phase {phase}"))
            ))
            seen.add(phase)
    for _, ep_s, _ in spans[1:]:
        ax.axvline(ep_s, color="gray", lw=0.8, ls="--", alpha=0.4, zorder=1)
    return handles


def _save(fig, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


def _find_json(results_dir: Path, filename: str) -> Optional[Path]:
    for p in [results_dir / filename, results_dir.parent / filename]:
        if p.exists():
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_training(csv_path: Path, out_dir: Path, window: int = 100,
                  title_suffix: str = "") -> None:
    rows = _load_csv(csv_path)
    if not rows:
        print(f"[plot] {csv_path} is empty, skipping.")
        return

    eps   = np.array([r["episode"]      for r in rows])
    rew   = np.array([r["total_reward"] for r in rows])
    suc   = np.array([r["success"]      for r in rows], dtype=float)
    col   = np.array([r["collision"]    for r in rows], dtype=float)
    stp   = np.array([r["steps"]        for r in rows], dtype=float)
    eps_v = np.array([r["epsilon"]      for r in rows])
    loss  = np.array([r["avg_loss"]     for r in rows])
    qv    = np.array([r["avg_q"]        for r in rows])
    spans = _phase_spans(rows)
    label = f" — {title_suffix}" if title_suffix else ""

    # ── 1a. Learning curve ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"Learning Curve{label}", fontweight="bold")

    ax1.plot(eps, rew, alpha=0.15, color=COLORS[0], lw=0.6)
    ax1.plot(eps, _rolling(rew, window), color=COLORS[0],
             label=f"Reward (rolling {window} ep)")
    ax1.axhline(10.0, color="green", ls=":", lw=1, alpha=0.5,
                label="Max reward (+10)")
    ax1.set_ylabel("Episode Reward")
    ph_h = _add_phase_shading(ax1, spans)
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + ph_h,
               loc="upper left", fontsize=9)

    ax2.plot(eps, _rolling(suc, window), color=COLORS[2],
             label=f"Success rate (rolling {window} ep)")
    ax2.plot(eps, _rolling(col, window), color=COLORS[3],
             label=f"Collision rate (rolling {window} ep)")
    ax2.set_ylim(-0.02, 1.08)
    ax2.set_ylabel("Rate")
    ax2.set_xlabel("Episode")
    _add_phase_shading(ax2, spans)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "01_learning_curve.png")

    # ── 1b. Loss + Q-value ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"Loss and Average Q-value{label}", fontweight="bold")

    mask = loss > 0
    if mask.sum() > window:
        ax1.plot(eps[mask], loss[mask], alpha=0.2, color=COLORS[1], lw=0.6)
        ax1.plot(eps[mask], _rolling(loss[mask], window), color=COLORS[1],
                 label=f"Huber Loss (rolling {window} ep)")
    ax1.set_ylabel("Huber Loss")
    ax1.legend()
    _add_phase_shading(ax1, spans)

    ax2.plot(eps, qv, alpha=0.2, color=COLORS[4], lw=0.6)
    ax2.plot(eps, _rolling(qv, window), color=COLORS[4],
             label=f"Avg Q-value (rolling {window} ep)")
    ax2.set_ylabel("Average Q-value")
    ax2.set_xlabel("Episode")
    ax2.legend()
    _add_phase_shading(ax2, spans)

    plt.tight_layout()
    _save(fig, out_dir, "02_loss_qvalue.png")

    # ── 1c. Epsilon decay ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(eps, eps_v, color=COLORS[6], lw=1.5)
    ax.set_ylabel("ε (epsilon)")
    ax.set_xlabel("Episode")
    ax.set_title(f"Epsilon Decay (per episode){label}", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ph_h = _add_phase_shading(ax, spans)
    for phase, ep_s, _ in spans:
        if phase in (1, 1.5, 2):
            idx = np.where(eps >= ep_s)[0]
            if len(idx):
                val = eps_v[idx[0]]
                ax.annotate(f"ε→{val:.2f}", xy=(ep_s, val + 0.02),
                            fontsize=8, color="gray", ha="left")
    ax.legend(handles=ph_h, loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "03_epsilon.png")

    # ── 1d. Steps per episode ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(eps, stp, alpha=0.15, color=COLORS[7], lw=0.6)
    ax.plot(eps, _rolling(stp, window), color=COLORS[7],
            label=f"Steps (rolling {window} ep)")
    ax.set_ylabel("Steps per Episode")
    ax.set_xlabel("Episode")
    ax.set_title(f"Episode Length{label}", fontweight="bold")
    ax.legend()
    _add_phase_shading(ax, spans)
    plt.tight_layout()
    _save(fig, out_dir, "04_steps.png")

    # ── 1e. Dashboard 2×3 ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Training Dashboard{label}", fontsize=14, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    panels = [
        (fig.add_subplot(gs[0, :2]), "Success / Collision Rate", eps, suc, col),
        (fig.add_subplot(gs[0, 2]),  "Episode Reward",           eps, rew, None),
        (fig.add_subplot(gs[1, 0]),  "Average Q-value",          eps, qv,  None),
        (fig.add_subplot(gs[1, 1]),  "Huber Loss",               eps, loss, None),
        (fig.add_subplot(gs[1, 2]),  "Epsilon",                  eps, eps_v, None),
    ]
    for ax, ttl, x, y1, y2 in panels:
        if y2 is not None:
            ax.plot(x, _rolling(y1, window), color=COLORS[2], lw=1.5,
                    label="Success")
            ax.plot(x, _rolling(y2, window), color=COLORS[3], lw=1.5,
                    label="Collision")
            ax.set_ylim(-0.02, 1.05)
            ax.legend(fontsize=8)
        else:
            mask2 = (y1 > 0) if "Loss" in ttl else np.ones(len(y1), bool)
            c = COLORS[1] if "Loss" in ttl else \
                COLORS[4] if "Q-" in ttl else COLORS[0]
            if mask2.sum() > window:
                ax.plot(x[mask2], _rolling(y1[mask2], window), color=c, lw=1.5)
        ax.set_title(ttl, fontsize=10)
        ax.tick_params(labelsize=8)
        _add_phase_shading(ax, spans)

    _save(fig, out_dir, "00_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ABLATION: NUMBER OF RAYS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation_rays(json_path: Path, out_dir: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    ray_counts = sorted(int(k) for k in data.keys())
    sr  = [data[str(n)]["success_rate"]   for n in ray_counts]
    cr  = [data[str(n)]["collision_rate"] for n in ray_counts]
    tr  = [data[str(n)]["timeout_rate"]   for n in ray_counts]
    stp = [data[str(n)]["avg_steps"]      for n in ray_counts]

    x, w = np.arange(len(ray_counts)), 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation Study — Number of LiDAR Rays (Standard DQN)",
                 fontweight="bold")

    b1 = ax1.bar(x - w, sr, w, color=COLORS[2], alpha=0.85, label="Success")
    b2 = ax1.bar(x,      cr, w, color=COLORS[3], alpha=0.85, label="Collision")
    b3 = ax1.bar(x + w, tr, w, color=COLORS[8], alpha=0.85, label="Timeout")
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            if v > 0.01:
                ax1.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                         f"{v:.1%}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} rays\n(dim={3+n+8})" for n in ray_counts])
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.18)
    ax1.set_title("Outcome Rates per N_RAYS")
    ax1.legend()

    ax2b = ax2.twinx()
    l1, = ax2.plot(ray_counts, [s*100 for s in sr], "o-",
                   color=COLORS[2], lw=2, ms=8, label="Success Rate (%)")
    l2, = ax2b.plot(ray_counts, stp, "s--",
                    color=COLORS[0], lw=2, ms=8, label="Avg Steps")
    ax2.set_xlabel("Number of LiDAR Rays")
    ax2.set_ylabel("Success Rate (%)", color=COLORS[2])
    ax2b.set_ylabel("Avg Steps", color=COLORS[0])
    ax2.set_xticks(ray_counts)
    ax2.set_ylim(50, 105)
    ax2.set_title("Success Rate and Efficiency vs N_RAYS")
    ax2.legend(handles=[l1, l2], loc="lower right")

    plt.tight_layout()
    _save(fig, out_dir, "05_ablation_rays.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ABLATION: OBSTACLE DENSITY
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation_obstacles(json_path: Path, out_dir: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    n_obs = sorted(int(k) for k in data.keys())
    sr    = [data[str(n)]["success_rate"]   for n in n_obs]
    cr    = [data[str(n)]["collision_rate"] for n in n_obs]
    tr    = [data[str(n)]["timeout_rate"]   for n in n_obs]
    stp   = [data[str(n)]["avg_steps"]      for n in n_obs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Robustness to Increasing Obstacle Density", fontweight="bold")

    ax1.plot(n_obs, [s*100 for s in sr], "o-", color=COLORS[2], lw=2, ms=7,
             label="Success rate")
    ax1.plot(n_obs, [c*100 for c in cr], "s-", color=COLORS[3], lw=2, ms=7,
             label="Collision rate")
    ax1.plot(n_obs, [t*100 for t in tr], "^-", color=COLORS[8], lw=2, ms=7,
             label="Timeout rate")
    ax1.axvline(8, color="gray", ls="--", lw=1, alpha=0.5,
                label="Training density (8 obs)")
    ax1.set_xlabel("Number of Obstacles")
    ax1.set_ylabel("Rate (%)")
    ax1.set_xticks(n_obs)
    ax1.set_ylim(-2, 108)
    ax1.legend(fontsize=9)
    ax1.set_title("Outcome Rates")

    ax2.plot(n_obs, stp, "o-", color=COLORS[0], lw=2, ms=7)
    ax2.axvline(8, color="gray", ls="--", lw=1, alpha=0.5)
    ax2.set_xlabel("Number of Obstacles")
    ax2.set_ylabel("Avg Steps per Episode")
    ax2.set_xticks(n_obs)
    ax2.set_title("Path Efficiency")

    plt.tight_layout()
    _save(fig, out_dir, "07_ablation_obstacles.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ABLATION: LIDAR NOISE
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation_noise(json_path: Path, out_dir: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    sigmas = sorted(float(k) for k in data.keys())
    sr     = [data[str(s)]["success_rate"]   for s in sigmas]
    cr     = [data[str(s)]["collision_rate"] for s in sigmas]
    tr     = [data[str(s)]["timeout_rate"]   for s in sigmas]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sigmas, [s*100 for s in sr], "o-", color=COLORS[2], lw=2, ms=8,
            label="Success rate")
    ax.plot(sigmas, [c*100 for c in cr], "s-", color=COLORS[3], lw=2, ms=8,
            label="Collision rate")
    ax.plot(sigmas, [t*100 for t in tr], "^-", color=COLORS[8], lw=2, ms=8,
            label="Timeout rate")
    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.5,
               label="Training (σ=0, no noise)")
    ax.set_xlabel("LiDAR Noise Standard Deviation (σ, normalised)")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(-2, 108)
    ax.set_title("Robustness to LiDAR Noise (zero-shot)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "08_ablation_noise.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ABLATION: STEP SIZE
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation_stepsize(json_path: Path, out_dir: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    sizes = sorted(float(k) for k in data.keys())
    sr    = [data[str(s)]["success_rate"]   for s in sizes]
    stp   = [data[str(s)]["avg_steps"]      for s in sizes]
    cr    = [data[str(s)]["collision_rate"] for s in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ablation Study — Step Size", fontweight="bold")

    ax1.plot(sizes, [s*100 for s in sr], "o-", color=COLORS[2], lw=2, ms=8,
             label="Success rate")
    ax1.plot(sizes, [c*100 for c in cr], "s--", color=COLORS[3], lw=2, ms=8,
             label="Collision rate")
    ax1.axvline(8, color="gray", ls="--", lw=1, alpha=0.5,
                label="Training step (8px)")
    ax1.set_xlabel("Step Size (px)")
    ax1.set_ylabel("Rate (%)")
    ax1.set_xticks(sizes)
    ax1.set_ylim(50, 105)
    ax1.legend()
    ax1.set_title("Success/Collision Rate vs Step Size")

    ax2.plot(sizes, stp, "o-", color=COLORS[0], lw=2, ms=8)
    ax2.axvline(8, color="gray", ls="--", lw=1, alpha=0.5,
                label="Training step (8px)")
    ax2.set_xlabel("Step Size (px)")
    ax2.set_ylabel("Avg Steps per Episode")
    ax2.set_xticks(sizes)
    ax2.set_title("Path Efficiency vs Step Size")
    ax2.legend()

    plt.tight_layout()
    _save(fig, out_dir, "09_ablation_stepsize.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. OBSTACLE SHAPES (zero-shot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_shapes(json_path: Path, out_dir: Path) -> None:
    with open(json_path) as f:
        raw = json.load(f)

    shapes_data = raw.get("shapes", raw)
    shapes = list(shapes_data.keys())
    sr     = [shapes_data[s]["success_rate"]   for s in shapes]
    cr     = [shapes_data[s]["collision_rate"] for s in shapes]
    tr     = [shapes_data[s]["timeout_rate"]   for s in shapes]
    stp    = [shapes_data[s]["avg_steps"]      for s in shapes]

    labels = {
        "circle":       "Circle\n(training)",
        "rect":         "Rectangle\n(zero-shot)",
        "rotated_rect": "Rotated Rect\n(zero-shot)",
        "triangle":     "Triangle\n(zero-shot)",
        "l_shape":      "L-Shape\n(zero-shot)",
    }
    xlabels = [labels.get(s, s) for s in shapes]
    x, w = np.arange(len(shapes)), 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Zero-Shot Generalisation to Unseen Obstacle Shapes",
                 fontweight="bold")

    b1 = ax1.bar(x - w, sr, w, color=COLORS[2], alpha=0.85, label="Success")
    b2 = ax1.bar(x,      cr, w, color=COLORS[3], alpha=0.85, label="Collision")
    b3 = ax1.bar(x + w, tr, w, color=COLORS[8], alpha=0.85, label="Timeout")
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            if v > 0.01:
                ax1.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                         f"{v:.1%}", ha="center", fontsize=8, rotation=40)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=9)
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.22)
    ax1.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.4)
    ax1.set_title("Outcome Rates per Shape")
    ax1.legend()

    colors_bar = [COLORS[2]] + [COLORS[0]] * (len(shapes)-1)
    bars2 = ax2.bar(x, stp, color=colors_bar, alpha=0.85)
    for bar, v in zip(bars2, stp):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                 f"{v:.0f}", ha="center", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=9)
    ax2.set_ylabel("Avg Steps per Episode")
    ax2.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.4)
    ax2.set_title("Path Efficiency per Shape")

    plt.tight_layout()
    _save(fig, out_dir, "10_shapes_eval.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DUELING vs STANDARD DQN
# ══════════════════════════════════════════════════════════════════════════════

def plot_dueling_vs_dqn(json_path: Path, out_dir: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    labels = list(data.keys())
    sr  = [data[l]["success_rate"]   for l in labels]
    cr  = [data[l]["collision_rate"] for l in labels]
    tr  = [data[l]["timeout_rate"]   for l in labels]
    ar  = [data[l]["avg_reward"]     for l in labels]
    stp = [data[l]["avg_steps"]      for l in labels]

    xlabels = {"dueling": "Dueling DQN", "standard_dqn": "Standard DQN"}
    x, w = np.arange(len(labels)), 0.22

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Architecture Comparison: Dueling DQN vs Standard DQN",
                 fontweight="bold")

    b1 = ax1.bar(x - w, sr, w, color=COLORS[2], alpha=0.85, label="Success")
    b2 = ax1.bar(x,      cr, w, color=COLORS[3], alpha=0.85, label="Collision")
    b3 = ax1.bar(x + w, tr, w, color=COLORS[8], alpha=0.85, label="Timeout")
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            if v > 0.01:
                ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                         f"{v:.1%}", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([xlabels.get(l, l) for l in labels], fontsize=11)
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.2)
    ax1.set_title("Outcome Rates")
    ax1.legend()

    ax2.bar(x - 0.2, ar, 0.38, color=COLORS[0], alpha=0.85, label="Avg Reward")
    ax2b = ax2.twinx()
    ax2b.bar(x + 0.2, stp, 0.38, color=COLORS[6], alpha=0.85, label="Avg Steps")
    ax2.set_xticks(x)
    ax2.set_xticklabels([xlabels.get(l, l) for l in labels], fontsize=11)
    ax2.set_ylabel("Average Reward", color=COLORS[0])
    ax2b.set_ylabel("Average Steps", color=COLORS[6])
    ax2.set_title("Reward and Steps")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    _save(fig, out_dir, "11_dueling_vs_dqn.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Q-VALUE DIVERGENCE — before vs after fix
# ══════════════════════════════════════════════════════════════════════════════

def plot_q_divergence(results_dir: Path, out_dir: Path, window: int = 100) -> None:
    paths = {
        "Dueling DQN\n(diverges)": [
            results_dir / "ablation_dueling"   / "metrics.csv",
            results_dir / "ablation_rays_16r"  / "metrics.csv",
        ],
        "Standard DQN\n(stable)": [
            results_dir / "ablation_standard_dqn" / "metrics.csv",
        ],
    }

    found = {}
    for label, candidates in paths.items():
        for p in candidates:
            if p.exists():
                found[label] = p
                break

    if not found:
        print("[plot] q_divergence: no metrics.csv found, skipping.")
        return

    fig, axes = plt.subplots(1, len(found), figsize=(7 * len(found), 5),
                             squeeze=False)
    fig.suptitle("Q-value Divergence: Problem and Solution", fontweight="bold")

    for idx, (label, csv_path) in enumerate(found.items()):
        ax = axes[0][idx]
        rows = _load_csv(csv_path)
        if not rows:
            continue
        eps  = np.array([r["episode"] for r in rows])
        qv   = np.array([r["avg_q"]   for r in rows])
        spans = _phase_spans(rows)

        is_diverged = qv.max() > 100
        if is_diverged:
            cap = 200
            qv_plot = np.clip(qv, None, cap)
            ax.plot(eps, qv_plot, alpha=0.3, color=COLORS[3], lw=0.6)
            ax.plot(eps, _rolling(qv_plot, window), color=COLORS[3], lw=2)
            ax.axhline(cap, color="red", ls=":", lw=1, alpha=0.6,
                       label=f"Cap={cap} (real max={qv.max():.0f})")
            ax.set_facecolor("#fff5f5")
        else:
            ax.plot(eps, qv, alpha=0.2, color=COLORS[4], lw=0.6)
            ax.plot(eps, _rolling(qv, window), color=COLORS[4], lw=2)
            ax.set_facecolor("#f5fff5")

        _add_phase_shading(ax, spans)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Q-value")
        if is_diverged:
            ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "06_q_divergence.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY — all results together
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(results_dir: Path, out_dir: Path) -> None:
    results = {}

    jp = _find_json(results_dir, "ablation_rays.json")
    if jp:
        with open(jp) as f:
            d = json.load(f)
        for n, v in d.items():
            results[f"{n}r LiDAR"] = v["success_rate"]

    jp = _find_json(results_dir, "dueling_vs_dqn.json")
    if jp:
        with open(jp) as f:
            d = json.load(f)
        for k, v in d.items():
            lbl = "Dueling DQN" if k == "dueling" else "Standard DQN"
            results[lbl] = v["success_rate"]

    jp = _find_json(results_dir, "shapes_eval.json")
    if jp:
        with open(jp) as f:
            d = json.load(f)
        shapes_data = d.get("shapes", d)
        shape_labels = {
            "rect":         "Rectangle\n(zero-shot)",
            "rotated_rect": "Rot. Rect\n(zero-shot)",
            "triangle":     "Triangle\n(zero-shot)",
            "l_shape":      "L-Shape\n(zero-shot)",
        }
        for s, v in shapes_data.items():
            if s != "circle":
                results[shape_labels.get(s, s)] = v["success_rate"]

    if not results:
        print("[plot] summary: no data found, skipping.")
        return

    labels = list(results.keys())
    values = [results[l] * 100 for l in labels]

    colors = []
    for l in labels:
        if "LiDAR" in l:
            colors.append(COLORS[0])
        elif "Standard" in l:
            colors.append(COLORS[2])
        elif "Dueling" in l:
            colors.append(COLORS[3])
        else:
            colors.append(COLORS[8])

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 6))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, width=0.65)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Summary of Experimental Results", fontweight="bold", fontsize=14)
    ax.axhline(93, color="green", ls="--", lw=1, alpha=0.5,
               label="Best model (SR=93%, 16 rays Standard DQN)")

    legend_h = [
        mpatches.Patch(color=COLORS[0], label="N_RAYS ablation"),
        mpatches.Patch(color=COLORS[2], label="Standard DQN"),
        mpatches.Patch(color=COLORS[3], label="Dueling DQN"),
        mpatches.Patch(color=COLORS[8], label="Zero-shot shapes"),
    ]
    ax.legend(handles=legend_h + [
        plt.Line2D([0], [0], color="green", ls="--", lw=1)
    ], labels=[h.get_label() for h in legend_h] + ["Best model (SR=93%)"],
               loc="lower right", fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "12_summary_all.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="DQN Navigation — Plot Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", default="all",
                   choices=["all", "training", "ablation_rays",
                            "ablation_obstacles", "ablation_noise",
                            "ablation_stepsize", "shapes", "dueling_vs_dqn",
                            "q_divergence", "summary"],
                   help="Plot type to generate (default: all)")
    p.add_argument("--results_dir", default="results",
                   help="Directory with metrics.csv and JSON files (default: results)")
    p.add_argument("--json",     default=None,
                   help="Direct path to a specific JSON file")
    p.add_argument("--out_dir",  default=None,
                   help="Output directory for plots (default: plots/)")
    p.add_argument("--window",   type=int, default=100,
                   help="Rolling mean window size (default: 100)")
    p.add_argument("--all_experiments", action="store_true",
                   help="[deprecated] use --mode all instead")

    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir else Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot] Mode       : {args.mode}")
    print(f"[plot] Results dir: {results_dir}")
    print(f"[plot] Output dir : {out_dir}\n")

    mode = args.mode

    if mode in ("training", "all"):
        candidates = [
            results_dir / "ablation_standard_dqn" / "metrics.csv",
            results_dir / "ablation_rays_16r"     / "metrics.csv",
            results_dir / "metrics.csv",
        ]
        found_csv = next((c for c in candidates if c.exists()), None)
        if found_csv:
            print(f"[plot] Training: {found_csv}")
            plot_training(found_csv, out_dir, args.window,
                          title_suffix=found_csv.parent.name)
        else:
            print("[plot] No metrics.csv found for training, skipping.")

    if mode in ("ablation_rays", "all"):
        jp = Path(args.json) if args.json else _find_json(results_dir, "ablation_rays.json")
        if jp:
            print(f"[plot] Ablation rays: {jp}")
            plot_ablation_rays(jp, out_dir)
        else:
            print("[plot] ablation_rays.json not found, skipping.")

    if mode in ("ablation_obstacles", "all"):
        jp = Path(args.json) if args.json else _find_json(results_dir, "ablation_obstacles.json")
        if jp:
            print(f"[plot] Ablation obstacles: {jp}")
            plot_ablation_obstacles(jp, out_dir)
        else:
            print("[plot] ablation_obstacles.json not found, skipping.")

    if mode in ("ablation_noise", "all"):
        jp = Path(args.json) if args.json else _find_json(results_dir, "ablation_noise.json")
        if jp:
            print(f"[plot] Ablation noise: {jp}")
            plot_ablation_noise(jp, out_dir)
        else:
            print("[plot] ablation_noise.json not found, skipping.")

    if mode in ("ablation_stepsize", "all"):
        jp = Path(args.json) if args.json else _find_json(results_dir, "ablation_stepsize.json")
        if jp:
            print(f"[plot] Ablation stepsize: {jp}")
            plot_ablation_stepsize(jp, out_dir)
        else:
            print("[plot] ablation_stepsize.json not found, skipping.")

    if mode in ("shapes", "all"):
        jp = Path(args.json) if args.json else _find_json(results_dir, "shapes_eval.json")
        if jp:
            print(f"[plot] Shapes: {jp}")
            plot_shapes(jp, out_dir)
        else:
            print("[plot] shapes_eval.json not found, skipping.")

    if mode in ("dueling_vs_dqn", "all"):
        jp = Path(args.json) if args.json else _find_json(results_dir, "dueling_vs_dqn.json")
        if jp:
            print(f"[plot] Dueling vs DQN: {jp}")
            plot_dueling_vs_dqn(jp, out_dir)
        else:
            print("[plot] dueling_vs_dqn.json not found, skipping.")

    if mode in ("q_divergence", "all"):
        print("[plot] Q-divergence ...")
        plot_q_divergence(results_dir, out_dir, args.window)

    if mode in ("summary", "all"):
        print("[plot] Summary ...")
        plot_summary(results_dir, out_dir)

    print(f"\n[plot] ✓ All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()