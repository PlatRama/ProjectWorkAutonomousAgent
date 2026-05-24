from __future__ import annotations
import copy
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Config:
    # ── Environment ────────────────────────────────────────────────────────────
    CANVAS_SIZE: int          = 600
    AGENT_RADIUS: int         = 10
    GOAL_RADIUS: int          = 28
    GOAL_SUCCESS_RADIUS: int  = 38      # GOAL_RADIUS + AGENT_RADIUS
    STEP_SIZE: float          = 8.0
    MAX_STEPS: int            = 400

    # ── LiDAR ──────────────────────────────────────────────────────────────────
    N_RAYS: int               = 16
    RAY_MAX_DIST: float       = 200.0
    LIDAR_NOISE_STD: float    = 0.0    # std of Gaussian noise on normalised readings [0,1]

    # ── Obstacles ──────────────────────────────────────────────────────────────
    N_OBSTACLES: int          = 8
    OBS_RADIUS_MIN: int       = 20
    OBS_RADIUS_MAX: int       = 45
    OBSTACLE_CLEARANCE: float = 20.0   # extra clearance around goal at spawn time

    # ── Actions ────────────────────────────────────────────────────────────────
    N_ACTIONS: int            = 8

    # ── Network ────────────────────────────────────────────────────────────────
    HIDDEN1: int              = 256
    HIDDEN2: int              = 128
    DUELING: bool             = True   # Dueling DQN architecture

    # ── Training ───────────────────────────────────────────────────────────────
    BATCH_SIZE: int           = 128
    REPLAY_BUFFER_SIZE: int   = 50_000
    GAMMA: float              = 0.99
    LR: float                 = 5e-4
    #LR: float                 = 2e-4
    LR_PHASE2: float = 1e-4
    GRAD_CLIP: float          = 0.5
    GRAD_CLIP_PHASE2: float = 0.3  # più conservativo in fase 2
    MIN_REPLAY_SIZE: int      = 1_000  # start training only after this many transitions

    # ── Target network ─────────────────────────────────────────────────────────
    TARGET_UPDATE_MODE: str   = "soft" # "soft" | "hard"
    TARGET_UPDATE_FREQ: int   = 500    # steps between hard updates
    TAU: float                = 0.005  # Polyak coefficient for soft update

    # ── Epsilon-greedy ─────────────────────────────────────────────────────────
    EPS_START: float          = 1.0
    EPS_END: float            = 0.10
    EPS_DECAY: float          = 0.997  # multiplicative decay *per episode*
    EPS_DECAY_PHASE2: float = 0.9990

    # ── Curriculum ─────────────────────────────────────────────────────────────
    PHASE0_EPISODES: int      = 800    # no obstacles, fixed positions
    PHASE1_EPISODES: int      = 1_500  # growing obstacles, fixed positions
    PHASE15_EPISODES: int     = 600
    PHASE2_EPISODES: int      = 3_000  # 8 random obstacles
    EPS_RESTART_PHASE2: float = 0.5    # epsilon reset when phase 2 begins
    CLEAR_BUFFER_PHASE2: bool = True

    # ── Reward shaping ─────────────────────────────────────────────────────────
    REWARD_GOAL: float        = 10.0
    REWARD_COLLISION: float   = -1.0
    REWARD_STEP: float        = -0.003
    PROXIMITY_ZONE: float     = 100.0  # px — zone where proximity shaping activates
    PROXIMITY_SCALE: float    = 0.08   # weight for delta-distance shaping

    # ── Evaluation ─────────────────────────────────────────────────────────────
    EVAL_EPISODES: int        = 200
    EVAL_SEED: int            = 42

    # ── I/O ────────────────────────────────────────────────────────────────────
    SEED: int                 = 42
    #SEED: int                 = 123
    SAVE_EVERY: int           = 250    # checkpoint every N episodes
    RESULTS_DIR: str          = "results"
    CHECKPOINT_DIR: str       = "checkpoints"
    exp_name: str             = "run_default"   # experiment name (used for subfolder)

    # ── Computed properties ────────────────────────────────────────────────────
    @property
    def STATE_DIM(self) -> int:
        """3 (goal info) + N_RAYS (LiDAR) + N_ACTIONS (last-action one-hot)."""
        return 3 + self.N_RAYS + self.N_ACTIONS

    @property
    def TOTAL_EPISODES(self) -> int:
        return self.PHASE0_EPISODES + self.PHASE1_EPISODES + self.PHASE15_EPISODES + self.PHASE2_EPISODES

    # ── Serialisation ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["STATE_DIM"]      = self.STATE_DIM
        d["TOTAL_EPISODES"] = self.TOTAL_EPISODES
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        for key in ("STATE_DIM", "TOTAL_EPISODES"):
            d.pop(key, None)
        valid = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)

    def override(self, **kwargs) -> "Config":
        c = copy.deepcopy(self)
        for k, v in kwargs.items():
            if k not in self.__dataclass_fields__:
                raise ValueError(f"Unknown Config field: '{k}'")
            setattr(c, k, v)
        return c

    def __repr__(self) -> str:
        lines = ["Config("]
        for k, v in asdict(self).items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
