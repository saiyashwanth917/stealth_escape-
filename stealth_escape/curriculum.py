"""
curriculum.py — Curriculum learning for Stealth Escape

Defines 4 progressive difficulty stages.
The agent auto-promotes when it hits the escape threshold
over a rolling window of recent episodes.
"""

from dataclasses import dataclass, field
from typing import List


# ── Stage definitions ────────────────────────────────────────────────────

@dataclass
class Stage:
    name:           str
    grid_size:      int
    num_guards:     int
    num_walls:      int
    fov_range:      int
    fov_angle:      int
    max_steps:      int
    promote_rate:   float   # escape rate required to advance (0–1)
    promote_window: int     # rolling episode window to evaluate
    description:    str = ""


CURRICULUM: List[Stage] = [
    Stage(
        name           = "Stage 1 — Novice",
        grid_size      = 9,
        num_guards     = 1,
        num_walls      = 5,
        fov_range      = 2,
        fov_angle      = 60,
        max_steps      = 150,
        promote_rate   = 0.50,
        promote_window = 50,
        description    = "1 guard, small map, narrow FOV. Learn to reach the exit.",
    ),
    Stage(
        name           = "Stage 2 — Apprentice",
        grid_size      = 11,
        num_guards     = 2,
        num_walls      = 10,
        fov_range      = 2,
        fov_angle      = 75,
        max_steps      = 200,
        promote_rate   = 0.40,
        promote_window = 50,
        description    = "2 guards, medium map, still narrow FOV. Route around patrols.",
    ),
    Stage(
        name           = "Stage 3 — Agent",
        grid_size      = 13,
        num_guards     = 2,
        num_walls      = 15,
        fov_range      = 3,
        fov_angle      = 90,
        max_steps      = 250,
        promote_rate   = 0.35,
        promote_window = 50,
        description    = "2 guards, larger map, wider FOV. Navigate complex patrol patterns.",
    ),
    Stage(
        name           = "Stage 4 — Ghost",
        grid_size      = 15,
        num_guards     = 3,
        num_walls      = 20,
        fov_range      = 4,
        fov_angle      = 90,
        max_steps      = 300,
        promote_rate   = 1.1,   # unreachable — final stage, never auto-promotes
        promote_window = 50,
        description    = "Full difficulty. 3 guards, dense walls. Become a ghost.",
    ),
]


# ── Tracker ──────────────────────────────────────────────────────────────

class CurriculumTracker:
    """
    Tracks episode outcomes and decides when to advance stages.

    Usage:
        tracker = CurriculumTracker()
        # after each episode:
        promoted = tracker.record(escaped=True)
        stage    = tracker.current_stage
    """

    def __init__(self, start_stage: int = 0):
        self.stage_idx    = start_stage
        self.history      = []      # list of bool (True = escaped)
        self.total_eps    = 0
        self.promotions   = []      # episode numbers where promotions happened

    @property
    def current_stage(self) -> Stage:
        return CURRICULUM[self.stage_idx]

    @property
    def is_final_stage(self) -> bool:
        return self.stage_idx >= len(CURRICULUM) - 1

    def record(self, escaped: bool) -> bool:
        """
        Record episode outcome. Returns True if a stage promotion occurred.
        """
        self.history.append(escaped)
        self.total_eps += 1

        stage  = self.current_stage
        window = self.history[-stage.promote_window:]

        if (
            len(window) >= stage.promote_window and
            not self.is_final_stage
        ):
            rate = sum(window) / len(window)
            if rate >= stage.promote_rate:
                self.stage_idx += 1
                self.history   = []   # reset window for new stage
                self.promotions.append(self.total_eps)
                return True

        return False

    def escape_rate(self) -> float:
        """Escape rate over the current rolling window."""
        stage  = self.current_stage
        window = self.history[-stage.promote_window:]
        if not window:
            return 0.0
        return sum(window) / len(window)

    def summary(self) -> str:
        stage = self.current_stage
        rate  = self.escape_rate() * 100
        return (
            f"{stage.name}  |  "
            f"escape={rate:.1f}%  |  "
            f"need={stage.promote_rate*100:.0f}%  |  "
            f"ep={self.total_eps}"
        )
