"""
train.py — Train PPO agent on StealthEnv  (FIXED v3)
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

from env.stealth_env import StealthEnv
from curriculum import CurriculumTracker, CURRICULUM
from config import (
    PPO_TIMESTEPS, PPO_LEARNING_RATE, PPO_N_STEPS,
    PPO_BATCH_SIZE, PPO_N_EPOCHS, PPO_GAMMA,
    PPO_CLIP_RANGE, PPO_ENT_COEF,
    MODEL_SAVE_PATH, LOG_PATH,
)

N_ENVS               = 8        # was 4 → more parallel experience
CURRICULUM_TIMESTEPS = 800_000  # was 200k → enough to progress through stages


# ── Callbacks ────────────────────────────────────────────────────────────

class StealthLogCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.catches   = 0
        self.escapes   = 0
        self.episodes  = 0
        self._last_printed = 0

    def _on_step(self):
        for info, done in zip(self.locals.get("infos",[]), self.locals.get("dones",[])):
            if done:
                self.episodes += 1
                if info.get("caught"):       self.catches += 1
                if info.get("reached_exit"): self.escapes += 1
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])

        if self.episodes > 0 and self.episodes % 100 == 0 and self.episodes != self._last_printed:
            self._last_printed = self.episodes
            avg_r       = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            escape_rate = self.escapes / self.episodes * 100
            catch_rate  = self.catches / self.episodes * 100
            print(
                f"  [ep {self.episodes:6d}] "
                f"avg_r={avg_r:+.2f}  "
                f"escape={escape_rate:.1f}%  "
                f"caught={catch_rate:.1f}%"
            )
        return True


class CurriculumCallback(BaseCallback):
    def __init__(self, tracker, raw_envs, verbose=1):
        super().__init__(verbose)
        self.tracker  = tracker
        self.raw_envs = raw_envs

    def _on_step(self):
        for info, done in zip(self.locals.get("infos",[]), self.locals.get("dones",[])):
            if not done:
                continue
            promoted = self.tracker.record(info.get("reached_exit", False))
            if promoted:
                stage = self.tracker.current_stage
                for env in self.raw_envs:
                    env.set_stage(stage)
                print("\n" + "="*55)
                print(f"  PROMOTED  →  {stage.name}")
                print(f"  {stage.description}")
                print("="*55 + "\n")
        return True


def get_raw_envs(vec_env):
    raw = []
    for e in vec_env.envs:
        inner = e
        while hasattr(inner, "env"):
            inner = inner.env
        raw.append(inner)
    return raw


# ── Training ─────────────────────────────────────────────────────────────

def train(use_curriculum=True):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or ".", exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    total_timesteps = CURRICULUM_TIMESTEPS if use_curriculum else PPO_TIMESTEPS

    print("=" * 55)
    print("  Stealth Escape — PPO Training  (FIXED v3)")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"  Curriculum      : {'ON' if use_curriculum else 'OFF'}")
    print("=" * 55 + "\n")

    if use_curriculum:
        tracker     = CurriculumTracker(start_stage=0)
        start_stage = tracker.current_stage
        print(f"  Starting : {start_stage.name}")
        print(f"  {start_stage.description}\n")
        vec_env  = make_vec_env(lambda: Monitor(StealthEnv(stage=start_stage)), n_envs=N_ENVS)
        raw_envs = get_raw_envs(vec_env)
        eval_env = Monitor(StealthEnv(stage=start_stage))
    else:
        vec_env  = make_vec_env(lambda: Monitor(StealthEnv()), n_envs=N_ENVS)
        eval_env = Monitor(StealthEnv())

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = PPO_LEARNING_RATE,
        n_steps         = PPO_N_STEPS,
        batch_size      = PPO_BATCH_SIZE,
        n_epochs        = PPO_N_EPOCHS,
        gamma           = PPO_GAMMA,
        clip_range      = PPO_CLIP_RANGE,
        ent_coef        = PPO_ENT_COEF,
        verbose         = 0,
        tensorboard_log = LOG_PATH,
        policy_kwargs   = dict(net_arch=[256, 256]),  # bigger network for complex task
    )

    callbacks = [
        StealthLogCallback(verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path = MODEL_SAVE_PATH + "_best",
            log_path             = LOG_PATH,
            eval_freq            = 20_000,
            n_eval_episodes      = 20,
            deterministic        = True,
            verbose              = 0,
        ),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = MODEL_SAVE_PATH + "_checkpoints/",
            name_prefix = "stealth_ppo",
            verbose     = 0,
        ),
    ]
    if use_curriculum:
        callbacks.append(CurriculumCallback(tracker, raw_envs))

    print("Training started... Ctrl+C to stop early.\n")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    model.save(MODEL_SAVE_PATH)
    print(f"\n  Model saved → {MODEL_SAVE_PATH}.zip")
    print(f"  Run: python evaluate.py")
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-curriculum", action="store_true")
    args = parser.parse_args()
    train(use_curriculum=not args.no_curriculum)