"""
evaluate.py — Load a trained PPO model and watch it play StealthEnv
"""

import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from env.stealth_env import StealthEnv
from config import MODEL_SAVE_PATH


def evaluate(model_path=None, episodes=10, render=True, deterministic=True):
    path = model_path or MODEL_SAVE_PATH

    print(f"Loading model from: {path}")
    model = PPO.load(path)

    render_mode = "human" if render else None
    env = StealthEnv(render_mode=render_mode)

    stats = {"escapes": 0, "caught": 0, "timeout": 0, "rewards": []}

    for ep in range(episodes):
        obs, _ = env.reset()
        done    = False
        total_r = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_r += reward
            done = terminated or truncated

            if render:
                time.sleep(0.05)

        stats["rewards"].append(total_r)
        if info.get("reached_exit"):
            stats["escapes"] += 1
            result = "ESCAPED"
        elif info.get("caught"):
            stats["caught"] += 1
            result = "CAUGHT"
        else:
            stats["timeout"] += 1
            result = "TIMEOUT"

        print(f"  Episode {ep+1:2d}: {result}  reward={total_r:+.2f}  steps={info['step']}")

    env.close()

    print("\n-- Summary ----------------------------------")
    print(f"  Escaped  : {stats['escapes']}/{episodes}  ({stats['escapes']/episodes*100:.0f}%)")
    print(f"  Caught   : {stats['caught']}/{episodes}  ({stats['caught']/episodes*100:.0f}%)")
    print(f"  Timeout  : {stats['timeout']}/{episodes}  ({stats['timeout']/episodes*100:.0f}%)")
    print(f"  Avg reward: {np.mean(stats['rewards']):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stealth Escape PPO Agent")
    parser.add_argument("--model",      type=str,  default=None, help="Path to model .zip")
    parser.add_argument("--episodes",   type=int,  default=10,   help="Number of eval episodes")
    parser.add_argument("--no-render",  action="store_true",      help="Disable pygame rendering")
    parser.add_argument("--stochastic", action="store_true",      help="Use stochastic policy")
    args = parser.parse_args()

    evaluate(
        model_path    = args.model,
        episodes      = args.episodes,
        render        = not args.no_render,
        deterministic = not args.stochastic,
    )