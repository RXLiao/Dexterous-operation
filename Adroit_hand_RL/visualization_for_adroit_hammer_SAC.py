import time
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401  # register Adroit envs

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


def make_adroit_hammer_env_raw(render: bool = False):
    """
    与训练脚本一致：
      gym.make("AdroitHandHammer-v1") -> Monitor
    """
    render_mode = "human" if render else None
    env = gym.make("AdroitHandHammer-v1", render_mode=render_mode)
    env = Monitor(env)
    return env


def evaluate_sac_raw(
    model_path: str,
    episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    fps: int = 60,
    base_seed: int = 89,
):
    print(f"Loading SAC model from: {model_path}")
    model = SAC.load(model_path, device="cpu")

    print("Creating env ...")
    env = make_adroit_hammer_env_raw(render=render)

    dt = 1.0 / max(1, fps)

    returns = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=base_seed + ep)
        terminated = False
        truncated = False
        ret = 0.0
        length = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            ret += float(reward)
            length += 1

            if render:
                time.sleep(dt)

        returns.append(ret)
        lengths.append(length)
        print(f"Episode {ep + 1}: return={ret:.3f}, length={length}")

    env.close()

    returns = np.asarray(returns, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int32)

    print("\nSummary:")
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    print(f"  mean return = {returns.mean():.3f} ± {std:.3f}")
    print(f"  mean length = {lengths.mean():.1f}")

    return returns, lengths


if __name__ == "__main__":
    # 二选一：
    # 1) 最终模型（训练里 model.save("raw_hammer_final") 会生成 raw_hammer_final.zip）
    # model_path = "raw_hammer_final.zip"

    # 2) 或者某个 checkpoint：
    model_path = "./checkpoints_raw/raw_ckpt_2640000_steps.zip"

    evaluate_sac_raw(
        model_path=model_path,
        episodes=30,
        deterministic=True,
        render=False,   # True 则 human 渲染并按 fps sleep
        fps=60,
        base_seed=89,   # 与训练 seed 无需相同；只影响评估 reset 的随机性
    )