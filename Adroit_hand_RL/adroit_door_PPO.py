import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


# =========================================================
# 1) Env factory for SB3
# =========================================================

def make_adroit_door_env(
    seed: int = 0,
    render: bool = False,
):
    def _init():
        render_mode = "human" if render else None
        env = gym.make("AdroitHandDoor-v1", render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


# =========================================================
# 2) Train PPO on original observation
# =========================================================

def linear_schedule_with_floor(
    initial_lr,
    min_lr=1e-5,
):
    def f(progress_remaining):
        return max(min_lr, initial_lr * progress_remaining)

    return f


def train_ppo(
    total_timesteps: int = 500_000,
    n_envs: int = 1,
):
    env_fns = [
        make_adroit_door_env(seed=i, render=False)
        for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    eval_env = make_adroit_door_env(seed=123, render=False)()

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path="./checkpoints_raw/",
        name_prefix="ppo_ckpt",
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="./best_model_raw/",
        log_path="./eval_logs_raw/",
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tb_raw/",
        learning_rate=linear_schedule_with_floor(3e-4, 1e-5),
        n_steps=2048,
        batch_size=64,
        ent_coef=0.02,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    model.save("raw_ppo_adroit_door_final")
    return model


if __name__ == "__main__":
    train_ppo(
        total_timesteps=5_000_000,
        n_envs=1,
    )