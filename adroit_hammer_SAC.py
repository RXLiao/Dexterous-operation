import gymnasium as gym
import gymnasium_robotics 

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


# =========================================================
# 1) Env factory for SB3 
# =========================================================

def make_adroit_hammer_env(
    seed: int = 0,
    render: bool = False,
):
    def _init():
        render_mode = "human" if render else None
        env = gym.make("AdroitHandHammer-v1", render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


# =========================================================
# 2) Train SAC 
# =========================================================

from stable_baselines3.common.callbacks import EvalCallback


class EvalCallbackWithBestInfo(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_timesteps = None

    def _on_step(self) -> bool:
        result = super()._on_step()

        # 当发现新的 best model 时
        if self.best_mean_reward == self.last_mean_reward:
            self.best_timesteps = self.num_timesteps
            print(
                f"[BEST MODEL UPDATED] "
                f"timesteps = {self.best_timesteps}, "
                f"mean_reward = {self.best_mean_reward:.3f}"
            )

        return result


def train_sac(
    total_timesteps: int = 3_000_000,
    n_envs: int = 1,
    seed: int = 0,
):
    env_fns = [
        make_adroit_hammer_env(seed=seed + i, render=False)
        for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    eval_env = make_adroit_hammer_env(seed=123, render=False)()

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path="./checkpoints_raw/",
        name_prefix="raw_ckpt",
    )

    # eval_callback = EvalCallback(
    #     eval_env=eval_env,
    #     best_model_save_path="./best_model_raw/",
    #     log_path="./eval_logs_raw/",
    #     eval_freq=20_000,
    #     deterministic=True,
    #     render=False,
    # )
    eval_callback = EvalCallbackWithBestInfo(
        eval_env=eval_env,
        best_model_save_path="./best_model_raw/",
        log_path="./eval_logs_raw/",
        eval_freq=20_000,
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tb_raw/",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    print("\n========== TRAINING FINISHED ==========")
    print(f"Best mean reward: {eval_callback.best_mean_reward}")
    print(f"Best model found at timesteps: {eval_callback.best_timesteps}")

    model.save("raw_hammer_final")
    return model


if __name__ == "__main__":
    train_sac(
        total_timesteps=2_800_000,
        n_envs=1,
        seed=0,
    )