import os
import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class ExtendedLoggingCallback(BaseCallback):
    """
    Callback that:
      - Computes an EMA of rewards (for logging similar to your Bayesian code).
      - Checks if current total timesteps match a 'checkpoint' and if so, saves the model.

    Args:
        model_checkpoints (list of int): The list of "timesteps" at which to save the model.
        experiment_name (str): Name of the experiment, used for folder paths.
        process_idx (int): The Julia worker process index (myid()) used to differentiate models.
        ema_weight (float): Weight for exponential moving average of reward.
        ema_start (float): Initial value for the EMA reward.
    """
    def __init__(
        self,
        model_checkpoints=None,
        experiment_name="StableBaselines",
        process_idx=1,
        ema_weight=0.0003,
        ema_start=-7.0,
        verbose=0,
    ):
        super(ExtendedLoggingCallback, self).__init__(verbose)
        self.ema_weight = ema_weight
        self.ema_reward = ema_start
        self.ema_history = []
        print(model_checkpoints)
        self.model_checkpoints = set(model_checkpoints) if model_checkpoints is not None else set()
        self.experiment_name = experiment_name
        self.process_idx = process_idx

        # Keep track of which checkpoints we've already saved,
        # so we don't save the same checkpoint multiple times.
        self.saved_checkpoints = set()

        # Create the directory for saving models if it doesn't exist
        self.save_dir = f"saved_models/{experiment_name}"
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        This is called at every environment step (for each rollout or sub-step).
        """
        # For a single-environment training scenario, you can get the instantaneous reward:
        reward = self.locals["rewards"][0]  # if only one env is used

        # Update the EMA
        if self.ema_reward is None:
            self.ema_reward = reward
        else:
            self.ema_reward = self.ema_weight * reward + (1.0 - self.ema_weight) * self.ema_reward

        self.ema_history.append(self.ema_reward)

        # If we've reached a checkpoint in terms of total timesteps, save the model.
        current_steps = self.model.num_timesteps  # or self.num_timesteps
        if current_steps in self.model_checkpoints:
            assert current_steps not in self.saved_checkpoints, "Checkpoint already saved!"
            # Mark as saved
            self.saved_checkpoints.add(current_steps)

            # Save the model
            save_path = f"{self.save_dir}/model_checkpoint_{current_steps}_{self.process_idx}.zip"
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"Saved model checkpoint at {save_path}")

        return True


def run_ppo_stable_baselines(
    env_name="Pendulum-v1",
    total_timesteps=256000,
    ema_start=-7.0,
    model_checkpoints=None,
    experiment_name="StableBaselines",
    process_idx=1,
    gamma=0.9,
    batch_size=256,
):
    """
    Run a single PPO on `env_name` for `total_timesteps`.

    Args:
        env_name (str): Gym environment name, e.g. "Pendulum-v1".
        total_timesteps (int): Total training timesteps.
        ema_start (float): Initial value for the reward EMA in the callback.
        gamma (float): Discount factor for PPO.
        batch_size (int): Batch size for PPO.
        model_checkpoints (list of int): Timesteps at which we should save the model.
        experiment_name (str): Used for naming saved model files/folders.
        process_idx (int): The process ID (from Julia's myid()).

    Returns:
        list of float: The EMA reward history (one entry per environment step).
    """
    env = gym.make(env_name)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        gamma=gamma,
        batch_size=batch_size,
    )

    callback = ExtendedLoggingCallback(
        model_checkpoints=model_checkpoints,
        experiment_name=experiment_name,
        process_idx=process_idx,
        ema_weight=0.0003,
        ema_start=ema_start,
        verbose=0
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Return the entire EMA reward history for plotting in Julia
    return callback.ema_history


# ---------------------------------------------------
# EVALUATION LOGIC (analogous to Bayesian code)
# ---------------------------------------------------

def evaluate_agent_once_stable_baselines(env_name, model, num_steps=10000):
    """
    Evaluate a loaded Stable Baselines PPO model for `num_steps` in `env_name`.
    Return the average reward per step.
    """
    env = gym.make(env_name)
    obs, info = env.reset()

    total_reward = 0.0
    for _ in range(num_steps):
        # Predict action, here deterministic=True for evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward
        if done or truncated:
            obs, info = env.reset()

    return total_reward / num_steps


def evaluate_fixed_model_stable_baselines(env_name, experiment_name, model_checkpoints, process_idx=1, steps=10000):
    """
    Loads each available checkpointed model (based on `model_checkpoints`) for
    the given `experiment_name` and `process_idx`.
    Evaluates each model for `steps` steps with no training,
    returning a list of average reward-per-step for each checkpoint in the order given.

    This mirrors the Bayesian A2C evaluation logic, but uses PPO's `.zip` files.
    """
    avg_rewards = []

    for checkpoint in model_checkpoints:
        # Construct the path where the checkpoint might have been saved
        path = f"saved_models/{experiment_name}/model_checkpoint_{checkpoint}_{process_idx}.zip"
        if os.path.isfile(path):
            # Load the PPO model
            model = PPO.load(path)
            # Evaluate
            avg_reward_for_this_checkpoint = evaluate_agent_once_stable_baselines(env_name, model, num_steps=steps)
            avg_rewards.append(avg_reward_for_this_checkpoint)
        else:
            # If file not found, we do not push a zero; we simply skip.
            # (Or push zero if you prefer.)
            pass

    print(f"[Process {process_idx}] Evaluated checkpoints for {experiment_name}: {avg_rewards}")
    return avg_rewards
