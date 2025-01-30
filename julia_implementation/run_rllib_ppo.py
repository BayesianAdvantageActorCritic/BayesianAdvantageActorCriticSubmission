# run_rllib_ppo.py
import faulthandler
faulthandler.disable()

# Overwrite faulthandler.enable with a no-op to prevent Ray from calling it successfully
def no_op(*args, **kwargs):
    pass

faulthandler.enable = no_op

import os
import ray
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym


def run_rllib_ppo(
    env_name="Pendulum-v1",
    total_timesteps=256000,
    ema_start=-7.0,
    model_checkpoints=None,
    experiment_name="RLlibExperiment",
    process_idx=1,
    gamma=0.9,
    batch_size=256,
):
    """
    Train PPO using Ray RLlib on the given `env_name` for `total_timesteps` steps.
    Returns a list of EMA (exponential moving average) rewards (one entry per iteration),
    and *also* saves model checkpoints analogous to the Bayesian A2C code.

    Parameters:
        env_name (str): Gym environment name
        total_timesteps (int): total number of environment steps
        ema_start (float): initial value for the EMA
        gamma (float): discount factor
        batch_size (int): batch size for training
        model_checkpoints (list): e.g. [10, 20, 50, 100, 200, 500, 1000]
        experiment_name (str): Name of experiment (used in path to save checkpoints)
        process_idx (int): The Julia process index (used in checkpoint file name)

    Returns:
        ema_rewards (list of float): The per-environment-step EMA of rewards.
    """

    # Initialize Ray
    ray.init(ignore_reinit_error=False, log_to_driver=False, configure_logging=False)

    # Configure PPO
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name)
        .training(gamma=gamma, train_batch_size=batch_size)
        .framework("torch")
        .learners(num_learners=1)
    )
    algo = config.build()

    # Prepare to save checkpoints
    if model_checkpoints is None:
        model_checkpoints = []
    model_checkpoints = sorted(model_checkpoints)  # ensure it’s sorted
    checkpoint_index = 0

    # Create directory for saving if needed
    save_dir = os.path.join("saved_models", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    timesteps_so_far = 0
    ema_rewards = []
    ema_value = ema_start

    # We'll define an EMA weight similar to your stable-baselines callback
    ema_weight = 0.0003

    while timesteps_so_far < total_timesteps:
        result = algo.train()

        # RLlib’s result dict changes depending on version; this is approximate:
        # Check result["env_runners"]["episode_return_mean"] for nan
        if "episode_return_mean" in result["env_runners"] and not np.isnan(result["env_runners"]["episode_return_mean"]):
            current_mean_reward = result["env_runners"]["episode_return_mean"]
            mean_reward_per_step = current_mean_reward / result["env_runners"]["episode_len_mean"]
        else:
            mean_reward_per_step = ema_start

        # Update timesteps
        timesteps_so_far = result["num_env_steps_sampled_lifetime"]

        # Update EMA for each environment step in this iteration
        steps_this_iter = result["info"]["num_env_steps_sampled"]
        assert timesteps_so_far == steps_this_iter
        for _ in range(256):
            ema_value = ema_weight * mean_reward_per_step + (1 - ema_weight) * ema_value
            ema_rewards.append(ema_value)

        assert len(ema_rewards) == timesteps_so_far, "EMA length mismatch"

        # ---- CHECKPOINT SAVING ----
        # If we passed any checkpoint(s), save the model.
        while checkpoint_index < len(model_checkpoints) and timesteps_so_far >= model_checkpoints[checkpoint_index]:
            c = model_checkpoints[checkpoint_index]
            checkpoint_path = os.path.join(save_dir, f"model_checkpoint_{c}_{process_idx}.pth")
            checkpoint_path = f"{os.path.abspath(checkpoint_path)}"
            algo.save(checkpoint_path)
            checkpoint_index += 1

    # Shutdown Ray
    ray.shutdown()

    return ema_rewards


# ---------------------------------------------------------------------
# EVALUATION LOGIC
# ---------------------------------------------------------------------

def evaluate_rllib_ppo(env_name, experiment_name, model_checkpoints, process_idx=1, num_eval_steps=10000):
    """
    Loads each available checkpoint for this process, evaluates it in a fresh environment
    for `num_eval_steps` steps (with no training), and returns a list of average rewards
    per step (one entry per found checkpoint).

    This is directly analogous to the Bayesian A2C's `evaluate_fixed_model`.
    """
    ray.init(ignore_reinit_error=False, log_to_driver=False, configure_logging=False)

    # We need the same config used for training (minus big training settings).
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name)
        .framework("torch")
        .learners(num_learners=1)
    )
    algo = config.build()

    save_dir = os.path.join("saved_models", experiment_name)
    avg_rewards = []

    for checkpoint in model_checkpoints:
        checkpoint_path = os.path.join(save_dir, f"model_checkpoint_{checkpoint}_{process_idx}.pth")
        checkpoint_path = f"{os.path.abspath(checkpoint_path)}"
        if os.path.isfile(checkpoint_path) or os.path.isdir(checkpoint_path):
            # Restore the policy weights
            algo.restore(checkpoint_path)
            # Evaluate
            avg_r = evaluate_agent_once(env_name, algo, num_eval_steps)
            avg_rewards.append(avg_r)
        else:
            print(f"Checkpoint file not found: {checkpoint_path}")

    ray.shutdown()

    print(f"The vector of average rewards for process {process_idx} is: {avg_rewards}")
    return avg_rewards


def evaluate_agent_once(env_name, algo, num_steps=10000):
    """
    Runs `num_steps` steps in `env_name` with the loaded RLlib PPO agent.
    Returns the average reward per step.
    """
    # return -1.0  # Placeholder for the actual evaluation logic
    env = gym.make(env_name)
    obs, info = env.reset()
    total_reward = 0.0

    for _ in range(num_steps):
        action = algo.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            obs, info = env.reset()

    return total_reward / num_steps


if __name__ == "__main__":
    # Example usage: training (no eval). Just for a local test:
    #    python run_rllib_ppo.py
    # If you want to see checkpoint saving, pass some example checkpoints below.

    ema_rewards = run_rllib_ppo(
        env_name="Pendulum-v1",
        total_timesteps=256000,
        ema_start=-7.0,
        gamma=0.9,
        batch_size=256,
        model_checkpoints=[10_000, 20_000, 50_000, 100_000, 200_000],
        experiment_name="Test_RLlib",
        process_idx=1
    )
    print("Training done, length of EMA:", len(ema_rewards))

    # Example usage of evaluation:
    # Evaluate for those same checkpoints
    eval_rewards = evaluate_rllib_ppo(
        env_name="Pendulum-v1",
        experiment_name="Test_RLlib",
        model_checkpoints=[10_000, 20_000, 50_000, 100_000, 200_000],
        process_idx=1,
        num_eval_steps=2000  # just for a short run
    )
    print("Eval results:", eval_rewards)
    print("Length of eval results:", len(eval_rewards))
