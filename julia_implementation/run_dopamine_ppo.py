import os
import pickle
import gymnasium as gym
import numpy as np
from dopamine.jax.agents.ppo import ppo_agent
import jax
import gin
from flax import serialization

def _save_dopamine_agent(agent, save_path):
    param_bytes = serialization.to_bytes(agent.network_params)
    with open(save_path, "wb") as f:
        f.write(param_bytes)

def _load_dopamine_agent(agent, load_path):
    with open(load_path, "rb") as f:
        param_bytes = f.read()
    agent.network_params = serialization.from_bytes(agent.network_params, param_bytes)
    return agent

def run_ppo_dopamine(
    env_name="Pendulum-v1",
    total_timesteps=256000,
    ema_start=-7.0,
    model_checkpoints=None,
    experiment_name="DopaminePPO",
    process_idx=1,
    gamma=0.9,
    batch_size=256,
    ema_weight=0.0003,
):
    """
    Train a Dopamine PPO agent for 'total_timesteps' steps in `env_name`.
    - Maintains an EMA of the reward after each step (stored in ema_list).
    - Saves the agent at the timesteps specified in `model_checkpoints`.
      Example: if model_checkpoints=[10000, 20000], it saves the agent at steps 10k & 20k.

    Returns:
        ema_list (list of float): The EMA reward history (length = total_timesteps).
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    act_limits = (env.action_space.low, env.action_space.high)

    agent = ppo_agent.PPOAgent(
        action_shape=act_dim,
        observation_shape=obs_dim,
        action_limits=act_limits,
        gamma=gamma,
        batch_size=batch_size,
        # max_capacity=1_000_000,  # Optional, depending on your environment
    )
    agent.eval_mode = False  # We want to train, not just eval

    # Create directory for saving models
    save_dir = os.path.join("saved_models", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Convert the list to a set for faster membership checks
    checkpoints_set = set(model_checkpoints) if model_checkpoints is not None else set()

    ema = ema_start
    ema_list = []

    obs, info = env.reset()
    action = agent.begin_episode(obs)  # The very first action
    for step in range(1, total_timesteps + 1):
        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Update the EMA
        ema = ema_weight * reward + (1.0 - ema_weight) * ema
        ema_list.append(ema)

        if not done and not truncated:
            # Continue episode
            action = agent.step(reward, obs)
        else:
            # End of episode
            # If you need to call agent.end_episode(reward), do it here:
            # agent.end_episode(reward)  # (If your PPO agent requires it)
            obs, info = env.reset()
            action = agent.begin_episode(obs)

        # ----- Checkpoint saving -----
        # If this step is a checkpoint, save the agent
        if step in checkpoints_set:
            save_path = os.path.join(save_dir, f"model_checkpoint_{step}_{process_idx}.pkl")
            _save_dopamine_agent(agent, save_path)
            # You might also print a small message to confirm:
            # print(f"[Process {process_idx}] Saved model at step={step} -> {save_path}")

    env.close()
    return ema_list

# -------------------------------------------------------------------------
# EVALUATION LOGIC (analogous to Bayesian or Stable Baselines)
# -------------------------------------------------------------------------

def evaluate_agent_once_dopamine(env_name, agent, num_steps=10000):
    """
    Evaluate a loaded Dopamine PPO agent for `num_steps` in `env_name`.
    Returns the average reward per step.
    """
    env = gym.make(env_name)
    obs, info = env.reset()
    total_reward = 0.0

    # If the agent has an `eval_mode` flag, set it here:
    agent.eval_mode = True

    # Begin the first episode
    action = agent.begin_episode(obs)
    for _ in range(num_steps):
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if not done and not truncated:
            action = agent.step(reward, obs)
        else:
            # End of episode
            # agent.end_episode(reward)  # If your agent needs end-of-episode
            obs, info = env.reset()
            action = agent.begin_episode(obs)

    env.close()
    return total_reward / num_steps

def evaluate_fixed_model_dopamine(env_name, experiment_name, model_checkpoints, process_idx=1, steps=10000):
    """
    Loads each available checkpointed Dopamine PPO agent for the given
    `experiment_name` and `process_idx`, evaluates for `steps` steps,
    returns a list of average reward-per-step for each checkpoint.

    If a checkpoint file is missing, it's skipped. The results are
    returned in the same order as `model_checkpoints`.
    """
    avg_rewards = []
    save_dir = os.path.join("saved_models", experiment_name)

    env = gym.make(env_name)

    for checkpoint in model_checkpoints:
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape
        act_limits = (env.action_space.low, env.action_space.high)

        agent = ppo_agent.PPOAgent(
            action_shape=act_dim,
            observation_shape=obs_dim,
            action_limits=act_limits,
            gamma=0.9,
            batch_size=256,
        )

        checkpoint_path = os.path.join(save_dir, f"model_checkpoint_{checkpoint}_{process_idx}.pkl")
        if os.path.isfile(checkpoint_path):
            # Load agent
            agent = _load_dopamine_agent(agent, checkpoint_path)
            avg_reward = evaluate_agent_once_dopamine(env_name, agent, num_steps=steps)
            avg_rewards.append(avg_reward)
        else:
            # If the checkpoint file isn't found, skip or (optionally) append something else
            # avg_rewards.append(0.0)  # optionally
            pass

    print(f"[Process {process_idx}] Evaluated checkpoints for {experiment_name}: {avg_rewards}")
    return avg_rewards


if __name__ == "__main__":
    ema_history = run_ppo_dopamine(
        env_name="Pendulum-v1",
        total_timesteps=256000,
        ema_start=-7.0,
        gamma=0.9,
        batch_size=256,
        ema_weight=0.0003,
        model_checkpoints=[10000, 20000, 50000],
        experiment_name="DopaminePPO",
        process_idx=1,
    )

    # Evaluate the saved models
    mylist = evaluate_fixed_model_dopamine(
        env_name="Pendulum-v1",
        experiment_name="DopaminePPO",
        model_checkpoints=[10000, 20000, 50000],
        process_idx=1,
        steps=10000,
    )
    print(mylist)
