# ENTER YOUR PATHS HERE
ENV["PYTHON"] = "ENTER_THE_PATH_TO_YOUR_ENVIRONMENT_SPECIFIC_PYTHON_EXECUTABLE_HERE"
ENV["PYTHONPATH"] = "PATH_TO_LOCAL_DIRECTORY_OF_REPO"

include("../julia_implementation/bayesian_a2c_fg.jl")
using PyCall
gym = pyimport("gymnasium")

# Define functions on all processes
function get_fast_configuration()
    return Dict(
        "number_environments" => 256,
        "num_batches" => 1000,
        "gamma" => 0.9,
        "learning_rate" => 0.05,
        "beta_ema_critic" => 1.0,
        "beta_ema_actor" => 1.0,
        "number_factor_graph_iterations" => 3,
        "probability_random_reset" => 0.005,
        "action_std_expansion_factor" => 4.0,
        "default_beta_actor" => 0.05,
        "default_beta_critic" => 0.4,
        "action_percentage_clipping" => 0.9,
        "pre_execution" => true,
        "sliding_reward_start" => -7.0,   # default, can be overridden
        "num_subsequent_steps" => 1,
        "clip_factor" => 1.25,
        "normalize_advantages" => true,
        "lr_decrease" => 0.999,
        "use_normal_actor" => false,
        "use_normal_critic" => false,
        "use_ivon_for_normal" => false,
        "use_pretrained_critic" => false,
        "use_independent_networks" => false,
        "ignore_std_output" => false,
        "learning_rate_normal" => 0.003,
        "learning_rate_ivon" => 0.1,
        "name" => "BayesianA2C",
    )
end

# Function to create the Pendulum-v1 environment. You can set a different environment here.
function create_pendulum_env()
    return gym.make("Pendulum-v1")
end

# Create the agent with the environment creation function
agent = BayesianA2CAgent(create_pendulum_env; config=get_fast_configuration(), experiment_name="YOUR_EXPERIMENT_NAME_FOR_SINGLE_AGENT_TRAINING")

# Train the agent
sliding_rewards, all_rewards = learn(agent; use_wandb=true, save_best_parameters=true)

println("Sizes and types of sliding_rewards and all_rewards:")
println(size(sliding_rewards), typeof(sliding_rewards))
println(size(all_rewards), typeof(all_rewards))
