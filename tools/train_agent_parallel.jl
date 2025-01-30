# ENTER YOUR PATHS HERE
ENV["PYTHON"] = "ENTER_THE_PATH_TO_YOUR_ENVIRONMENT_SPECIFIC_PYTHON_EXECUTABLE_HERE"
ENV["PYTHONPATH"] = "PATH_TO_LOCAL_DIRECTORY_OF_REPO"

using PyCall
using Distributed
using Statistics

# Number of processes to add
num_processes = 3

# Add worker processes
addprocs(num_processes)

# Include necessary files and packages on all processes
@everywhere include("../julia_implementation/bayesian_a2c_fg.jl")
@everywhere using PyCall
@everywhere using Statistics

# Define functions on all processes
@everywhere function get_fast_configuration()
    Dict(
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
@everywhere function create_and_train_agent()
    gym = pyimport("gymnasium")

    # Function to create the Pendulum-v1 environment
    function create_pendulum_env()
        return gym.make("Pendulum-v1")
    end

    agent = BayesianA2CAgent(create_pendulum_env; config=get_fast_configuration(), experiment_name="YOUR_EXPERIMENT_NAME_FOR_PARALLEL_TRAINING")
    sliding_rewards, all_rewards_other_scale = learn(agent; process_idx=myid(), use_wandb=true, save_best_parameters=true)
    return all_rewards_other_scale
end

# Run the function on each process and collect futures
futures = Vector{Future}(undef, num_processes)
@sync begin
    for i in 1:num_processes
        futures[i] = @spawnat i create_and_train_agent()
    end
end

# Fetch the sliding rewards from all processes
sliding_rewards_list = fetch.(futures)

# Plotting function
using Plots
using Statistics

function plot_combined_sliding_rewards(sliding_rewards_list::Vector{Vector{Float64}})
    current_time = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    file_path_pdf = "figures/Learning_Curve_Combined_$(current_time).pdf"

    # Convert to matrix
    rewards_matrix = hcat(sliding_rewards_list...)

    # Compute statistics
    min_rewards = Statistics.minimum(rewards_matrix, dims=2)[:]
    max_rewards = Statistics.maximum(rewards_matrix, dims=2)[:]
    median_rewards = Statistics.median(rewards_matrix, dims=2)[:]
    x = 1:length(median_rewards)

    # Plot median with ribbon between min and max
    p = plot(
        x, median_rewards,
        ribbon = (median_rewards .- min_rewards, max_rewards .- median_rewards),
        xlabel = "Learned Samples",
        ylabel = "Sliding Reward",
        title = "Sliding Reward over Time",
        label = "Median",
        color = :blue,
        legend = :bottomright,
    )
    savefig(p, file_path_pdf)
end

# Call the plotting function
plot_combined_sliding_rewards(sliding_rewards_list)
