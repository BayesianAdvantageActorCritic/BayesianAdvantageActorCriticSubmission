# ENTER YOUR PATHS HERE
ENV["PYTHON"] = "ENTER_THE_PATH_TO_YOUR_ENVIRONMENT_SPECIFIC_PYTHON_EXECUTABLE_HERE"
ENV["PYTHONPATH"] = "PATH_TO_LOCAL_DIRECTORY_OF_REPO"

using Distributed
using PyCall
using Statistics
using Plots
using Dates

# Number of processes
num_processes = 5
addprocs(num_processes)

@everywhere using PyCall
@everywhere using Statistics

# -------------------------------------------------------------------------
# 1) Bayesian A2C code definitions
# -------------------------------------------------------------------------
@everywhere include("julia_implementation/bayesian_a2c_fg.jl")
@everywhere include("julia_implementation/fixed_model_evaluator.jl")

@everywhere function get_fast_configuration(env_config::Dict)
    # Define default configuration
    default_conf = Dict(
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
    return merge(default_conf, env_config)
end

@everywhere function create_and_train_bayesian_agent(env_name::String, env_config::Dict, experiment_name::String, model_checkpoints::Vector{Int})
    gym = pyimport("gymnasium")

    function create_env()
        return gym.make(env_name)
    end

    agent = BayesianA2CAgent(create_env; config = env_config, experiment_name=experiment_name)
    sliding_rewards, all_rewards_other_scale = learn(agent; process_idx = myid(), use_wandb=true, model_checkpoints=model_checkpoints)

    return all_rewards_other_scale
end

# -------------------------------------------------------------------------
# 2) Python PPO code definitions
# -------------------------------------------------------------------------
@everywhere begin
    run_ppo_module = pyimport("julia_implementation.run_ppo_pendulum")
end

@everywhere begin
    run_rllib_module = pyimport("julia_implementation.run_rllib_ppo")
end

@everywhere begin
    run_dopamine_ppo_module = pyimport("julia_implementation.run_dopamine_ppo")
end

# -------------------------------------------------------------------------
# 3) Plot functions
# -------------------------------------------------------------------------
function compute_min_med_max(mat)
    return (minimum(mat, dims=2)[:],
            Statistics.median(mat, dims=2)[:],
            maximum(mat, dims=2)[:])
end

const color_map = Dict(
    "Bayesian Message Passing Actor/Critic"  => :blue,
    "Bayesian A2C"                           => :blue,
    "Bayesian With Normalization (standard)" => :blue,
    "BA2C with LR decrease 0.999 (standard)" => :blue,
    "BA2C LR=0.05 (standard)"                => :blue,
    "Bayes beta actor 0.05 (standard)"       => :blue,
    "BA2C LR=0.05 (standard)"                => :blue,
    "PyTorch NN Actor/Critic"                => :green,
    "IVON Actor/Critic"                      => :cyan,
    "Bayesian MP Actor/Critic ignore std"    => :brown,
    "SB3 PPO"                                => :red,
    "RLlib PPO"                              => :yellow,
    "Dopamine PPO"                           => :purple,
    "Actor: Bayesian MP, Critic: PyTorch NN" => :pink,
    "Actor: PyTorch NN, Critic: Bayesian MP" => :orange,
)

const fallback_colors = [
    :red, :green, :orange, :purple, :pink, :cyan, :magenta, 
    :brown, :coral, :navy, :olive, :gold, :lime, :teal, :maroon,
    :orchid, :salmon, :slateblue, :forestgreen, :hotpink, :chocolate,
    :darkturquoise, :darkviolet, :dodgerblue, :lawngreen
]

function get_color_for_label(label::String)
    c = get(color_map, label, nothing)
    return c === nothing ? rand(fallback_colors) : c
end

### NEW / MODIFIED: A more general function that plots multiple runs in one figure
function plot_multiple_runs(all_rewards::Vector{Tuple{String,Vector{Vector{Float64}}}}, env_name::String, experiment_name::String)
    # 'all_rewards' is a Vector of tuples: ( legend_label, [ [Float64], [Float64], ... ] )
    p = nothing
    x_range = nothing

    for (label, reward_lists) in all_rewards
        mat = hcat(reward_lists...)
        r_min, r_median, r_max = compute_min_med_max(mat)

        if x_range === nothing
            x_range = 1:length(r_median)
            # Initialize a plot
            p = plot(
                x_range,
                r_median,
                ribbon = (r_median .- r_min, r_max .- r_median),
                xlabel = "Timesteps",
                ylabel = "Sliding Reward",
                title  = experiment_name,
                label  = label,
                legend = :bottomright,
                color  = get_color_for_label(label)  # modified
            )
        else
            plot!(
                p,
                x_range,
                r_median,
                ribbon = (r_median .- r_min, r_max .- r_median),
                label = label,
                color  = get_color_for_label(label)  # modified
            )
        end
    end

    if p !== nothing
        filename = replace(experiment_name, " " => "_")
        savefig(p, "figures/$(filename)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")).pdf")
    end
end

"""
    plot_multiple_runs_evaluation(all_evals, env_name, experiment_name)

Takes a vector of (label, [ [Float64], [Float64], ... ]) 
where each inner vector-of-Float64 is the evaluation result from one process 
(i.e. the output of evaluate_fixed_model). 

Plots min/median/max as a ribbon plot, with x-axis = the 
checkpoint indices [10, 20, 50, ...], y-axis = average reward. 
Saves as a PDF with '_evaluation_' in the filename.
"""
function plot_multiple_runs_evaluation(all_evals::Vector{Tuple{String,Vector{Vector{Float64}}}}, 
                                       env_name::String, 
                                       experiment_name::String,
                                       model_checkpoints::Vector{Int})
    # Determine the length of the individual evaluations of all_evals.
    # Assert that all evaluations have the same length.
    # Then, take the first elements of model_checkpoints as the x-axis, a part that has as many elements as the evaluations.
    num_realized_evals = length(all_evals[1][2][1])
    println("Number of realized evaluations: $num_realized_evals")
    for (label, runs_of_checkpoints) in all_evals
        for run in runs_of_checkpoints
            @assert length(run) == num_realized_evals "For label $label, run has length $(length(run)), expected $num_realized_evals"
        end
    end
    model_checkpoints = model_checkpoints[1:num_realized_evals]
    
    p = nothing
    for (label, runs_of_checkpoints) in all_evals
        # runs_of_checkpoints is a Vector of Vectors: each sub-vector is the 
        # average-reward-over-checkpoints for one process.

        # Turn that into a matrix of shape (#processes, #checkpoints), 
        # or (#checkpoints, #processes). We typically want 
        # each row = a single checkpoint index, each column = a different process.
        # So let's do:
        mat = reduce(hcat, runs_of_checkpoints)  # dimension = (#checkpoints, #processes)

        # We want min/med/max across processes => dimension=2.
        r_min, r_med, r_max = compute_min_med_max(mat)

        if p === nothing
            p = plot(
                model_checkpoints,
                r_med,
                ribbon = (r_med .- r_min, r_max .- r_med),
                xlabel = "Checkpoint",
                ylabel = "Avg Reward per Step",
                title  = experiment_name * " (Evaluator)",
                label  = label,
                seriestype = :line,
                legend = :bottomright,
                color  = get_color_for_label(label)  # modified
            )
        else
            plot!(
                p,
                model_checkpoints,
                r_med,
                ribbon = (r_med .- r_min, r_max .- r_med),
                label = label,
                seriestype = :line,
                color      = get_color_for_label(label)  # modified
            )
        end
    end

    if p !== nothing
        filename = replace(experiment_name, " " => "_")
        savefig(p, "figures/evaluation_$(filename)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")).pdf")
    end
end

# -------------------------------------------------------------------------
# 4) Single function to run an algorithm, with minimal changes
# -------------------------------------------------------------------------
### NEW / MODIFIED:  We unify how we call the training
@everywhere function train_algorithm(algorithm_name::String, env_name::String, env_config::Dict, experiment_name::String, model_checkpoints::Vector{Int})
    # We preserve your logic for timesteps from the pipeline:
    # Suppose we read from env_config or just do a default
    total_timesteps_ppo = env_config["num_batches"] * env_config["number_environments"]

    # We also pick up the sliding reward from the config if present, else default
    sliding_reward_start = get(env_config, "sliding_reward_start", -7.0)

    # In experiment_name, replace spaces with underscores
    experiment_name = replace(experiment_name, " " => "_")

    if algorithm_name == "Bayesian A2C"
        return create_and_train_bayesian_agent(env_name, deepcopy(env_config), experiment_name, deepcopy(model_checkpoints))
    elseif algorithm_name == "Stable Baselines 3"
        return run_ppo_module.run_ppo_stable_baselines(env_name, total_timesteps_ppo, sliding_reward_start, model_checkpoints, experiment_name, myid())
    elseif algorithm_name == "RLlib PPO"
        return run_rllib_module.run_rllib_ppo(env_name, total_timesteps_ppo, sliding_reward_start, model_checkpoints, experiment_name, myid())
    elseif algorithm_name == "Dopamine PPO"
        return run_dopamine_ppo_module.run_ppo_dopamine(env_name, total_timesteps_ppo, sliding_reward_start, model_checkpoints, experiment_name, myid())
    else
        error("Unknown algorithm name: $algorithm_name")
    end
end


@everywhere function evaluate_algorithm(algo_name::String, env_name::String, experiment_name::String, model_checkpoints::Vector{Int}; process_id::Int=myid())
    # In experiment_name, replace spaces with underscores
    experiment_name = replace(experiment_name, " " => "_")

    if algo_name == "Bayesian A2C"
        return evaluate_fixed_model(env_name, experiment_name, deepcopy(model_checkpoints))
    elseif algo_name == "Stable Baselines 3"
        return run_ppo_module.evaluate_fixed_model_stable_baselines(env_name, experiment_name, model_checkpoints, process_id)
    elseif algo_name == "RLlib PPO"
        return run_rllib_module.evaluate_rllib_ppo(env_name, experiment_name, model_checkpoints, process_id)
    elseif algo_name == "Dopamine PPO"
        return run_dopamine_ppo_module.evaluate_fixed_model_dopamine(env_name, experiment_name, model_checkpoints, process_id)
    else
        error("Unknown algorithm name: $algo_name")
    end
end

@everywhere function config_to_string(conf::Dict{String,Any})
    # Convert the Dict to a sorted list of key=>value pairs, then join them.
    # This ensures the same order each time for the same contents.
    keys_sorted = sort(collect(keys(conf)))
    serialization = join([string(k) * "=" * string(conf[k]) for k in keys_sorted], ", ")
    println("Serialized configuration: $serialization")
    return serialization
end

# The key will be (algo_name, env_name, experiment_name, config_string).
# The value will be the Vector-of-Vector-of-Float64 returned from evaluate_algorithm (one vector per process).
const EVALUATION_CACHE = Dict{Tuple{String,String,String}, Vector{Vector{Float64}}}()


function generate_standard_configuration_different_environment(env_name::String, sliding_reward_start::Float64)
    return (
        env_name,
        [
            ("Bayesian A2C", Dict("sliding_reward_start"=>sliding_reward_start), "Bayesian A2C"),
            ("Stable Baselines 3", Dict("sliding_reward_start"=>sliding_reward_start), "SB3 PPO"),
            ("RLlib PPO", Dict("sliding_reward_start"=>sliding_reward_start), "RLlib PPO"),
            ("Dopamine PPO", Dict("sliding_reward_start"=>sliding_reward_start), "Dopamine PPO"),
        ],
        "Bayesian A2C vs Library PPO on $env_name"
    )
end

function generate_standard_std_ignore_comparison(env_name::String, sliding_reward_start::Float64)
    return (
        env_name,
        [
            ("Bayesian A2C", Dict("use_normal_actor"=>false, "use_normal_critic"=>false, "use_ivon_for_normal"=>false, "sliding_reward_start"=>sliding_reward_start), "Bayesian Message Passing Actor/Critic"),
            ("Bayesian A2C", Dict("use_normal_actor"=>false, "use_normal_critic"=>false, "use_ivon_for_normal"=>false, "ignore_std_output"=>true, "sliding_reward_start"=>sliding_reward_start), "Bayesian MP Actor/Critic ignore std"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "sliding_reward_start"=>sliding_reward_start), "PyTorch NN Actor/Critic"),
        ],
        "Bayesian A2C fixed std comparison on $env_name"
    )
end

function generate_standard_ivon_comparison(env_name::String, sliding_reward_start::Float64)
    return (
        env_name,
        [
            ("Bayesian A2C", Dict("use_normal_actor"=>false, "use_normal_critic"=>false, "use_ivon_for_normal"=>false, "sliding_reward_start"=>sliding_reward_start), "Bayesian Message Passing Actor/Critic"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "sliding_reward_start"=>sliding_reward_start), "IVON Actor/Critic"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "sliding_reward_start"=>sliding_reward_start), "PyTorch NN Actor/Critic"),
        ],
        "Bayesian A2C IVON comparison on $env_name"
    )
end

function generate_ivon_learning_rate_comparison(env_name::String, sliding_reward_start::Float64, num_batches::Int=2500)
    return (
        env_name,
        [
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>0.0001, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=0.0001"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>0.0003, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=0.0003"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>0.001, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=0.001"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>0.003, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=0.003"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>0.01, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=0.01"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>0.03, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=0.03"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>0.1, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=0.1"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>0.3, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=0.3"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>true, "learning_rate_ivon"=>1.0, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "IVON lr=1.0"),
        ],
        "IVON lr comparison on $env_name"
    )
end

function generate_normal_nn_learning_rate_comparison(env_name::String, sliding_reward_start::Float64, num_batches::Int=2500)
    return (
        env_name,
        [
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>0.0001, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=0.0001"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>0.0003, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=0.0003"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>0.001, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=0.001"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>0.003, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=0.003"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>0.01, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=0.01"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>0.03, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=0.03"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>0.1, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=0.1"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>0.3, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=0.3"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "learning_rate_normal"=>1.0, "sliding_reward_start"=>sliding_reward_start, "num_batches"=>num_batches), "PyTorch NN lr=1.0"),
        ],
        "PyTorch NN lr comparison on $env_name"
    )
end

function generate_bayesian_vs_normal_combinations(env_name::String, sliding_reward_start::Float64; num_batches::Int=1000)
    return (
        env_name,
        [
            ("Bayesian A2C", Dict("use_normal_actor"=>false, "use_normal_critic"=>false, "use_ivon_for_normal"=>false, "num_batches"=>num_batches, "sliding_reward_start"=>sliding_reward_start), "Bayesian Message Passing Actor/Critic"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>false, "use_ivon_for_normal"=>false, "num_batches"=>num_batches, "sliding_reward_start"=>sliding_reward_start), "Actor: PyTorch NN, Critic: Bayesian MP"),
            ("Bayesian A2C", Dict("use_normal_actor"=>false, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "num_batches"=>num_batches, "sliding_reward_start"=>sliding_reward_start), "Actor: Bayesian MP, Critic: PyTorch NN"),
            ("Bayesian A2C", Dict("use_normal_actor"=>true, "use_normal_critic"=>true, "use_ivon_for_normal"=>false, "num_batches"=>num_batches, "sliding_reward_start"=>sliding_reward_start), "PyTorch NN Actor/Critic"),
        ],
        "Bayesian vs Normal combinations on $env_name"
    )
end

# -------------------------------------------------------------------------
all_experiments = [
    generate_standard_configuration_different_environment("Pendulum-v1", -7.0),
    generate_standard_configuration_different_environment("MountainCarContinuous-v0", -0.06),
    generate_standard_configuration_different_environment("BipedalWalker-v3", -0.5),
    generate_standard_configuration_different_environment("Swimmer-v5", 0.0),
    generate_standard_configuration_different_environment("InvertedPendulum-v5", 0.8),
    generate_standard_configuration_different_environment("InvertedDoublePendulum-v5", 7.0),
    generate_standard_configuration_different_environment("HalfCheetah-v5", -0.5),
    generate_standard_configuration_different_environment("Reacher-v5", -1.4),

    generate_normal_nn_learning_rate_comparison("Pendulum-v1", -7.0),
    generate_ivon_learning_rate_comparison("Pendulum-v1", -7.0),

    (
        "Pendulum-v1",
        [
            ("Bayesian A2C", Dict("lr_decrease" => 0.99, "num_batches"=>2500), "BA2C with LR decrease 0.99 (fast)"),
            ("Bayesian A2C", Dict("lr_decrease" => 0.999, "num_batches"=>2500), "BA2C with LR decrease 0.999 (standard)"),
            ("Bayesian A2C", Dict("lr_decrease" => 0.9999, "num_batches"=>2500), "BA2C with LR decrease 0.9999 (slow)"),
            ("Bayesian A2C", Dict("lr_decrease" => 1.0, "num_batches"=>2500), "BA2C no LR decrease"),
        ],
        "Bayesian A2C LR decrease on Pendulum-v1"
    ),

    (
        "Pendulum-v1",
        [
            ("Bayesian A2C", Dict("normalize_advantages" => false, "num_batches"=>2500), "Bayesian No Normalization"),
            ("Bayesian A2C", Dict("normalize_advantages" => true, "num_batches"=>2500), "Bayesian With Normalization (standard)"),
        ],
        "Bayesian A2C Normalization on Pendulum-v1"
    ),

    (
        "Pendulum-v1",
        [
            ("Bayesian A2C", Dict("use_normal_actor" => true, "use_normal_critic" => true, "use_ivon_for_normal" => false), "PyTorch NN Actor/Critic"),
            ("Bayesian A2C", Dict("use_normal_actor" => true, "use_normal_critic" => true, "use_ivon_for_normal" => true), "IVON Actor/Critic"),
            ("Bayesian A2C", Dict("use_normal_actor" => false, "use_normal_critic" => false, "use_ivon_for_normal" => false), "Bayesian Message Passing Actor/Critic"),
        ],
        "Bayesian A2C IVON comparison on Pendulum-v1"
    ),

    (
        "Pendulum-v1",
        [
            ("Bayesian A2C", Dict("learning_rate"=>0.03), "BA2C LR=0.03"),
            ("Bayesian A2C", Dict("learning_rate"=>0.04), "BA2C LR=0.04"),
            ("Bayesian A2C", Dict("learning_rate"=>0.05), "BA2C LR=0.05 (standard)"),
            ("Bayesian A2C", Dict("learning_rate"=>0.06), "BA2C LR=0.06"),
            ("Bayesian A2C", Dict("learning_rate"=>0.07), "BA2C LR=0.07"),
        ],
        "Bayesian A2C Learning Rate on Pendulum-v1"
    ),

    generate_bayesian_vs_normal_combinations("Pendulum-v1", -7.0, num_batches=2500),
    generate_bayesian_vs_normal_combinations("Pendulum-v1", -7.0),
    generate_bayesian_vs_normal_combinations("MountainCarContinuous-v0", -0.06),
    generate_bayesian_vs_normal_combinations("BipedalWalker-v3", -0.5),
    generate_bayesian_vs_normal_combinations("Swimmer-v5", 0.0),
    generate_bayesian_vs_normal_combinations("InvertedPendulum-v5", 0.8),
    generate_bayesian_vs_normal_combinations("InvertedDoublePendulum-v5", 7.0),
    generate_bayesian_vs_normal_combinations("HalfCheetah-v5", -0.5),
    generate_bayesian_vs_normal_combinations("Reacher-v5", -1.4),

    generate_standard_ivon_comparison("Pendulum-v1", -7.0),
    generate_standard_ivon_comparison("MountainCarContinuous-v0", -0.06),
    generate_standard_ivon_comparison("BipedalWalker-v3", -0.5),
    generate_standard_ivon_comparison("Swimmer-v5", 0.0),
    generate_standard_ivon_comparison("InvertedPendulum-v5", 0.8),
    generate_standard_ivon_comparison("InvertedDoublePendulum-v5", 7.0),
    generate_standard_ivon_comparison("HalfCheetah-v5", -0.5),
    generate_standard_ivon_comparison("Reacher-v5", -1.4),

    generate_standard_std_ignore_comparison("Pendulum-v1", -7.0),
    generate_standard_std_ignore_comparison("MountainCarContinuous-v0", -0.06),
    generate_standard_std_ignore_comparison("BipedalWalker-v3", -0.5),
    generate_standard_std_ignore_comparison("Swimmer-v5", 0.0),
    generate_standard_std_ignore_comparison("InvertedPendulum-v5", 0.8),
    generate_standard_std_ignore_comparison("InvertedDoublePendulum-v5", 7.0),
    generate_standard_std_ignore_comparison("HalfCheetah-v5", -0.5),
    generate_standard_std_ignore_comparison("Reacher-v5", -1.4),

    (
        "Pendulum-v1",
        [
            ("Bayesian A2C", Dict("action_std_expansion_factor"=>3.0), "Bayes factor 3.0"),
            ("Bayesian A2C", Dict("action_std_expansion_factor"=>3.5), "Bayes factor 3.5"),
            ("Bayesian A2C", Dict("action_std_expansion_factor"=>4.1), "Bayes factor 4.0 (standard)"),
            ("Bayesian A2C", Dict("action_std_expansion_factor"=>4.3), "Bayes factor 4.5"),
            ("Bayesian A2C", Dict("action_std_expansion_factor"=>5.0), "Bayes factor 5.0"),
        ],
        "Bayesian A2C Action Std Expansion Factor on Pendulum-v1"
    ),

    (
        "Pendulum-v1",
        [
            ("Bayesian A2C", Dict("default_beta_actor"=>0.025), "Bayes beta actor 0.025"),
            ("Bayesian A2C", Dict("default_beta_actor"=>0.03), "Bayes beta actor 0.03"),
            ("Bayesian A2C", Dict("default_beta_actor"=>0.04), "Bayes beta actor 0.04"),
            ("Bayesian A2C", Dict("default_beta_actor"=>0.05), "Bayes beta actor 0.05 (standard)"),
            ("Bayesian A2C", Dict("default_beta_actor"=>0.06), "Bayes beta actor 0.06"),
            ("Bayesian A2C", Dict("default_beta_actor"=>0.07), "Bayes beta actor 0.07"),
        ],
        "Bayesian A2C Beta Actor on Pendulum-v1"
    ),
]

# -------------------------------------------------------------------------
# 6) Main loop over all_experiments
# -------------------------------------------------------------------------


# model_checkpoints = [200, 500,
#     1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
#     10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000,
#     30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000, 100000,
#     120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000,
#     300000, 350000, 400000, 450000, 500000, 600000, 700000, 800000, 900000, 1000000,
#     1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2400000, 2600000, 2800000,
#     3000000, 3500000, 4000000, 4500000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]

# For this experiment setup, we are not saving model parameters
model_checkpoints = []
# Transform model_checkpoints into an (empty) vector of Int64
model_checkpoints = Vector{Int}(model_checkpoints)

for (env_name, run_list, experiment_name) in all_experiments
    println("Running experiments for $env_name ...")
    # We will collect for each run a tuple: (legend_label, [ [rewards for each parallel run] ])
    all_runs_for_env = Vector{Tuple{String,Vector{Vector{Float64}}}}()
    evaluation_runs_for_env = Vector{Tuple{String,Vector{Vector{Float64}}}}()

    for (algo_name, config_changes, run_label) in run_list
        println("  - Launching $algo_name with label '$run_label' and config changes: $config_changes ...")

        env_config = get_fast_configuration(config_changes)

        config_str = config_to_string(env_config)
        cache_key  = (algo_name, env_name, config_str)

        if haskey(EVALUATION_CACHE, cache_key)
            @info "Evaluation is cached for $algo_name / $run_label, skipping actual evaluate_algorithm."
            rewards_list = EVALUATION_CACHE[cache_key]
        else
            futures_algo = Vector{Future}(undef, num_processes)

            @sync begin
                for i in 1:num_processes
                    sleep(i)  # to avoid overloading the scheduler
                    futures_algo[i] = @spawnat i train_algorithm(algo_name, env_name, env_config, experiment_name, model_checkpoints)
                end
            end

            # Gather
            rewards_list = fetch.(futures_algo)

            # Save the results
            EVALUATION_CACHE[cache_key] = rewards_list
        end

        # Save the results
        push!(all_runs_for_env, (run_label, rewards_list))

        println("Training of $experiment_name with $algo_name done. Now evaluating ...")

        # futures_evaluator = Vector{Future}(undef, num_processes)
        # @sync begin
        #     for i in 1:num_processes
        #         sleep(i)
        #         futures_evaluator[i] = @spawnat i evaluate_algorithm(algo_name, env_name, experiment_name, model_checkpoints)
        #     end
        # end
        # evaluated_results = fetch.(futures_evaluator)
        # println("Got these results: $evaluated_results")
        # push!(evaluation_runs_for_env, (run_label, evaluated_results))

        # experiment_name_replaced = replace(experiment_name, " " => "_")
        # # Delete the directory with the saved models
        # rm("saved_models/$experiment_name_replaced", recursive=true)
    end

    # Now plot all runs for this environment in a single diagram
    println("Plotting combined results for $env_name ...")
    plot_multiple_runs(all_runs_for_env, env_name, experiment_name)

    # if !isempty(evaluation_runs_for_env)
    #     println("Plotting evaluator results for $env_name ...")
    #     plot_multiple_runs_evaluation(evaluation_runs_for_env, env_name, experiment_name, model_checkpoints)
    #     println("Done evaluator plots for $env_name\n\n")
    # end
    
    println("Done with $env_name\n\n")
end

println("All environments done!")
