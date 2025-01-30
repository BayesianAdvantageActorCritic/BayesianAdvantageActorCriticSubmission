using PyCall
using LinearAlgebra
using Plots
using ProgressMeter
using Statistics
using Serialization
using Dates

include("lib/factor_graph.jl")

np = pyimport("numpy")
gym = pyimport("gymnasium")
os = pyimport("os")
wandb = pyimport("wandb")
gym_spaces = pyimport("gymnasium.spaces")

mutable struct BayesianA2CAgent
    config::Dict{String, Any}
    experiment_name::String
    environments::Vector{PyObject}
    input_shape::Int
    output_shape::Int
    """ 
    actor_mean can now be either:
      - a single FactorGraph/PyObject (for 1D action or original multi-D approach)
      - a Vector{FactorGraph} (for multiple 1D networks if 'use_independent_networks' is set)
    """
    actor_mean::Union{FactorGraph, Vector{FactorGraph}, PyObject}
    actor_trainer::Union{Trainer, Vector{Trainer}, Nothing}
    critic::Union{FactorGraph, PyObject}
    critic_trainer::Union{Trainer, Nothing}
    observations::Union{Array{Float64, 2}, Nothing}
    actions::Union{Array{Float64, 2}, Nothing}
    rewards::Union{Array{Float64, 1}, Nothing}
    dones::Union{Array{Bool, 1}, Nothing}
    min_action::Float64
    max_action::Float64
end

function generate_architecture(input_shape, action_space_dimensions, regression_basic_architecture)
    return [
        (input_shape,),
        (:Linear, 64),
        (:LeakyReLU, 0.1),
        (:Linear, 64),
        (:LeakyReLU, 0.1),
        (:Linear, action_space_dimensions),
        (:Regression, regression_basic_architecture^2),
    ]
end

"""
    create_independent_networks(input_shape, output_dim, beta, number_environments, num_batches)

Create a vector of factor graphs, each having exactly 1 output dimension. We also create
the corresponding trainers in a vector.
"""
function create_independent_networks(input_shape::Int, output_dim::Int, beta::Float64,
                                     number_environments::Int, num_batches::Int)
    networks = Vector{FactorGraph}(undef, output_dim)
    trainers = Vector{Trainer}(undef, output_dim)
    for i in 1:output_dim
        # Each FactorGraph has 1 output dimension
        fg = create_factor_graph(generate_architecture(input_shape, 1, beta), number_environments)
        networks[i] = fg
        trainers[i] = Trainer(fg, num_batches)
    end
    return networks, trainers
end


function BayesianA2CAgent(env_class::Any; config::Dict{String, Any} = Dict(
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
    ),
    experiment_name::String = "BayesianA2C")
    println("Creating agent with config: $config")

    number_environments = config["number_environments"]
    num_batches = config["num_batches"]

    environments = [env_class() for _ in 1:number_environments]

    action_space = environments[1][:action_space]
    min_action = action_space[:low][1]
    max_action = action_space[:high][1]
    println("Min action: $min_action, Max action: $max_action")

    input_shape = Int(environments[1][:observation_space][:shape][1])
    if !(pyisinstance(action_space, gym_spaces.Box))
        error("Action space is not a Box space")
    end
    output_shape = Int(action_space[:shape][1])

    println("Input shape: $input_shape, Output shape: $output_shape")

    # --- Actor creation ---
    if config["use_normal_actor"] == true
        if config["use_ivon_for_normal"]
            actor_mean = pyimport("julia_implementation.actor_network_ivon").ActorNetwork(input_shape, output_shape, config["learning_rate_ivon"])
        else
            actor_mean = pyimport("julia_implementation.actor_network").ActorNetwork(input_shape, output_shape, config["learning_rate_normal"])
        end
        actor_trainer = nothing
    else
        # If user wants an independent network for each action dimension:
        if haskey(config, "use_independent_networks") && config["use_independent_networks"] == true && output_shape > 1
            actor_mean, actor_trainer = create_independent_networks(
                input_shape, output_shape, config["default_beta_actor"], number_environments, num_batches)
        else
            actor_mean = create_factor_graph(generate_architecture(input_shape, output_shape, config["default_beta_actor"]), number_environments)
            actor_trainer = Trainer(actor_mean, num_batches)
        end
    end

    # --- Critic creation ---
    if config["use_normal_critic"] == true
        if config["use_ivon_for_normal"]
            critic = pyimport("julia_implementation.critic_network_ivon").CriticNetwork(input_shape, config["learning_rate_ivon"])
        else
            critic = pyimport("julia_implementation.critic_network").CriticNetwork(input_shape, config["learning_rate_normal"])
        end
        critic_trainer = nothing
        if config["use_pretrained_critic"]
            critic[:load_critic]("saved_models/ppo_pendulum.zip")
        end
    else
        # We usually only need 1D output for critic, so no changes here
        critic = create_factor_graph(generate_architecture(input_shape, 1, config["default_beta_critic"]), number_environments)
        critic_trainer = Trainer(critic, num_batches)
    end

    observations = nothing
    actions = nothing
    rewards = nothing
    dones = nothing

    return BayesianA2CAgent(config, experiment_name, environments, input_shape, output_shape,
                           actor_mean, actor_trainer, critic, critic_trainer,
                           observations, actions, rewards, dones, min_action, max_action)
end

"""
    unpack_predictions(model::FactorGraph, input::AbstractMatrix{Float64})

Unchanged from your original single-network version.
"""
function unpack_predictions(model::FactorGraph, input::AbstractMatrix{Float64})
    predictions = predict(model, input, silent = true)
    means = [[mean(pred) for pred in predictions[i, :]] for i in 1:size(predictions,1)]
    stds = [[sqrt(variance(pred)) for pred in predictions[i, :]] for i in 1:size(predictions,1)]
    return Matrix(hcat(means...)'), Matrix(hcat(stds...)')
end

"""
    unpack_predictions(models::Vector{FactorGraph}, input::AbstractMatrix{Float64})

Multi-dispatch version that handles multiple 1D networks. We simply call unpack_predictions
on each factor graph individually, then assemble them into a (output_dim, N) shape.
"""
function unpack_predictions(models::Vector{FactorGraph}, input::AbstractMatrix{Float64})
    output_dim = length(models)
    N = size(input, 2)
    means_all = zeros(output_dim, N)
    stds_all  = zeros(output_dim, N)

    for i in 1:output_dim
        m, s = unpack_predictions(models[i], input)
        # Because each sub-network has shape (1, N):
        @assert size(m,1) == 1 && size(m,2) == N
        @assert size(s,1) == 1 && size(s,2) == N
        means_all[i, :] .= m[1, :]
        stds_all[i,  :] .= s[1, :]
    end
    return means_all, stds_all
end


function create_folders()
    if !isdir("figures")
        mkdir("figures")
    end
    if !isdir("saved_models")
        mkdir("saved_models")
    end
end

function initialize_observations(agent::BayesianA2CAgent)
    observations = [agent.environments[i][:reset]()[1] for i in 1:agent.config["number_environments"]]
    return Float64.(Matrix(hcat(observations...)))
end

function pre_execute_environments(agent::BayesianA2CAgent)
    println("Executing random actions to bring the environment to a different state")
    for (env_idx, env) in enumerate(agent.environments)
        for _ in 1:(env_idx*3)
            result = env[:step](env[:action_space][:sample]())
            if result[3] || result[4]
                env[:reset]()
            end
        end
    end
end

function get_actions(agent::BayesianA2CAgent, observations::AbstractMatrix{Float64}; use_expansion::Bool = true)
    # If we use normal networks, call :predict, else call our Bayesian approach.
    if agent.config["use_normal_actor"] == true
        action_means, action_stds = agent.actor_mean[:predict](observations)
    else
        # This will dispatch to either the single-FG or multiple-FG version
        action_means, action_stds = unpack_predictions(agent.actor_mean, observations)
    end

    if agent.config["ignore_std_output"]
        action_stds .= agent.config["default_beta_actor"]
    end

    if use_expansion
        action_stds .= action_stds .* agent.config["action_std_expansion_factor"]  # scale
    end

    @assert !any(isnan.(action_means)) "action_means: $action_means"
    @assert !any(isnan.(action_stds)) "action_stds: $action_stds"

    action_means = clamp.(action_means, agent.config["action_percentage_clipping"] * agent.min_action, agent.config["action_percentage_clipping"] * agent.max_action)

    actions = randn(size(action_means)) .* action_stds .+ action_means
    actions = clamp.(actions, agent.min_action, agent.max_action)

    return actions, action_means, action_stds
end

function take_environment_steps(agent::BayesianA2CAgent, actions::AbstractMatrix{Float64})
    next_observations = []
    rewards = Float64[]
    dones = Bool[]
    for env_idx in 1:agent.config["number_environments"]
        env = agent.environments[env_idx]
        action = actions[:, env_idx]
        cumulative_reward = 0.0
        done = false
        next_obs = nothing
        for _ in 1:agent.config["num_subsequent_steps"]
            if done
                break
            end
            result = env[:step](np.array(action))
            next_obs = result[1]
            reward = result[2]
            done1 = result[3]
            done2 = result[4]
            cumulative_reward += reward
            done = done1 || done2
        end
        push!(next_observations, next_obs)
        push!(rewards, cumulative_reward)
        push!(dones, done)
    end
    return next_observations, rewards, dones
end

function compute_advantages(agent::BayesianA2CAgent, rewards::AbstractMatrix{Float64}, value_means::AbstractMatrix{Float64}, value_stds::AbstractMatrix{Float64}, next_value_means::AbstractMatrix{Float64}, next_value_stds::AbstractMatrix{Float64})
    td_target_means = rewards .+ agent.config["gamma"] .* next_value_means
    td_target_stds = agent.config["gamma"] .* next_value_stds
    advantage_means = td_target_means .- value_means
    advantage_stds = sqrt.((td_target_stds .^ 2) .+ (value_stds .^ 2))

    @assert size(td_target_means) == (1, agent.config["number_environments"])
    @assert size(td_target_stds) == (1, agent.config["number_environments"])
    @assert size(advantage_means) == (1, agent.config["number_environments"])
    @assert size(advantage_stds) == (1, agent.config["number_environments"])

    # Calculate normalized advantages
    scaling_factor = min(1.0 / (Statistics.std(advantage_means) .+ 1e-8), 10)
    if agent.config["normalize_advantages"]
        advantage_means .= scaling_factor .* advantage_means
        advantage_stds  .= scaling_factor .* advantage_stds
    end

    return td_target_means, td_target_stds, advantage_means, advantage_stds, scaling_factor
end

function adjust_means(
    action_means::AbstractMatrix{Float64},
    action_stds::AbstractMatrix{Float64},
    actions::AbstractMatrix{Float64},
    advantage_means::AbstractMatrix{Float64},
    advantage_stds::AbstractMatrix{Float64},
    agent::BayesianA2CAgent,
)
    function_dependent_factor = (actions .- action_means) ./ (action_stds .^ 2)
    changes = advantage_means .* function_dependent_factor
    changes_stds = advantage_stds .* function_dependent_factor
    adjusted_means = action_means .+ agent.config["learning_rate"] .* changes
    adjusted_means_stds = action_stds .+ agent.config["learning_rate"] .* changes_stds

    spread_before_clamp = vec(abs.(action_means .- adjusted_means))

    if haskey(agent.config, "clip_factor")
        adjusted_means .= clamp.(adjusted_means,
                            action_means .- agent.config["clip_factor"] .* action_stds,
                            action_means .+ agent.config["clip_factor"] .* action_stds)
    end

    adjusted_means .= clamp.(adjusted_means,
                        agent.config["action_percentage_clipping"] * agent.min_action,
                        agent.config["action_percentage_clipping"] * agent.max_action)
    adjusted_means_stds .= clamp.(adjusted_means_stds, 0.01, agent.max_action)

    spread_after_clamp = vec(abs.(action_means .- adjusted_means))

    return adjusted_means, adjusted_means_stds, spread_before_clamp, spread_after_clamp, 0
end

function reset_environments_if_needed(agent::BayesianA2CAgent, dones::Vector{Bool}, next_observations::AbstractVector{})
    for env_idx in 1:agent.config["number_environments"]
        if dones[env_idx] || rand() < agent.config["probability_random_reset"]
            obs_info = agent.environments[env_idx][:reset]()
            next_observations[env_idx] = obs_info[1]
        end
    end
    next_observations = Float64.(Matrix(hcat(next_observations...)))

    @assert size(next_observations) == (agent.input_shape, agent.config["number_environments"])

    return next_observations
end

function should_save_model(sliding_reward::Float64, best_sliding_reward::Float64, timestep::Int, warmup::Int)
    return sliding_reward > best_sliding_reward &&
           (abs(best_sliding_reward) < 0.01 || (sliding_reward - best_sliding_reward) / abs(best_sliding_reward) >= 0.02) &&
           timestep > warmup + div(warmup, 2)
end

function save_agent(agent::BayesianA2CAgent, process_idx::Int, name::String = "best_agent", subfolder::Union{String, Nothing} = nothing)
    agent_to_save = deepcopy(agent)
    agent_to_save.environments = []
    agent_to_save.actor_trainer = nothing
    agent_to_save.critic_trainer = nothing
    agent_to_save.observations = nothing
    agent_to_save.actions = nothing
    agent_to_save.rewards = nothing
    agent_to_save.dones = nothing
    path = "saved_models"
    if !isnothing(subfolder)
        path = joinpath(path, subfolder)
    end
    if !isdir(path)
        mkdir(path)
    end
    path = joinpath(path, "$(name)_$process_idx.jls")
    open(path, "w") do io
        serialize(io, agent_to_save)
    end
end

function plot_sliding_rewards(sliding_rewards::Vector{Float64})
    current_time = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    file_path_pdf = "figures/Learning_Curve_$(current_time).pdf"
    p = Plots.plot(sliding_rewards, xlabel = "Learned Samples (x64)", ylabel = "Sliding Reward", title = "Sliding Reward over Time")
    savefig(p, file_path_pdf)
end

function learn(agent::BayesianA2CAgent; process_idx::Int = 0, use_wandb::Bool = true, model_checkpoints::Union{Vector{Int}, Nothing} = nothing, save_best_parameters=false)
    create_folders()
    observations = initialize_observations(agent)
    if use_wandb
        wandb.init(project = "Evaluation for $(agent.experiment_name)", entity = "your_wandb_account")
    end
    if agent.config["pre_execution"]
        pre_execute_environments(agent)
    end
    sliding_f_clipped = 0.0
    sliding_reward = agent.config["sliding_reward_start"]
    sliding_reward_other_scale = agent.config["sliding_reward_start"]
    best_sliding_reward = agent.config["sliding_reward_start"]
    sliding_rewards = Float64[]
    sliding_rewards_other_scale = Float64[]
    warmup = 100

    if haskey(agent.config, "lr_decrease")
        agent.config["default_learning_rate"] = agent.config["learning_rate"]
    end

    println("Starting training")
    @showprogress for timestep in 1:agent.config["num_batches"]
        if haskey(agent.config, "lr_decrease")
            agent.config["learning_rate"] = max(
                agent.config["default_learning_rate"] * (agent.config["lr_decrease"]^timestep),
                agent.config["learning_rate"] / 10
            )
        end

        actions, action_means, action_stds = get_actions(agent, observations)
        next_observations, rewards, dones = take_environment_steps(agent, actions)
        observations_next = Float64.(Matrix(hcat(next_observations...)))
        rewards = Float64.(hcat(rewards...))

        for slos in rewards
            sliding_reward_other_scale = 0.9997 * sliding_reward_other_scale + 0.0003 * slos
            push!(sliding_rewards_other_scale, sliding_reward_other_scale)
        end

        sliding_reward = 0.97 * sliding_reward + 0.03 * Statistics.mean(rewards)

        # Critic predictions
        if agent.config["use_normal_critic"]
            value_means, value_stds = agent.critic[:predict](observations)
            next_value_means, next_value_stds = agent.critic[:predict](observations_next)
        else
            value_means, value_stds = unpack_predictions(agent.critic, observations)
            next_value_means, next_value_stds = unpack_predictions(agent.critic, observations_next)

            if agent.config["ignore_std_output"]
                value_stds .= agent.config["default_beta_critic"]
                next_value_stds .= agent.config["default_beta_critic"]
            end
        end

        td_target_means, td_target_stds, advantage_means, advantage_stds, scaling_factor = compute_advantages(
            agent, rewards, value_means, value_stds, next_value_means, next_value_stds)

        taus_observation_actor = [0.0]
        precisions_observation_actor = [0.0]
        taus_observation_critic = [0.0]
        precisions_observation_critic = [0.0]

        # Train critic
        if agent.config["use_normal_critic"]
            if !agent.config["use_pretrained_critic"]
                agent.critic[:train_batch](observations, td_target_means)
            end
        else
            train_batch_new(agent.critic_trainer, timestep, observations, td_target_means, td_target_stds,
                agent.config["number_factor_graph_iterations"], agent.config["beta_ema_critic"],
                taus_observation_critic, precisions_observation_critic)
        end

        if agent.config["use_normal_critic"]
            value_means_new, value_stds_new = agent.critic[:predict](observations)
        else
            value_means_new, value_stds_new = unpack_predictions(agent.critic, observations)
        end

        realized_spread_critic = abs.(value_means_new .- value_means)
        relative_realized_spread_critic = (value_means_new .- value_means) ./ (td_target_means .- value_means)

        adjusted_means, adjusted_means_stds, spread_before_clamp, spread_after_clamp, f_clipped =
            adjust_means(action_means, action_stds, actions, advantage_means, advantage_stds, agent)

        if agent.config["use_pretrained_critic"] && timestep < warmup
            # Shuffle means for warmup
            adjusted_means .= 2 .* rand(size(adjusted_means)) .- 1
        end

        sliding_f_clipped = 0.9 * sliding_f_clipped + 0.1 * f_clipped

        # Train the actor: either in normal mode or Bayesian mode. 
        if agent.config["use_normal_actor"] == true
            agent.actor_mean[:train_batch](observations, adjusted_means)
        else
            # If actor_mean is a vector => train each dimension separately
            if agent.actor_mean isa Vector{FactorGraph}
                @assert agent.actor_trainer isa Vector{Trainer}
                for dim in 1:length(agent.actor_mean)
                    # Each sub-network trains on the single column for that action dimension
                    train_batch_new(agent.actor_trainer[dim], timestep,
                                    observations,
                                    adjusted_means[dim:dim, :],      # 1 row
                                    adjusted_means_stds[dim:dim, :], # 1 row
                                    agent.config["number_factor_graph_iterations"],
                                    agent.config["beta_ema_actor"],
                                    taus_observation_actor,
                                    precisions_observation_actor)
                end
            else
                # Original single FactorGraph approach
                train_batch_new(agent.actor_trainer, timestep, observations, adjusted_means, adjusted_means_stds,
                    agent.config["number_factor_graph_iterations"], agent.config["beta_ema_actor"],
                    taus_observation_actor, precisions_observation_actor)
            end
        end

        # Evaluate new actor predictions
        actions_new, action_means_new, action_stds_new = get_actions(agent, observations)
        realized_spread_actor = abs.(action_means_new .- action_means)
        relative_realized_spread_actor = (action_means_new .- action_means) ./ (adjusted_means .- action_means .+ 1e-4)

        observations = reset_environments_if_needed(agent, dones, next_observations)

        if save_best_parameters && should_save_model(sliding_reward, best_sliding_reward, timestep, warmup)
            best_sliding_reward = sliding_reward
            println("New best sliding reward: $best_sliding_reward after timestep: $timestep")
            save_agent(agent, process_idx)
        end

        if !isnothing(model_checkpoints) && length(model_checkpoints) > 0
            next_checkpoint = model_checkpoints[1]
            if timestep * agent.config["number_environments"] >= next_checkpoint
                save_agent(agent, process_idx, "model_checkpoint_$next_checkpoint", agent.experiment_name)
                println("Saved model at timestep $timestep with next_checkpoint $next_checkpoint with sliding reward $sliding_reward (experiment_name: $(agent.experiment_name), process_idx: $process_idx)")
                popfirst!(model_checkpoints)
            end
        end

        push!(sliding_rewards, sliding_reward)
        if use_wandb
            log_metrics(wandb, agent, sliding_reward, best_sliding_reward, rewards, action_means, action_stds,
                        adjusted_means, adjusted_means_stds, td_target_means, td_target_stds,
                        value_means, value_stds, advantage_means, actions, sliding_f_clipped,
                        spread_before_clamp, spread_after_clamp,
                        realized_spread_actor, relative_realized_spread_actor,
                        realized_spread_critic, relative_realized_spread_critic,
                        taus_observation_actor, precisions_observation_actor,
                        taus_observation_critic, precisions_observation_critic, scaling_factor)
        end
    end

    # plot_sliding_rewards(sliding_rewards)

    if use_wandb
        wandb.finish()
    end

    return sliding_rewards, sliding_rewards_other_scale
end

function log_metrics(
    wandb::PyObject,
    agent::BayesianA2CAgent,
    sliding_reward::Float64,
    best_sliding_reward::Float64,
    rewards::AbstractMatrix{Float64},
    action_means::AbstractMatrix{Float64},
    action_stds::AbstractMatrix{Float64},
    adjusted_means::AbstractMatrix{Float64},
    adjusted_means_stds::AbstractMatrix{Float64},
    td_target_means::AbstractMatrix{Float64},
    td_target_stds::AbstractMatrix{Float64},
    value_means::AbstractMatrix{Float64},
    value_stds::AbstractMatrix{Float64},
    advantage_means::AbstractMatrix{Float64},
    actions::AbstractMatrix{Float64},
    sliding_f_clipped::Float64,
    spread_before_clamp::AbstractVector{Float64},
    spread_after_clamp::AbstractVector{Float64},
    realized_spread_actor::AbstractMatrix{Float64},
    relative_realized_spread_actor::AbstractMatrix{Float64},
    realized_spread_critic::AbstractMatrix{Float64},
    relative_realized_spread_critic::AbstractMatrix{Float64},
    taus_observation_actor::AbstractVector{Float64},
    precisions_observation_actor::AbstractVector{Float64},
    taus_observation_critic::AbstractVector{Float64},
    precisions_observation_critic::AbstractVector{Float64},
    scaling_factor::Float64,
)
    metrics = Dict()
    percentiles = [10, 25, 50, 75, 90]

    # Performance
    metrics["performance/Sliding Reward"] = sliding_reward
    metrics["performance/Mean Reward"] = Statistics.mean(rewards)
    metrics["performance/Best Sliding Reward"] = best_sliding_reward

    # Current action metrics
    metrics["current_action/Mean Action Mean"] = Statistics.mean(action_means)
    metrics["current_action/Mean Action Std"] = Statistics.mean(action_stds)
    for p in percentiles
        metrics["current_action/Mean Action Mean $(p)% Percentile"] = quantile(vec(action_means), p / 100)
        metrics["current_action/Mean Action Std $(p)% Percentile"] = quantile(vec(action_stds), p / 100)
    end

    # Actor training metrics
    metrics["actor_training/Mean adjusted_means"] = Statistics.mean(adjusted_means)
    metrics["actor_training/Mean adjusted_stds"] = Statistics.mean(adjusted_means_stds)
    for p in percentiles
        metrics["actor_training/adjusted_means $(p)% Percentile"] = quantile(vec(adjusted_means), p / 100)
        metrics["actor_training/adjusted_stds $(p)% Percentile"] = quantile(vec(adjusted_means_stds), p / 100)
    end

    # Spread metrics
    metrics["spread/Spread after clamp"] = Statistics.mean(spread_after_clamp)
    metrics["spread/Spread before clamp"] = Statistics.mean(spread_before_clamp)
    for p in percentiles
        metrics["spread/Spread after clamp $(p)% Percentile"] = quantile(spread_after_clamp, p / 100)
        metrics["spread/Spread before clamp $(p)% Percentile"] = quantile(spread_before_clamp, p / 100)
    end

    # Spread analysis metrics
    changes = abs.(actions .- action_means)
    metrics["spread_analysis/Changes Mean"] = Statistics.mean(changes)
    for p in percentiles
        metrics["spread_analysis/Changes $(p)% Percentile"] = quantile(vec(changes), p / 100)
        metrics["spread_analysis/Action difference $(p)% Percentile"] = quantile(vec(abs.(actions .- action_means)), p / 100)
        metrics["spread_analysis/Weighted Changes $(p)% Percentile"] = quantile(vec(abs.(agent.config["learning_rate"] .* changes)), p / 100)
    end
    metrics["spread_analysis/Action difference Mean"] = Statistics.mean(abs.(actions .- action_means))
    metrics["spread_analysis/Action std"] = Statistics.mean(action_stds)
    metrics["spread_analysis/Weighted Changes Mean"] = Statistics.mean(abs.(agent.config["learning_rate"] .* changes))

    # Critic training metrics
    for p in percentiles
        metrics["critic_training/td_target_means $(p)% Percentile"] = quantile(vec(td_target_means), p / 100)
        metrics["critic_training/td_target_stds $(p)% Percentile"] = quantile(vec(td_target_stds), p / 100)
        metrics["critic_training/Value $(p)% Percentile"] = quantile(vec(value_means), p / 100)
        metrics["critic_training/Value Std $(p)% Percentile"] = quantile(vec(value_stds), p / 100)
    end
    metrics["critic_training/Mean Value"] = Statistics.mean(value_means)
    metrics["critic_training/Mean Value Std"] = Statistics.mean(value_stds)

    # MSE between value_means and td_target_means
    mse = Statistics.mean((value_means .- td_target_means).^2)
    metrics["critic_training/MSE Values TD_Targets"] = mse

    # Advantage metrics
    metrics["advantage/Mean Advantage"] = Statistics.mean(advantage_means)
    for p in percentiles
        metrics["advantage/Advantage $(p)% Percentile"] = quantile(vec(advantage_means), p / 100)
    end

    # Training metrics
    metrics["training/learning_rate"] = agent.config["learning_rate"]
    metrics["training/gamma"] = agent.config["gamma"]
    metrics["training/sliding_f_clipped"] = sliding_f_clipped
    metrics["training/Scaling Factor"] = scaling_factor

    # Actor realized spread
    metrics["actor_realized_spread/Realized Spread Mean"] = Statistics.mean(realized_spread_actor)
    for p in percentiles
        metrics["actor_realized_spread/Realized Spread $(p)% Percentile"] = quantile(vec(realized_spread_actor), p / 100)
    end
    metrics["actor_realized_spread/Relative Realized Spread Mean"] = Statistics.mean(relative_realized_spread_actor)
    for p in percentiles
        metrics["actor_realized_spread/Relative Realized Spread $(p)% Percentile"] = quantile(vec(relative_realized_spread_actor), p / 100)
    end

    # Critic realized spread
    metrics["critic_realized_spread/Realized Spread Mean"] = Statistics.mean(realized_spread_critic)
    for p in percentiles
        metrics["critic_realized_spread/Realized Spread $(p)% Percentile"] = quantile(vec(realized_spread_critic), p / 100)
    end
    metrics["critic_realized_spread/Relative Realized Spread Mean"] = Statistics.mean(relative_realized_spread_critic)
    for p in percentiles
        metrics["critic_realized_spread/Relative Realized Spread $(p)% Percentile"] = quantile(vec(relative_realized_spread_critic), p / 100)
    end

    # Actor observation metrics
    metrics["actor_observations/Mean Tau"] = Statistics.mean(taus_observation_actor)
    for p in percentiles
        metrics["actor_observations/Tau $(p)% Percentile"] = quantile(vec(taus_observation_actor), p / 100)
    end
    metrics["actor_observations/Mean Precision"] = Statistics.mean(precisions_observation_actor)
    for p in percentiles
        metrics["actor_observations/Precision $(p)% Percentile"] = quantile(vec(precisions_observation_actor), p / 100)
    end

    # Critic observation metrics
    metrics["critic_observations/Mean Tau"] = Statistics.mean(taus_observation_critic)
    for p in percentiles
        metrics["critic_observations/Tau $(p)% Percentile"] = quantile(vec(taus_observation_critic), p / 100)
    end
    metrics["critic_observations/Mean Precision"] = Statistics.mean(precisions_observation_critic)
    for p in percentiles
        metrics["critic_observations/Precision $(p)% Percentile"] = quantile(vec(precisions_observation_critic), p / 100)
    end

    wandb.log(metrics)
end
