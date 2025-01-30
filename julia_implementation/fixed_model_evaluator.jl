# file: fixed_model_evaluator.jl
using PyCall
using Serialization
using Statistics
using Dates
using Plots

# We can keep the same function for min, median, max
function compute_min_med_max(mat)
    return (
        Statistics.minimum(mat, dims=2)[:],
        Statistics.median(mat,  dims=2)[:],
        Statistics.maximum(mat, dims=2)[:]
    )
end

"""
    evaluate_fixed_model(env_name::String, dummy::Vector{Float64})

Loads each available checkpointed model (based on the known checkpoint list)
for the current process, evaluates it in a fresh environment for 10,000 steps 
with no training (just running the model's policy), and returns a vector of 
average rewards per step (one entry per checkpoint).

`dummy` is unused in this minimal example; it's there only because 
the pipeline passes `rewards_list[i]` from the training.
"""
function evaluate_fixed_model(env_name::String, experiment_name::String, model_checkpoints::Vector{Int})
    process_idx = myid()

    avg_rewards = Float64[]

    # For each checkpoint, we try to find a matching file in any subfolder
    for checkpoint in model_checkpoints
        path = "saved_models/$(experiment_name)/model_checkpoint_$(checkpoint)_$(process_idx).jls"
        avg_reward_for_this_checkpoint = 0.0

        if isfile(path)
            # Found the file, load the agent
            agent = nothing
            open(path, "r") do io
                agent = deserialize(io)
            end
            # Evaluate the agent for 10,000 steps
            avg_reward_for_this_checkpoint = evaluate_agent_once(env_name, agent, 10_000)
            push!(avg_rewards, avg_reward_for_this_checkpoint)
        end
    end

    println("The vector of average rewards for process $(process_idx) is: ", avg_rewards)

    return avg_rewards
end

"""
    evaluate_agent_once(env_name, agent, num_steps)

Creates a fresh environment for `env_name`, runs `num_steps` steps 
(or until done repeatedly, with environment resets), using the loaded 
BayesianA2CAgent's actor to pick actions (no training). 
Returns the average reward per step.
"""
function evaluate_agent_once(env_name::String, agent, num_steps::Int)
    gym = pyimport("gymnasium")
    np = pyimport("numpy")

    # Make a fresh environment
    env = gym.make(env_name)
    obs_reset = env[:reset]()  # (obs, info)
    obs = obs_reset[1]

    total_reward = 0.0

    for _ in 1:num_steps
        obs = Float64.(Matrix(hcat(obs...)))'
        action, _, _ = get_actions(agent, obs; use_expansion=false)

        # println("Action: ", action)
        # println("Action Type: ", typeof(action))
        # println("Action Shape: ", size(action))
        action = np.array(action)
        # Reshape it to one dimension
        action = np.reshape(action, -1)

        result = env[:step](action)
        new_obs  = result[1]
        reward   = result[2][1][1]
        done1    = result[3]
        done2    = result[4]

        total_reward += reward
        obs = new_obs

        if done1 || done2
            # reset
            obs_reset = env[:reset]()
            obs = obs_reset[1]
        end
    end

    return total_reward / num_steps
end
