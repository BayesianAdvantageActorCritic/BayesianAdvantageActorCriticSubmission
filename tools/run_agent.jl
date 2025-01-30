# ENTER YOUR PATHS HERE
ENV["PYTHON"] = "ENTER_THE_PATH_TO_YOUR_ENVIRONMENT_SPECIFIC_PYTHON_EXECUTABLE_HERE"
ENV["PYTHONPATH"] = "PATH_TO_LOCAL_DIRECTORY_OF_REPO"

# Include necessary code
include("../julia_implementation/bayesian_a2c_fg.jl")
using PyCall
using Serialization
using Plots

function run_agent()    
    # Import the necessary Python modules
    np = pyimport("numpy")
    gym = pyimport("gymnasium")
    
    # Create the environment: Choose the environment you want to test
    env = gym.make("Pendulum-v1", render_mode="human")
    
    # Load the trained agent from disk
    # Please provide the path the the agent you want to test
    agent = deserialize(open("saved_models/best_agent_0.jls", "r"))
    
    # Reset the environment to get the initial observation
    obs_info = env[:reset]()
    observation = obs_info[1]  # Assuming reset returns (observation, info)
    
    done = false
    step_number = 0
    total_reward = 0.0
    step_rewards = Float64[]

    while !done
        step_number += 1

        # Get action from the agent
        observation = Float64.(hcat(observation))

        println("\n")
        println("Observation: ", observation)
        println("Observation Type: ", typeof(observation))
        println("Observation Shape: ", size(observation))
        action, _, _ = get_actions(agent, observation)

        println("Action: ", action)
        println("Action Type: ", typeof(action))
        println("Action Shape: ", size(action))

        # Step the environment
        action = np.array(action)
        action = np.reshape(action, -1)
        result = env[:step](action)

        # Print step number and action
        println("Step $step_number: Action taken: $(action[1]), Reward: $(result[2][1])")

        # Execute env.render()
        env[:render]()

        observation = result[1]  # Next observation
        reward = result[2][1]  # Reward
        done = result[3] || result[4]

        # Accumulate total reward
        total_reward += reward
        push!(step_rewards, reward)
    end

    # Calculate average reward per step
    average_reward = total_reward / step_number

    # Print total reward and average reward
    println("Total reward: $total_reward")
    println("Average reward per step: $average_reward")

    # Plot and save the rewards per step
    plot(step_rewards, label="Reward per Step", xlabel="Step", ylabel="Reward", legend=false)
    savefig("figures/rewards_per_step.pdf")

    # Close the environment
    env[:close]()
end

run_agent()
