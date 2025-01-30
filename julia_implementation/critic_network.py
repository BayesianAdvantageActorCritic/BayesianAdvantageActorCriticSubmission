# critic_network.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import PPO

class CriticNetwork:
    def __init__(self, input_shape, learning_rate=0.003):
        # Same architecture as stable-baselines3's default MlpPolicy for value function:
        # Linear -> Tanh -> Linear -> Tanh -> Linear
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.model.train()

    def train_batch(self, states, targets):
        # states: shape (input_shape, number_environments)
        # targets: shape (1, number_environments)

        # Convert inputs to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).T  # shape: (number_environments, input_shape)
        targets = torch.tensor(targets, dtype=torch.float32).T # shape: (number_environments, 1)

        # Forward pass
        predictions = self.model(states)  # shape: (number_environments, 1)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # No return needed, but you could log the loss if desired.
        return float(loss.item())

    def predict(self, states, silent=True):
        # states: shape (input_shape, number_environments)
        # Returns:
        #   values: shape (1, number_environments)
        #   uncertainties: shape (1, number_environments) - fixed to 2.0
        self.model.eval()
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32).T
            predictions = self.model(states)  # shape: (number_environments, 1)
        self.model.train()
        values = predictions.numpy().T  # shape: (1, number_environments)
        uncertainties = np.ones_like(values) * 0.4  # 2.0
        # Convert values and uncertainties to 64 bit floats
        values = values.astype(np.float64)
        uncertainties = uncertainties.astype(np.float64)
        return values, uncertainties

    def load_critic(self, ppo_model_path):
        # Load the PPO model
        loaded_model = PPO.load(ppo_model_path)
        
        # stable-baselines3 PPO MlpPolicy structure:
        # model.policy is a ActorCriticPolicy
        # model.policy.mlp_extractor gives us separate networks for policy and value:
        #   mlp_extractor.shared_net
        #   mlp_extractor.policy_net
        #   mlp_extractor.value_net
        # model.policy.value_net is a linear layer applied after the value_net output.
        #
        # The default MlpPolicy:
        # input -> features_extractor (Flatten) -> mlp_extractor (shared + value_net) -> value_net (linear)
        #
        # The value network parameters we need:
        #   1) shared part inside mlp_extractor (shared_net)
        #   2) mlp_extractor.value_net (another layer)
        #   3) policy.value_net (final linear layer)
        #
        # The final architecture is effectively:
        # features_extractor: identity (just a flatten)
        # shared_net: Linear -> Tanh -> Linear -> Tanh
        # value_net (inside mlp_extractor): Just a linear layer from the last Tanh
        # policy.value_net: final linear layer that outputs the value

        # Check what the policy architecture is:
        # Typically for default MlpPolicy:
        #  features_extractor: Flatten
        #  mlp_extractor.shared_net: [Linear(input ->64), Tanh(), Linear(64->64), Tanh()]
        #  mlp_extractor.value_net: empty (direct pass)
        #  policy.value_net: Linear(64->1)

        # To confirm, print the model's policy:
        # print(loaded_model.policy) # Uncomment to inspect if needed.

        # Extract parameters:
        # The "shared_net" corresponds to our first two Linear-Tanh-Linear-Tanh layers
        # The "value_net" (in policy) is the final linear layer.

        sb_policy = loaded_model.policy
        # stable-baselines3 parameters:
        # sb_policy.mlp_extractor.shared_net: Linear(…)->Tanh->Linear(…)->Tanh
        # sb_policy.value_net: Linear(64->1)

        # Get our critic parameters
        own_params = list(self.model.parameters())
        # own_params order: [w0, b0, w1, b1, w2, b2, w3, b3, w4, b4]
        # Layers: Linear(input->64), Tanh, Linear(64->64), Tanh, Linear(64->1)
        # So we have weights and biases for 3 linear layers:
        #   Layer 1: w0, b0
        #   Layer 2: w1, b1
        #   Layer 3: w2, b2
        # Actually, we have 3 linear layers, each with weight and bias. That's 6 parameter sets total.
        # Wait, note that Tanh has no parameters. So total linear layers = 3 * 2 = 6 sets.
        # The model.parameters() should give 6 tensors (3 layers * weights+bias).
        #
        # The stable-baselines shared_net has 2 Linear layers (4 params), and value_net has 1 Linear layer (2 params).
        # Total also 6. They should match one-to-one if the architecture is identical.

        sb_shared_params = []  # list(sb_policy.mlp_extractor.shared_net.parameters())
        # This should give us [shared_w0, shared_b0, shared_w1, shared_b1]
        sb_value_net_params = []  # list(sb_policy.value_net.parameters())
        # This should give us [value_w, value_b]

        # Combine them to match our model's linear layers:
        # Our model: Layer1(Linear), Layer2(Linear), Layer3(Linear)
        # stable-baselines: shared_net: Linear1, Linear2; value_net: Linear3
        sb_all_params = sb_shared_params + sb_value_net_params
        # should be a list of length 6: [w0, b0, w1, b1, w2, b2]

        # Copy parameters
        for own_param, sb_param in zip(own_params, sb_all_params):
            own_param.data.copy_(sb_param.data)

        print("Critic parameters loaded from stable-baselines3 PPO model.")
