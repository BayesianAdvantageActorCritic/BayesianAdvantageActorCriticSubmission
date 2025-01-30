import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork:
    def __init__(self, input_shape, output_shape, learning_rate=0.003):
        # Similar architecture to stable-baselines3's default MlpPolicy for the actor:
        # Linear -> Tanh -> Linear -> Tanh -> Linear
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_shape)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.model.train()

        # Fixed standard deviation
        self.fixed_std = 0.05

    def predict(self, states):
        # states: shape (input_shape, number_environments)
        # Returns:
        #   means: shape (output_shape, number_environments)
        #   stds: shape (output_shape, number_environments), fixed to 0.05
        self.model.eval()
        with torch.no_grad():
            # Convert to PyTorch tensor and transpose to (number_environments, input_shape)
            states_t = torch.tensor(states, dtype=torch.float32).T
            means_t = self.model(states_t)  # shape: (number_environments, output_shape)
        self.model.train()
        means = means_t.cpu().numpy().T.astype(np.float64)  # shape: (output_shape, number_environments)
        stds = np.ones_like(means) * self.fixed_std
        return means, stds

    def train_batch(self, states, target_means):
        # states: shape (input_shape, number_environments)
        # target_means: shape (output_shape, number_environments)
        #
        # We train the actor network to predict the target means.
        # Note: We do not train or predict uncertainties; they remain fixed at 0.05.
        states_t = torch.tensor(states, dtype=torch.float32).T  # (number_environments, input_shape)
        target_means_t = torch.tensor(target_means, dtype=torch.float32).T  # (number_environments, output_shape)

        predictions = self.model(states_t)  # (number_environments, output_shape)

        loss = self.loss_fn(predictions, target_means_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
