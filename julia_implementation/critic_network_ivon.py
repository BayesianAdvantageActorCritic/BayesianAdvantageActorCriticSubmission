# critic_network_ivon.py
import torch
import torch.nn as nn
import numpy as np

# --- NEW: import ivon instead of torch.optim
import ivon  # <--- CHANGED

from stable_baselines3 import PPO

class CriticNetwork:
    def __init__(self, input_shape, learning_rate=0.1):
        # Same architecture as stable-baselines3's default MlpPolicy for value function:
        # Linear -> Tanh -> Linear -> Tanh -> Linear
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Replace Adam with IVON:
        #   'ess' = approximate number of datapoints, adjust to your use case.
        self.optimizer = ivon.IVON(self.model.parameters(), lr=learning_rate, ess=256000)  # <--- CHANGED
        self.loss_fn = nn.MSELoss()
        self.model.train()

    def train_batch(self, states, targets):
        # states: shape (input_shape, number_environments)
        # targets: shape (1, number_environments)

        # Convert inputs to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).T  # shape: (number_environments, input_shape)
        targets = torch.tensor(targets, dtype=torch.float32).T # shape: (number_environments, 1)

        # For IVON, we typically sample the weights (with MC samples = 1 for most use cases).
        # If you want more MC samples, increase train_samples.
        train_samples = 10  # <--- CHANGED (for clarity; adjust if desired)

        for _ in range(train_samples):
            # Sample parameters for the forward pass
            with self.optimizer.sampled_params(train=True):  # <--- CHANGED
                predictions = self.model(states)
                loss = self.loss_fn(predictions, targets)

                self.optimizer.zero_grad()
                loss.backward()

        self.optimizer.step()  # <--- CHANGED: call outside 'with' block

        return float(loss.item())

    def predict(self, states, silent=True, test_samples=10):  # <--- CHANGED: added test_samples
        """
        We draw multiple samples from the IVON posterior to estimate both the mean prediction
        and its uncertainty. The final 'values' is the mean prediction across samples and
        'uncertainties' is the standard deviation of the predictions.
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32).T  # shape: (envs, input_shape)
            sampled_preds = []

            # Draw multiple forward passes from the variational posterior
            for _ in range(test_samples):  # <--- CHANGED
                with self.optimizer.sampled_params():
                    sampled_preds.append(self.model(states))  # (envs, 1)

            # Stack all predictions to compute mean & std
            preds_stacked = torch.stack(sampled_preds, dim=0)  # shape: (test_samples, envs, 1)
            mean_preds = preds_stacked.mean(dim=0)  # (envs, 1)
            std_preds = preds_stacked.std(dim=0)    # (envs, 1)

        self.model.train()
        values = mean_preds.numpy().T         # shape: (1, envs)
        uncertainties = std_preds.numpy().T   # shape: (1, envs)

        # Convert to 64-bit floats if needed
        values = values.astype(np.float64)
        uncertainties = uncertainties.astype(np.float64)
        return values, uncertainties + 0.4
