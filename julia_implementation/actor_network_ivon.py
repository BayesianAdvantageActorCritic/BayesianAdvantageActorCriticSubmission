# actor_network_ivon.py

import torch
import torch.nn as nn
import numpy as np

# 1) Import IVON
import ivon

class ActorNetwork:
    def __init__(self, input_shape, output_shape, learning_rate=0.1, ess=256000, train_samples=1):
        """
        :param input_shape: dimension of each input observation
        :param output_shape: dimension of the action space
        :param learning_rate: learning rate for the IVON optimizer
        :param ess: effective sample size (a recommended choice is len(dataset), but tune as needed)
        :param train_samples: number of Monte Carlo samples used during training
        """

        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_shape)
        )

        # 2) Use IVON instead of Adam
        self.optimizer = ivon.IVON(self.model.parameters(), lr=learning_rate, ess=ess)

        self.loss_fn = nn.MSELoss()
        self.model.train()

        # For demonstration, keep track of how many samples to draw in training/prediction
        self.train_samples = train_samples

    def predict(self, states, test_samples=10):
        """
        Predict means and (optionally) uncertainties using the variational posterior.
        Uses posterior averaging if test_samples > 1.
        :param states: shape (input_shape, number_environments)
        :param test_samples: number of Monte Carlo samples from posterior
        :return: means, stds arrays of shape (output_shape, number_environments)
        """
        self.model.eval()

        # Convert to PyTorch tensor and transpose to (number_environments, input_shape)
        states_t = torch.tensor(states, dtype=torch.float32).T

        # (A) Single-sample prediction
        if test_samples == 1:
            with torch.no_grad():
                # Sample from the posterior
                with self.optimizer.sampled_params(train=False):
                    means_t = self.model(states_t)
            means = means_t.cpu().numpy().T.astype(np.float64)
            # Return a trivial standard deviation (could also approximate from multiple draws)
            stds = np.full_like(means, 0.05)
            return means, stds

        # (B) Multi-sample (posterior averaging) for more robust uncertainty estimates
        sampled_outputs = []
        with torch.no_grad():
            for _ in range(test_samples):
                # self.optimizer._device = "cpu"
                if not hasattr(self.optimizer, '_device'):
                    print("Object is: ", self.optimizer.__dict__)
                # else:
                #     print("Optimizer device is: ", self.optimizer._device)
                with self.optimizer.sampled_params(train=False):
                    sampled_outputs.append(self.model(states_t).unsqueeze(0))  # shape: (1, N, output_shape)

        # Stack over the first dimension => shape (test_samples, N, output_shape)
        sampled_outputs = torch.cat(sampled_outputs, dim=0)
        # Mean over samples => shape (N, output_shape)
        mean_t = sampled_outputs.mean(dim=0)
        # Std over samples => shape (N, output_shape)
        std_t = sampled_outputs.std(dim=0, unbiased=False)

        means = mean_t.cpu().numpy().T.astype(np.float64)  # shape (output_shape, number_environments)
        stds = std_t.cpu().numpy().T.astype(np.float64)    # shape (output_shape, number_environments)

        return means, stds + 0.05

    def train_batch(self, states, target_means):
        """
        :param states: shape (input_shape, number_environments)
        :param target_means: shape (output_shape, number_environments)
        :return: training loss as float
        """
        # Convert to PyTorch tensors
        states_t = torch.tensor(states, dtype=torch.float32).T   # (number_environments, input_shape)
        target_means_t = torch.tensor(target_means, dtype=torch.float32).T  # (number_environments, output_shape)

        # We do multiple forward-backward passes (Monte Carlo samples).
        losses = []
        for _ in range(self.train_samples):
            with self.optimizer.sampled_params(train=True):
                predictions = self.model(states_t)
                loss = self.loss_fn(predictions, target_means_t)
                self.optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                
        self.optimizer.step()

        # Return average loss across the (train_samples) samples
        return float(np.mean(losses))
