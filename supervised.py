import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from base import BaseNetwork

class Supervised:

    def __init__(self, model, action_size, lrate, decay, device, bounds, **kwargs):

        self.action_size = action_size
        self.device = device
        self.bounds = bounds

        self.model = model

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lrate,
                                    betas=(0.95, 0.99),
                                    weight_decay=decay)

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>'%checkpoint_dir)
        self.model.load_state_dict(torch.load(checkpoint_dir + '/model.pt'))

    def save_checkpoint(self, checkpoint_dir):
        """Saves a model to a directory containing a single checkpoint."""

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.model.state_dict(), checkpoint_dir + '/model.pt')

    def sample_action(self, state, timestep, explore_prob):
        """Samples an action to perform in the environment."""

        # Either explore
        if np.random.random() < explore_prob:
            return np.random.uniform(*self.bounds, size=(self.action_size,))

        # Or return the optimal action
        with torch.no_grad():
            return self.model.sample_action(state, timestep)

    def train(self, memory, batch_size, **kwargs):

        # Train the networks
        s0, act, r, _, _, timestep = memory.sample(batch_size, balanced=True)

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        t0 = torch.from_numpy(timestep).to(self.device)

        # Predict a binary outcom
        pred = self.model(s0, t0, act).view(-1)

        # Use the outcome of the episode as the label
        loss = torch.nn.BCELoss(size_average=True)(pred, r)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #return loss.item()

        acc = pred.round_().eq(r.view(-1)).float().mean()
        return acc.item()

    def update(self):
        pass
