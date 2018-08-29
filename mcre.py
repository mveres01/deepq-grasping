import os
import copy
import numpy as np
import torch
import torch.optim as optim

from base import BaseNetwork

class MCRE:

    def __init__(self, model, action_size, lrate, decay, device, bounds, **kwargs):

        self.action_size = action_size
        self.device = device
        self.bounds = bounds

        self.model = model

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lrate,
                                    weight_decay=decay)

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>'%checkpoint_dir)
        self.model.load_state_dict(torch.load(checkpoint_dir + '/model.pt'))
        self.update()

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

    def train(self, memory, gamma, batch_size, **kwargs):

        # Sample data from the memory buffer & put on GPU
        loss = 0
        for batch in memory.sample_episode(batch_size):

            s0, act, r, _, _, timestep = batch

            s0 = torch.from_numpy(s0).to(self.device)
            act = torch.from_numpy(act).to(self.device)
            r = torch.from_numpy(r).to(self.device)
            t0 = torch.from_numpy(timestep).to(self.device)

            # Train the models
            pred = self.model(s0, t0, act).view(-1)

            # Since the rewards in this environment are sparse (and only
            # occur at the final timestep of the episode), the loss
            # \sum_{t'=t}^T \gamma^{t' - t} r(s_t, a_t) will always just
            # be the single-step reward
            loss = loss + torch.sum((pred - r) ** 2)

        loss = loss / batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def update(self):
        pass
