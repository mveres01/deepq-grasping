import os
import copy
import numpy as np
import torch

from base import BaseNetwork


class DDQN:

    def __init__(self, config):

        self.model = BaseNetwork(**config).to(config['device'])
        self.target = copy.deepcopy(self.model)
        self.target.eval()

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          config['lrate'],
                                          betas=(0.5, 0.99),
                                          weight_decay=config['decay'])

    def get_weights(self):
        return (self.model.state_dict(), self.target.state_dict())

    def set_weights(self, weights):
        self.model.load_state_dict(weights[0])
        self.target.load_state_dict(weights[1])

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        weights = torch.load(checkpoint_dir + '/model.pt', self.device)
        self.model.load_state_dict(weights)
        self.update()
        
    def save_checkpoint(self, checkpoint_dir):
        """Saves a model to a directory containing a single checkpoint."""

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.model.state_dict(), checkpoint_dir + '/model.pt')

    @torch.no_grad()
    def sample_action(self, state, timestep, explore_prob):
        """Samples an action to perform in the environment."""

        self.model.eval()

        if np.random.random() < explore_prob:
            return np.random.uniform(*self.bounds, size=(self.action_size,))

        return self.model.sample_action(state, timestep)

    def train(self, memory, gamma, batch_size, **kwargs):
        """Performs a single step of Q-Learning."""

        self.model.train()

        # Sample data from the memory buffer & put on GPU
        s0, act, r, s1, done, timestep = memory.sample(batch_size)

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        s1 = torch.from_numpy(s1).to(self.device)
        done = torch.from_numpy(done).to(self.device)
        t0 = torch.from_numpy(timestep).to(self.device)
        t1 = torch.from_numpy(timestep + 1).to(self.device)

        pred = self.model(s0, t0, act).view(-1)

        with torch.no_grad():

            # DDQN works by finding the maximal action for the current policy
            ap = self.model.sample_action(s1, t1, mode='uniform')

            # but using the q-value from the target network
            target = r + (1. - done) * gamma * self.target(s1, t1, ap).view(-1)
    
        loss = torch.mean((pred - target) ** 2).clamp(-1, 1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def update(self):
        """Copy the network weights every few epochs."""
        self.target.load_state_dict(self.model.state_dict())
