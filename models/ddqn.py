import os
import copy
import numpy as np
import torch

from .base.network import BaseNetwork
from .base.optimizer import CEMOptimizer, UniformOptimizer


class DDQN:

    def __init__(self, config):

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.model = BaseNetwork(**config).to(config['device'])
        self.target = copy.deepcopy(self.model)
        self.target.eval()

        self.action_select_eval = CEMOptimizer(**config)
        self.action_select_train = UniformOptimizer(**config)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          config['lrate'],
                                          eps=1e-3,
                                          weight_decay=config['decay'])

    def get_weights(self):
        return (self.model.state_dict(), self.target.state_dict())

    def set_weights(self, weights):
        self.model.load_state_dict(weights[0])
        self.target.load_state_dict(weights[1])

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>' % checkpoint_dir)

        path = os.path.join(checkpoint_dir, 'model.pt')
        self.model.load_state_dict(torch.load(path, self.device))
        self.update()

    def save_checkpoint(self, checkpoint_dir):
        """Saves a model to a directory containing a single checkpoint."""

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        path = os.path.join(checkpoint_dir, 'model.pt')
        torch.save(self.model.state_dict(), path)

    @torch.no_grad()
    def sample_action(self, state, timestep, explore_prob):
        """Samples an action to perform in the environment."""

        if np.random.random() < explore_prob:
            return np.random.uniform(*self.bounds, size=(self.action_size,))
        return self.action_select_eval(self.model, state, timestep)[0].detach()

    def train(self, memory, gamma, batch_size, **kwargs):
        """Performs a single step of Q-Learning."""

        self.model.train()

        # Sample a minibatch from the memory buffer
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

            # DDQN finds the maximal action for the current policy
            aopt, _ = self.action_select_train(self.model, s1, t1)

            # but uses the q-value from the target network
            target = r + (1. - done) * gamma * self.target(s1, t1, aopt).view(-1)

        loss = torch.mean((pred - target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()

        return loss.item()

    def update(self):
        """Copy the network weights every few epochs."""
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
