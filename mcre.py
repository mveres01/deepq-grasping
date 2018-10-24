import os
import copy
import numpy as np
import torch
import torch.optim as optim

from base.network import BaseNetwork
from base.memory import BaseMemory


class Memory(BaseMemory):

    def __init__(self, *args, **kwargs):
        super(Memory, self).__init__(*args, **kwargs)

    def load(self, data_dir, buffer_size=100000, gamma=0.92, **kwargs):
        """Modifies the reward of loaded data to be discounted future sum.

        The off-policy version of MCRE (this file) returns an entire episode
        of experience at a time when Memory.sample() is called. As this is a
        sparse reward task, we'll initialize the reward to be discounted so
        it won't have to be re-computed every time on the fly.
        """
        super(Memory, self).load(data_dir, buffer_size)

        # Find where episodes start and end, and modify them based on
        # the discount rate gamma
        start, end = 0, 1
        while end < self.state.shape[0]:

            while self.timestep[end] > self.timestep[start]:
                if end >= self.state.shape[0] - 1:
                    break
                end = end + 1

            if self.reward[end - 1] == 1:
                self.reward[start:end] = gamma ** np.arange(end-start)[::-1]

            start, end = end, end + 1

    def sample(self, batch_size, batch_idx=None):
        """Samples grasping episodes rather then single timesteps.

        This function will sample :batch_size: episodes from memory,
        and return a generator that can be used to iterate over all
        sampled episodes one-by-one. This allows us to doo messy
        specification for uneven episode lengths
        """

        # Find where each episode in the memory buffer starts
        starts = np.where(self.timestep == 0)[0]

        # Pick :batch_size: random episodes
        batch_idx = np.random.choice(len(starts) - 1, batch_size, replace=False)

        # We've already set the value to be discounted
        indices = [np.arange(starts[i], starts[i+1]) for i in batch_idx]
        indices = np.hstack(indices)

        return self[indices]


class MCRE:

    def __init__(self, config):

        self.model = BaseNetwork(**config).to(config['device'])

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          config['lrate'],
                                          betas=(0.5, 0.99),
                                          weight_decay=config['decay'])

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>'%checkpoint_dir)
        self.model.load_state_dict(torch.load(checkpoint_dir + '/model.pt',
                                              self.device))

    def save_checkpoint(self, checkpoint_dir):
        """Saves a model to a directory containing a single checkpoint."""

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.model.state_dict(), checkpoint_dir + '/model.pt')

    @torch.no_grad()
    def sample_action(self, state, timestep, explore_prob):
        """Samples an action to perform in the environment."""

        if np.random.random() < explore_prob:
            return np.random.uniform(*self.bounds, size=(self.action_size,))

        return self.model.sample_action(state, timestep)

    def train(self, memory, gamma, batch_size, **kwargs):

        # Sample data from the memory buffer & put on GPU
        s0, act, r, _, _, timestep = memory.sample(batch_size // 8)

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        t0 = torch.from_numpy(timestep).to(self.device)

        pred = self.model(s0, t0, act).view(-1)

        # Note that the reward 'r' has been discounted in memory.load
        loss = torch.mean((pred - r) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()

        return loss.item()

    def update(self):
        pass
