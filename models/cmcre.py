import os
import copy
import numpy as np
import torch
import torch.optim as optim

from .base.network import BaseNetwork
from .base.memory import BaseMemory
from .base.optimizer import CEMOptimizer, UniformOptimizer


class Memory(BaseMemory):

    def __init__(self, *args, **kwargs):
        super(Memory, self).__init__(*args, **kwargs)

    def sample(self, batch_size, batch_idx=None):
        """Samples grasping episodes rather then single timesteps."""

        # Find where each episode in the memory buffer starts
        starts = np.where(self.timestep == 0)[0]

        # Pick :batch_size: random episodes
        batch_idx = np.random.choice(len(starts) - 1, batch_size, replace=False)

        # We've already set the value to be discounted
        indices = [np.arange(starts[i], starts[i+1]) for i in batch_idx]
        indices = np.hstack(indices)

        return self[indices]


class CMCRE:

    def __init__(self, config):

        self.model = BaseNetwork(**config).to(config['device'])

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.cem = CEMOptimizer(**config)
        self.uniform = UniformOptimizer(**config)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          config['lrate'],
                                          eps=1e-3,
                                          weight_decay=config['decay'])

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>' % checkpoint_dir)

        path = os.path.join(checkpoint_dir, 'model.pt')
        self.model.load_state_dict(torch.load(path, self.device))

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

        return self.cem(self.model, state, timestep)[0].detach()

    def _loss(self, Q, pred, r, gamma):
        """Calculates corrected loss over a single episode.

        Assumes that all inputs (Q, pred, r) belong to a single episode 
        only. These are obtained by slicing the input at each timestep == 0.
        """

        # As we sample a full episode, we can just take the difference 
        # between consecutive value predictions as the advantage
        adv = r.clone() - Q  # set the last element
        adv[:-1] = r[:-1] + gamma * Q[1:] - Q[:-1]

        out = torch.zeros_like(r)
        for i in reversed(range(r.shape[0] - 1)):
            out[i] = gamma * (out[i+1] + (r[i+1] - adv[i+1]))
        
        out = (out + r).detach()

        # Later normalize over batch size
        loss = ((pred - out) ** 2).sum()

        return loss

    def train(self, memory, gamma, batch_size, **kwargs):

        # Sample full episodes from memory
        s0, act, r, _, _, timestep = memory.sample(batch_size // 8)
    
        # Used to help compute proper loss per episode
        starts = np.hstack((np.where(timestep == 0)[0], r.shape[0]))

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        t0 = torch.from_numpy(timestep).to(self.device)

        pred = self.model(s0, t0, act).view(-1)
   
        # Calculate V* = \max_a Q(s_t, a)
        _, Q = self.uniform(self.model, s0, t0)
   
        # Sum the loss for each of the episodes
        loss = 0
        for s, e in zip(starts[:-1], starts[1:]):
            loss = loss + self._loss(Q[s:e], pred[s:e], r[s:e], gamma)

        loss = loss / s0.shape[0]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()

        return loss.item()

    def update(self):
        pass
