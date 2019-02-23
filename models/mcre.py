import os
import numpy as np
import torch
from models.base.policy import BasePolicy
from models.base.network import BaseNetwork
from models.base.memory import BaseMemory
from models.base.optimizer import CEMOptimizer, UniformOptimizer


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


class MCRE(BasePolicy):

    def __init__(self, config):

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.model = BaseNetwork(**config).to(config['device'])

        self.cem = CEMOptimizer(**config)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          config['lrate'],
                                          eps=1e-3,
                                          weight_decay=config['decay'])

    def get_weights(self):
        return (self.model.state_dict(),)

    def set_weights(self, weights):
        self.model.load_state_dict(weights[0])

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>'%checkpoint_dir)

        weights = torch.load(os.path.join(checkpoint_dir, 'model.pt'), self.device)
        self.model.load_state_dict(weights)

    def save_checkpoint(self, checkpoint_dir):
        """Saves a model to a directory containing a single checkpoint."""

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        path = os.path.join(checkpoint_dir, 'model.pt')
        torch.save(self.model.state_dict(), path)

    @torch.no_grad()
    def sample_action(self, state, timestep, explore_prob):
        """Samples an action to perform using CEM."""

        if np.random.random() < explore_prob:
            return np.random.uniform(*self.bounds, size=(self.action_size,))
        return self.cem(self.model, state, timestep)[0].detach()

    def train(self, memory, gamma, batch_size, **kwargs):

        del gamma  # unused

        # Sample a minibatch from the memory buffer. Note that we sample
        # full grasping episodes in this method, so the output of
        # memory.sample will be episode_length * num_episodes
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
