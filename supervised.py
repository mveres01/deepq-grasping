import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from base.network import BaseNetwork
from base.memory import BaseMemory


class Memory(BaseMemory):

    def __init__(self, *args, **kwargs):
        super(Memory, self).__init__(self, *args, **kwargs)

   def load(self, data_dir, buffer_size=100000, **kwargs):
        """Converts independent actions and rewards to sequence-based.

        This is used for learning to grasp in the supervised learning case.
        + Action labels for each trial is given by: a_t = p_T - p_t
        + Episode labels for timestep in the episode is the episode reward
        """
        super(Memory, self).load(data_dir, buffer_size)

        start, end = 0, 1

        while end < self.state.shape[0]:

            while self.timestep[end] > self.timestep[start]:
                if end >= self.state.shape[0] - 1:
                    break
                end = end + 1

            # Convert the label for each element to be the straight-line
            # path from the action at the current time step & final action.
            # As each action is represented by [dx, dy, dz, d_theta], we can
            # just sum the actions for the episode.
            path = np.cumsum(self.action[start:end], axis=0)
            self.action[start:end] = path[::-1]

            # Normalize each action by number of steps remaining in episode
            self.action[start:end] /= np.arange(1, end-start+1)[::-1][:, np.newaxis]

            # Set the reward for each timestep as the episode reward
            self.reward[start:end] = self.reward[end - 1]

            start, end = end, end + 1


class Supervised:

    def __init__(self, config):

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.model = BaseNetwork(**config).to(config['device'])

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
        self.model.load_state_dict(torch.load(checkpoint_dir + '/model.pt', self.device))

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

    def train(self, memory, batch_size, **kwargs):
        """Performs a single training step."""

        s0, act, r, _, _, timestep = memory.sample(batch_size, balanced=True)

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        t0 = torch.from_numpy(timestep).to(self.device)

        pred = self.model(s0, t0, act).clamp(1e-10, 1-1e-10)

        # Uses the outcome of the episode as individual step label
        loss = torch.nn.BCELoss()(pred, r)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()

        return loss.item()

    def update(self):
        pass
