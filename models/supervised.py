import os
import numpy as np
import torch

from models.base.policy import BasePolicy
from models.base.network import BaseNetwork
from models.base.memory import BaseMemory
from models.base.optimizer import CEMOptimizer


class Memory(BaseMemory):

    def __init__(self, *args, **kwargs):
        super(Memory, self).__init__(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Converts independent actions and rewards to sequence-based.

        This is used for learning to grasp in the supervised learning case.
        + Action labels for each trial is given by: a_t = p_T - p_t
        + Episode labels for timestep in the episode is the episode reward
        """
        super(Memory, self).load(*args, **kwargs)

        start, end = 0, 1
        while end < self.buffer_size:

            while self.timestep[end] > self.timestep[start]:
                if end >= self.buffer_size - 1:
                    break
                end = end + 1

            # Convert the label for each element to be the straight-line
            # path from the current to final step. Note how we take the
            # cumulative sum from the end of the list and work backwords;
            # it doesn't matter how we got to that point from the forwards
            # direction, we only care about the actions we take from the current
            # state to reach the goal.
            path = np.cumsum(self.action[start:end][::-1], axis=0)
            path /= np.arange(1, end-start+1)[:, np.newaxis]  # normalize
            self.action[start:end] = path[::-1]

            # Set the reward for each timestep as the episode reward
            self.reward[start:end] = self.reward[end - 1]

            start, end = end, end + 1


class Supervised(BasePolicy):

    def __init__(self, config):

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.model = BaseNetwork(**config).to(config['device'])

        self.action_select_eval = CEMOptimizer(**config)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          config['lrate'],
                                          eps=1e-3,
                                          weight_decay=config['decay'])

    def get_weights(self):
        return (self.model.state_dict(),)  # as tuple

    def set_weights(self, weights):
        self.model.load_state_dict(weights[0])

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>'%checkpoint_dir)

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
        return self.action_select_eval(self.model, state, timestep)[0].detach()

    def train(self, memory, batch_size, **kwargs):
        """Performs a single training step."""

        s0, act, r, _, _, timestep = memory.sample(batch_size)

        # The dataset contains more failures then successes, so we'll
        # balance the minibatch loss by weighting it by class frequency
        weight = np.sum(r) / (batch_size - np.sum(r))
        weight = np.where(r == 0, weight, 1).astype(np.float32)
        weight = torch.from_numpy(weight).to(self.device).view(-1)

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        t0 = torch.from_numpy(timestep).to(self.device)

        pred = self.model(s0, t0, act).clamp(1e-8, 1-1e-8).view(-1)

        # Uses the outcome of the episode as individual step label
        loss = torch.nn.BCELoss(weight=weight)(pred, r)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()

        return loss.item()

    def update(self):
        pass
