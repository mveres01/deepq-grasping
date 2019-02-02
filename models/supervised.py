import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from models.base.policy import BasePolicy
from models.base.network import BaseNetwork
from models.base.memory import BaseMemory
from models.base.optimizer import CEMOptimizer, _preprocess_inputs


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


class SupervisedCEMOptimizer(CEMOptimizer):
    """Implements the cross entropy method for supervised learning.

    In the supervised learning setting, the action taken at each step
    is the (normalized) vector from a_t to a_T, where values lie between
    [-1, 1].

    During data collection, the z-dim was biased to move downwards. A quick
    histogram of z-values shows a roughly bimodal distribution, where
    most values are between [-1, -0.5], with very few values > -0.5. Sampling
    values > -0.5 during eval shows a little bit of instability in the
    Q-function, so we'll prevent against that scenario here as prior
    knowledge.

    See: https://en.wikipedia.org/wiki/Cross-entropy_method
    """

    def __init__(self, zbnd=(-1, -0.5), *args, **kwargs):
        super(SupervisedCEMOptimizer, self).__init__(*args, **kwargs)

        self.zbnd = zbnd

    @torch.no_grad()
    def __call__(self, network, image, timestep):

        network.eval()

        image, timestep = _preprocess_inputs(image, timestep, self.device)

        # We repeat the hidden state representation of the input (rather
        # then the raw input itself) to save some memory
        hstate = network.state_net(image, timestep)

        # (B, N, R, C) -> repeat (B, Cem, N, R, C) -> (B * Cem, N, R, C)
        hstate = hstate.unsqueeze(1) \
                       .repeat(1, self.pop_size, 1, 1, 1) \
                       .view(-1, *hstate.size()[1:])

        mu = torch.zeros(image.size(0), 1, self.action_size, device=self.device)
        mu[0, 0, 2] = -1  # downward bias

        std = torch.ones_like(mu) * 0.5

        for i in range(self.iters):

            # Sample actions from the Gaussian parameterized by (mu, std)
            action = torch.normal(mu.expand(-1, self.pop_size, -1),
                                  std.expand(-1, self.pop_size, -1))\
                                 .clamp(*self.bounds).to(self.device)
            action = action.view(-1, self.action_size)

            # NOTE: Injecting prior here by clamping z bound
            action[:, 2] = action[:, 2].clamp(*self.zbnd)

            # Evaluate the actions using a forward pass through the network
            q = network.qnet(hstate, network.action_net(action))
            q = q.view(-1, self.pop_size)

            # Find the top actions and use them to update the sampling dist
            topq, topk = torch.topk(q, self.elite, dim=1)

            # Book-keeping to extract topk actions for each batch sample
            topk = topk.unsqueeze(2).expand(-1, -1, self.action_size)
            action = action.view(-1, self.pop_size, self.action_size)
            action = torch.gather(action, 1, topk)

            s = max(1. - i / self.iters, 0)

            mu = action.mean(dim=1, keepdim=True).detach()
            std = action.std(dim=1, keepdim=True).detach() + s

        network.train()
        return mu.squeeze(1), topq


class Supervised(BasePolicy):

    def __init__(self, config):

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.model = BaseNetwork(**config).to(config['device'])

        # Note this uses a slightly different variant of the CEM optimizer
        # that restricts sampled z-values to favour datasets z-distribution
        self.action_select_eval = SupervisedCEMOptimizer(**config)

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
