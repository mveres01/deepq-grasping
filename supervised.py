import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from base import BaseNetwork

class Supervised:

    def __init__(self, config):

        self.model = BaseNetwork(**config).to(config['device'])

        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          config['lrate'],
                                          betas=(0.1, 0.6),
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

        # Train the networks
        #0, act, r, _, _, timestep = memory.sample(batch_size, balanced=True)
        s0, act, r, _, _, timestep = memory.sample(batch_size, balanced=True)

        act = act / 15.

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        t0 = torch.from_numpy(timestep).to(self.device)

        # Predict a binary outcom
        pred = self.model(s0, t0, act).view(-1)

        # Use the outcome of the episode as the label
        loss = torch.nn.BCEWithLogitsLoss()(pred, r)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.optimizer.step()

        #return loss.item()
        acc = pred.round_().eq(r.view(-1)).float().mean()
        #print('Loss: %2.4f, Pred: %2.4f, Acc: %2.4f, Ratio: %2.4f'%(loss.item(), pred.mean().item(), acc.item(), r.mean().item()))

        return acc.item()

    def update(self):
        pass
