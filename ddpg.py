import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent import BaseNetwork, StateNetwork


class Actor(nn.Module):
    def __init__(self, out_channels, action_size):
        super(Actor, self).__init__()

        self.state_net = StateNetwork(out_channels)
        self.fc1 = nn.Linear(7 * 7 * (out_channels + 1), out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.fc3 = nn.Linear(out_channels, action_size)

        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, image, cur_time):

        out = self.state_net(image, cur_time)
        out = out.view(image.shape[0], -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.tanh(self.fc3(out)) # Constrain actions to [-1, 1]
        return out


class DDPG:

    def __init__(self, num_features, decay, lrate, num_uniform, num_cem,
                 cem_iter, cem_elite, checkpoint, action_size, device, **kwargs):

        self.device = device

        self.actor = Actor(num_features, action_size).to(device)

        self.critic = BaseNetwork(num_features, action_size,
                                  num_uniform, num_cem, cem_iter, cem_elite)\
                                 .to(device)

        self.atarget = copy.deepcopy(self.actor)
        self.ctarget = copy.deepcopy(self.critic)

        self.aoptimizer = optim.Adam(self.actor.parameters(), lrate,
                                     weight_decay=decay)
        self.coptimizer = optim.Adam(self.critic.parameters(), lrate,
                                     weight_decay=decay)

    def load_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>'%checkpoint_dir)
        self.actor.load_state_dict(torch.load(checkpoint_dir + '/actor.pt'))
        self.critic.load_state_dict(torch.load(checkpoint_dir + '/critic.pt'))

    def save_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.actor.state_dict(), checkpoint_dir + '/actor.pt')
        torch.save(self.critic.state_dict(), checkpoint_dir + '/critic.pt')

    def sample_action(self, state, timestep, explore_prob):
        with torch.no_grad():
            if np.random.random() < explore_prob:
                return np.random.uniform(-1, 1)

            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).to(self.device)
            if isinstance(timestep, float):
                timestep = torch.tensor([timestep], device=self.device)

            return self.actor(state, timestep).cpu().numpy().flatten()

    def train(self, memory, gamma, batch_size, **kwargs):

        s0, act, r, s1, term, timestep = memory.sample(batch_size)

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        s1 = torch.from_numpy(s1).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        term = torch.from_numpy(term).to(self.device)

        t0 = torch.from_numpy(timestep).to(self.device)
        t1 = torch.from_numpy(timestep + 1.).to(self.device)

        # Train the models
        pred = self.critic(s0, t0, act).view(-1)

        # Training the critic is through Mean Squared Error
        with torch.no_grad():

            at = self.atarget(s1, t1)
            qt = self.ctarget(s1, t1, at).view(-1)
            target = r + (1. - term) * gamma * qt

        loss = torch.sum((pred - target) ** 2)

        self.coptimizer.zero_grad()
        loss.backward()
        self.coptimizer.step()
        self.coptimizer.zero_grad()

        # Update the actor network by following the policy gradient
        q_pred = -self.critic(s0, t0, self.actor(s0, t0)).sum()

        self.aoptimizer.zero_grad()
        q_pred.backward()
        self.aoptimizer.step()
        self.aoptimizer.zero_grad()

        return loss.detach()

    def update(self):
        self.atarget = copy.deepcopy(self.actor)
        self.ctarget = copy.deepcopy(self.critic)

