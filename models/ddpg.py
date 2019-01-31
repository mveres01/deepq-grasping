import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.network import StateNetwork, BaseNetwork
from base.policy import BasePolicy


class Actor(nn.Module):
    """Defines the actor model that learns to predict actions."""

    def __init__(self, out_channels, action_size, **kwargs):
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
        out = F.tanh(self.fc3(out))
        return out


class DDPG(BasePolicy):

    def __init__(self, config):

        # Needed for sampling actions
        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.critic = BaseNetwork(**config).to(config['device'])
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        self.actor = Actor(**config).to(config['device'])
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()

        self.aopt = torch.optim.Adam(self.actor.parameters(),
                                     config['lrate'],
                                     eps=1e-3,
                                     weight_decay=config['decay'])

        self.copt = torch.optim.Adam(self.critic.parameters(),
                                     config['lrate'],
                                     eps=1e-3,
                                     weight_decay=config['decay'])

    def get_weights(self):
        return (self.actor.state_dict(),
                self.critic.state_dict(),
                self.actor_target.state_dict(),
                self.critic_target.state_dict())

    def set_weights(self, weights):
        self.actor.load_state_dict(weights[0])
        self.critic.load_state_dict(weights[1])
        self.actor_target.load_state_dict(weights[2])
        self.critic_target.load_state_dict(weights[3])

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model from a directory containing a checkpoint."""

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>' % checkpoint_dir)

        path = os.path.join(checkpoint_dir, 'actor.pt')
        self.actor.load_state_dict(torch.load(path, self.device))

        path = os.path.join(checkpoint_dir, 'critic.pt')
        self.critic.load_state_dict(torch.load(path, self.device))
        self.update()

    def save_checkpoint(self, checkpoint_dir):
        """Saves a model to a directory containing a single checkpoint."""

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        path = os.path.join(checkpoint_dir, 'actor.pt')
        torch.save(self.actor.state_dict(), path)

        path = os.path.join(checkpoint_dir, 'critic.pt')
        torch.save(self.critic.state_dict(), path)

    @torch.no_grad()
    def sample_action(self, state, timestep, explore_prob):
        """Samples an action to perform in the environment."""

        if np.random.random() < explore_prob:
            return np.random.uniform(-1, 1, self.action_size)

        self.actor.eval()

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], device=self.device)

        return self.actor(state, timestep).detach()

    def train(self, memory, gamma, batch_size, **kwargs):

        self.actor.train()

        s0, act, r, s1, term, timestep = memory.sample(batch_size)

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        s1 = torch.from_numpy(s1).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        term = torch.from_numpy(term).to(self.device)

        t0 = torch.from_numpy(timestep).to(self.device)
        t1 = torch.from_numpy(timestep + 1.).to(self.device)

        # Train the critic
        pred = self.critic(s0, t0, act).view(-1)

        with torch.no_grad():

            at = self.actor_target(s1, t1)
            qt = self.critic_target(s1, t1, at).view(-1)
            target = r + (1. - term) * gamma * qt

        loss = torch.mean((pred - target) ** 2)#.clamp(-1, 1)

        self.aopt.zero_grad()
        self.copt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.)
        self.copt.step()
        self.copt.zero_grad()

        # Train the actor by following the policy gradient
        self.aopt.zero_grad()

        action = self.actor(s0, t0)
        q_pred = -self.critic(s0, t0, action).mean()

        q_grad = torch.autograd.grad(q_pred, action)[0]

        action.backward(gradient=q_grad)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.aopt.step()

        return loss.item()

    def update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
