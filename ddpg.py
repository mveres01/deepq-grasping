import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.network import StateNetwork, BaseNetwork


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
        out = F.tanh(self.fc3(out))
        return out


class DDPG:

    def __init__(self, config):

        # Needed for sampling actions
        self.action_size = config['action_size']
        self.device = config['device']
        self.bounds = config['bounds']

        self.critic = BaseNetwork(**config).to(config['device'])
        self.critic_target = copy.deepcopy(self.critic)

        self.model = Actor(config['out_channels'], 
                           config['action_size']).to(config['device'])
        self.model_target = copy.deepcopy(self.model)

        self.critic_target.eval()
        self.model_target.eval()

        self.aopt = torch.optim.Adam(self.model.parameters(),
                                     config['lrate'],
                                     betas=(0.5, 0.99),
                                     weight_decay=config['decay'])

        self.copt = torch.optim.Adam(self.critic.parameters(),
                                     config['lrate'],
                                     betas=(0.5, 0.99),
                                     weight_decay=config['decay'])

        self.optimizer = self.aopt

    def get_weights(self):

        return (self.model.state_dict(),
                self.critic.state_dict(),
                self.model_target.state_dict(),
                self.critic_target.state_dict())

    def set_weights(self, weights):

        self.model.load_state_dict(weights[0])
        self.critic.load_state_dict(weights[1])
        self.model_target.load_state_dict(weights[2])
        self.critic_target.load_state_dict(weights[3])

    def load_checkpoint(self, checkpoint_dir):

        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>' % checkpoint_dir)

        weights = torch.load(checkpoint_dir + '/actor.pt', self.device)
        self.model.load_state_dict(weights)

        weights = torch.load(checkpoint_dir + '/critic.pt', self.device)
        self.critic.load_state_dict(weights)
        self.update()

    def save_checkpoint(self, checkpoint_dir):

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(self.model.state_dict(), checkpoint_dir + '/actor.pt')
        torch.save(self.critic.state_dict(), checkpoint_dir + '/critic.pt')

    @torch.no_grad()
    def sample_action(self, state, timestep, explore_prob):

        if np.random.random() < explore_prob:
            return np.random.uniform(-1, 1, self.action_size)

        self.model.eval()

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], device=self.device)

        return self.model(state, timestep).detach()

    def train(self, memory, gamma, batch_size, **kwargs):

        self.model.train()

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

            at = self.model_target(s1, t1)
            qt = self.critic_target(s1, t1, at).view(-1)
            target = r + (1. - term) * gamma * qt

        loss = torch.mean((pred - target) ** 2).clamp(-1, 1)

        self.aopt.zero_grad()
        self.copt.zero_grad()
        loss.backward()
        self.copt.step()
        self.copt.zero_grad()

        # Update the actor network by following the policy gradient
        q_pred = -self.critic(s0, t0, self.model(s0, t0)).mean().clamp(-1, 1)

        self.aopt.zero_grad()
        q_pred.backward()
        self.aopt.step()
        self.aopt.zero_grad()
        self.copt.zero_grad()

        return loss.item()

    def update(self):

        self.model_target.load_state_dict(self.model.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
