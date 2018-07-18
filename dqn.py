import os
import copy
import time
import numpy as np
import torch
import torch.optim as optim

from agent import BaseNetwork


class DQN:
    
    def __init__(self, num_features, decay, lrate, num_uniform, num_cem,
                 cem_iter, cem_elite, checkpoint, action_size, in_channels, 
                 device, **kwargs):

        self.device = device

        self.model = BaseNetwork(in_channels, num_features, action_size, 
                                 num_uniform, num_cem, cem_iter, cem_elite)\
                                .to(device)
        self.target = copy.deepcopy(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lrate, weight_decay=decay)
        
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, 0.999)

    def load_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            raise Exception('No checkpoint directory <%s>'%checkpoint_dir)
        self.model.load_state_dict(torch.load(checkpoint_dir + '/model.pt'))

    def save_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.model.state_dict(), checkpoint_dir + '/model.pt')

    def sample_action(self, state, timestep, explore_prob):
        return self.model.sample_action(state, timestep, explore_prob)

    def train(self, memory, gamma, batch_size, **kwargs):

        # Sample data from the memory buffer & put on GPU
        s0, act, r, s1, term, timestep = memory.sample(batch_size)

        s0 = torch.from_numpy(s0).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        s1 = torch.from_numpy(s1).to(self.device)
        term = torch.from_numpy(term).to(self.device)

        t0 = torch.from_numpy(timestep).to(self.device)
        t1 = torch.from_numpy(timestep + 1).to(self.device)

        # Train the models
        pred = self.model(s0, t0, act).view(-1)  # saved action

        # We don't calculate a gradient for the target network; these
        # weights instead get updated by copying the prediction network
        # weights every few training iterations
        with torch.no_grad():
            label = r + (1. - term) * gamma * self.target(s1, t1).view(-1)

        loss = torch.sum((pred - label) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def update(self):
        self.target = copy.deepcopy(self.model)
        #self.scheduler.step()
