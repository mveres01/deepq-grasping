import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StateNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(StateNetwork, self).__init__()

        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=0),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.net = net

    def forward(self, image, timestep):
        """Concatenate the timestep with processed image representation."""

        out = self.net(image)

        timestep = timestep.view(-1, 1, 1, 1)
        timestep = timestep.expand(-1, 1, out.size(2), out.size(3))

        out = torch.cat((out, timestep), dim=1)
        return out


class ActionNetwork(nn.Module):
    def __init__(self, action_size, num_outputs):
        super(ActionNetwork, self).__init__()
        self.fc1 = nn.Linear(action_size, num_outputs)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        return out


class Agent(nn.Module):
    def __init__(self, in_channels, out_channels, action_size, num_random,
                 num_cem, cem_iter, cem_elite, use_cuda=False):
        super(Agent, self).__init__()

        self.action_size = action_size
        self.num_random = num_random
        self.num_cem = num_cem
        self.cem_iter = cem_iter
        self.cem_elite = cem_elite
        self.use_cuda = use_cuda

        self.state_net = StateNetwork(in_channels, out_channels)
        self.action_net = ActionNetwork(action_size, out_channels + 1)

        self.fc1 = nn.Linear(7 * 7 * (out_channels + 1), out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.fc3 = nn.Linear(out_channels, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform(p)

    def _cross_entropy_method(self, hidden):
        """See: https://en.wikipedia.org/wiki/Cross-entropy_method

        This method is only run in evaluation mode, so batch size will always=1.
        """

        hidden_var = Variable(hidden.data, requires_grad=False)
        hidden_var = hidden_var.expand(self.num_cem, -1, -1, -1)

        if self.use_cuda:
            mu = torch.cuda.FloatTensor(1, self.action_size).fill_(0)
            std = torch.cuda.FloatTensor(1, self.action_size).fill_(1)
        else:
            mu = torch.FloatTensor(1, self.action_size).fill_(0)
            std = torch.FloatTensor(1, self.action_size).fill_(1)

        size = (self.num_cem, self.action_size)

        for _ in range(self.cem_iter):

            # Sample a few actions from the Gaussian parameterized by (mu, var)
            action = torch.normal(mu.expand(*size), std.expand(*size))
            action = action.clamp(-1, 1)

            if self.use_cuda:
                action = action.cuda()
            action_var = Variable(action, requires_grad=False)

            # Calculate the Q value for the generated actions; we use the top
            # "k" elite samples to recompute actions on subsequent iters
            q = self._forward_q(hidden_var, action_var).view(-1)
            _, top_k = torch.topk(q, self.cem_elite)

            top_actions = action_var[top_k].data

            # Update the sampling distribution for next action selection
            mu = torch.mean(top_actions, dim=0, keepdim=True)
            std = torch.std(top_actions, dim=0, keepdim=True)

        return mu

    def _random_uniform(self, hidden):
        """Samples random actions from [-1, 1], returning those with highest Q."""

        # Pre-compute the samples we want to try, so we don't generate each time
        action_size = (hidden.size(0), self.action_size)
        optim_size = (hidden.size(0), self.num_random, self.action_size)

        if self.use_cuda:
            actions = torch.cuda.FloatTensor(*action_size).fill_(0)
            uniform = torch.cuda.FloatTensor(*optim_size).uniform_(-1, 1)
        else:
            actions = torch.FloatTensor(*action_size).fill_(0)
            uniform = torch.FloatTensor(*optim_size).uniform_(-1, 1)

        # Calculate the optimal action for each sample in the batch.
        for i in range(hidden.size(0)):

            hidden_var = Variable(hidden[i:i + 1].data, requires_grad=False)
            hidden_var = hidden_var.expand(self.num_random, -1, -1, -1)

            action_var = Variable(uniform[i], requires_grad=False)

            _, top1 = self._forward_q(hidden_var, action_var).max(0)

            actions[i] = action_var[top1.data].data

        return actions

    def _forward_q(self, hidden, action):
        """Computes the Q-Value from a hidden state rep & raw action."""

        # Process the action & broadcast to state shape
        out = self.action_net(action)
        out = out.unsqueeze(2).unsqueeze(3).expand_as(hidden)

        # hidden + action -> Q
        out = (hidden + out).view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def choose_action(self, image, cur_time, use_cem=True):
        """Used by the agent to select an action in an environment."""

        # Process the state representation using a CNN
        image_var = Variable(torch.FloatTensor(image) / 255., requires_grad=False)

        if isinstance(cur_time, float):
            time_var = torch.FloatTensor([cur_time])
        else:
            time_var = torch.from_numpy(cur_time)
        time_var = Variable(time_var, requires_grad=False)

        if self.use_cuda:
            image_var = image_var.cuda()
            time_var = time_var.cuda()

        hidden = self.state_net(image_var, time_var)

        # Find the optimal action to use through M iters of CEM
        if use_cem:
            return self._cross_entropy_method(hidden).cpu().numpy()
        return self._random_uniform(hidden).cpu().numpy()

    def forward(self, image, cur_time, action=None, requires_grad=True):
        """Calculates the Q-value for a given state and action.

        During training, the current policy will execute a recorded action and
        observe the output Q value, while target policy will perform a small
        optimization over random actions, and return the best one for each sample.
        """

        # Process the state representation using a CNN
        image_var = Variable(torch.from_numpy(image / 255.),
                             requires_grad=requires_grad)
        time_var = Variable(torch.from_numpy(cur_time),
                            requires_grad=requires_grad)

        if self.use_cuda:
            image_var = image_var.cuda()
            time_var = time_var.cuda()

        hidden = self.state_net(image_var, time_var)

        # Although it's a bit inefficient to compute the optimal action and then
        # calculate the Q-value again, it's not terribly expensive when the
        # layers are small-scale / fully-connected
        if action is None:
            action = Variable(self._random_uniform(hidden),
                              requires_grad=requires_grad)
        elif isinstance(action, np.ndarray):
            action = Variable(torch.from_numpy(action), 
                              requires_grad=requires_grad)
            if self.use_cuda:
                action = action.cuda()

        return self._forward_q(hidden, action)
