import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class StateNetwork(nn.Module):
    """Used to compute a nonlinear representation for the state."""

    def __init__(self, out_channels, kernel=3):
        super(StateNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel, padding=0),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel, padding=1),
            nn.MaxPool2d(2))

    def forward(self, image, time):
        """Computes a hidden rep for the image & concatenates time."""

        out = self.net(image)
        time = time.view(-1, 1, 1, 1).expand(-1, 1, out.size(2), out.size(3))
        out = torch.cat((out, time), dim=1)
        return out


class ActionNetwork(nn.Module):
    """Used to compute a nonlinear representation for the action."""

    def __init__(self, action_size, num_outputs):
        super(ActionNetwork, self).__init__()
        self.fc1 = nn.Linear(action_size, num_outputs)

    def forward(self, input):
        out = self.fc1(input)
        return out


class StateActionNetwork(nn.Module):
    """Used to compute the final path from hidden [state, action] -> Q.

    Seperating the computation this way allows us to efficiently compute
    Q values by calculating the hidden state representation through a
    minibatch, then performing a full pass for each sample.
    """

    def __init__(self, out_channels, out_size=1):
        super(StateActionNetwork, self).__init__()

        self.fc1 = nn.Linear(7 * 7 * (out_channels + 1), out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.fc3 = nn.Linear(out_channels, out_size)

    def forward(self, hidden_state, hidden_action):
        """Computes the Q-Value from a hidden state rep & raw action."""

        # Process the action & broadcast to state shape
        out = hidden_action.unsqueeze(2).unsqueeze(3).expand_as(hidden_state)

        # (h_s, h_a) -> q
        out = F.relu(hidden_state + out).view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


class BaseNetwork(nn.Module):
    """Implements the main (state, action) -> (outcome) network.

    To make optimization more efficient (with CEM or Uniform action optimizers),
    the network is split into different processing blocks. The expensive state 
    computation is performed only once, while the hidden action representations 
    may be called repeatedly during the optimization process.
    """

    def __init__(self, out_channels, action_size, **kwargs):
        super(BaseNetwork, self).__init__()

        self.state_net = StateNetwork(out_channels)
        self.action_net = ActionNetwork(action_size, out_channels + 1)
        self.qnet = StateActionNetwork(out_channels)

        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)

    def forward(self, image, time, action):
        """Calculates the Q-value for a given state and action."""

        hidden_state = self.state_net(image, time)
        hidden_action = self.action_net(action)

        return self.qnet(hidden_state, hidden_action)

    @torch.no_grad()
    def optim_forward(self, hidden_state, action):
        """Passed to an optimizer to calculate optimal action"""
        return self.qnet(hidden_state, self.action_net(action))
