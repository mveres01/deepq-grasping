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
            nn.MaxPool2d(2),
            )

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

    This module is shared across several different learning algorithms. When
    an action is passed as input, it will use that in calculating an output
    grasp probability. When an action is missing, this module will calculate
    an action by performing an optimization using either the cross entropy
    method, or a uniform optimizer.

    To make optimization more efficient, the network is split into different
    processing blocks. The expensive state computation is performed only once,
    and the hidden representation is replicated to match the number of action
    samples we wish to evaluate.
    """

    def __init__(self, out_channels, action_size, num_uniform, num_cem,
                 cem_iter, cem_elite, device, bounds=(-1, 1), **kwargs):
        super(BaseNetwork, self).__init__()

        self.action_size = action_size
        self.num_uniform = num_uniform
        self.num_cem = num_cem
        self.cem_iter = cem_iter
        self.cem_elite = cem_elite
        self.device = device
        self.bounds = bounds  # action bounds for sampling / optimization

        self.state_net = StateNetwork(out_channels)
        self.action_net = ActionNetwork(action_size, out_channels + 1)
        self.qnet = StateActionNetwork(out_channels)

        for param in self.parameters():
            if len(param.shape) > 1:
                #nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)
                #nn.init.kaiming_uniform_(param)

    @torch.no_grad()
    def _cem_optimizer(self, hidden_state):
        """Implements the cross entropy method.

        This function is only implemented for a running with a single sample
        at a time. Extending to multiple samples is straightforward, but not
        needed in this case as CEM method is only run in evaluation mode.

        See: https://en.wikipedia.org/wiki/Cross-entropy_method
        """

        self.eval()
        
        hidden_state = hidden_state.expand(self.num_cem, -1, -1, -1)
    
        mu = torch.zeros(1, self.action_size, device=self.device)
        std = torch.ones_like(mu) * 0.5

        for i in range(self.cem_iter):

            # Sample actions from the Gaussian parameterized by (mu, std)
            action = torch.normal(mu.expand(self.num_cem, -1),
                                  std.expand(self.num_cem, -1))\
                                 .clamp(*self.bounds).to(self.device)
            hidden_action = self.action_net(action)

            q = self.qnet(hidden_state, hidden_action).view(-1)

            # Find the top actions and use them to update the sampling dist
            _, topk = torch.topk(q, self.cem_elite)

            s = np.sqrt(max(5 - i / 10., 0))

            mu = action[topk].mean(dim=0, keepdim=True).detach()
            std = action[topk].std(dim=0, keepdim=True).detach() + s

        self.train()
        return mu

    @torch.no_grad()
    def _uniform_optimizer(self, hidden_state):
        """Used during training to find the most likely actions.

        This function samples a batch of vectors from [-1, 1], and computes
        the Q value using the corresponding state. The action with the
        highest Q value is returned as the optimal action.
        """

        self.eval()

        # Repeat the samples along dim 1 so extracting action later is easy
        # (B, N, R, C) -> repeat (B, Unif, N, R, C) -> (B * Unif, N, R, C)
        hidden = hidden_state.unsqueeze(1)\
                             .repeat(1, self.num_uniform, 1, 1, 1)\
                             .view(-1, *hidden_state.size()[1:])

        # Sample uniform actions actions between environment bounds
        actions = torch.zeros((hidden.size(0), self.action_size),
                              device=self.device).uniform_(*self.bounds)

        q = self.qnet(hidden, self.action_net(actions))

        # Reshape to (Batch, Uniform) to find max action along dim=1
        top1 = q.view(-1, self.num_uniform).argmax(1)

        # Need to reshape the vectors to index the proper actions
        top1 = top1.view(-1, 1, 1).expand(-1, 1, self.action_size)
        actions = actions.view(-1, self.num_uniform, self.action_size)
        actions = torch.gather(actions, 1, top1).squeeze(1)

        self.train()
        return actions

    def forward(self, image, time, action=None):
        """Calculates the Q-value for a given state and action.

        During training, the current policy will execute a recorded action and
        observe the output Q value. The target policy is required to find
        an optimal action, which we do using the uniform optimizer.
        """

        hidden_state = self.state_net(image, time)

        # If no action is given, we need to first calculate an optimal action,
        # then return the Q-value associated with it. Having to re-compute
        # the Q-value is a tad inefficient, but for small networks is OK
        if action is None:
            action = self._uniform_optimizer(hidden_state)

        hidden_action = self.action_net(action)
        return self.qnet(hidden_state, hidden_action)

    @torch.no_grad()
    def sample_action(self, image, time, mode='cem'):
        """Uses the CEM optimizer to sample an action in the environment."""

        with torch.no_grad():

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).to(self.device)
            if isinstance(time, float):
                time = torch.tensor([time], device=self.device)

            hidden = self.state_net(image, time)
            if mode == 'cem':
                return self._cem_optimizer(hidden).detach()
            return self._uniform_optimizer(hidden).detach()
