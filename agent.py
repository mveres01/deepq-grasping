import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StateNetwork(nn.Module):
    """Used to compute a nonlinear representation for the state."""

    def __init__(self, out_channels, kernel=5):
        super(StateNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU())

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
        self.fc2 = nn.Linear(num_outputs, num_outputs)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))

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

        # (s, a) -> q
        out = (hidden_state + out).view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


class BaseNetwork(nn.Module):

    def __init__(self, out_channels, action_size, num_uniform, 
                 num_cem, cem_iter, cem_elite, bounds=(-1, 1)):
        super(BaseNetwork, self).__init__()

        self.action_size = action_size
        self.num_uniform = num_uniform
        self.num_cem = num_cem
        self.cem_iter = cem_iter
        self.cem_elite = cem_elite
        self.bounds = bounds # action bounds

        self.state_net = StateNetwork(out_channels)
        self.action_net = ActionNetwork(action_size, out_channels + 1)
        self.qnet = StateActionNetwork(out_channels)

        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)

    def _cem_optimizer(self, hidden_state):
        """Implements the cross entropy method.

        This function is only implemented for a running with a single sample
        at a time. Extending to multiple samples is straightforward, but not
        needed in this case as CEM method is only run in evaluation mode.

        See: https://en.wikipedia.org/wiki/Cross-entropy_method
        """

        hidden_state = hidden_state.expand(self.num_cem, -1, -1, -1)

        mu = torch.zeros(1, self.action_size, device=DEVICE)
        std = torch.ones_like(mu)

        size = (self.num_cem, self.action_size)
        for _ in range(self.cem_iter):

            # Sample actions from the Gaussian parameterized by (mu, std)
            action = torch.normal(mu.expand(*size),
                                  std.expand(*size))\
                                 .clamp(*self.bounds).to(DEVICE)
            hidden_action = self.action_net(action)

            q = self.qnet(hidden_state, hidden_action).view(-1)

            # Find the top actions and use them to update the sampling dist
            _, topk = torch.topk(q, self.cem_elite)

            mu = action[topk].mean(dim=0, keepdim=True).detach()
            std = action[topk].std(dim=0, keepdim=True).detach()
            
        return mu

    def _uniform_optimizer(self, hidden_state):
        """Used during training to find the most likely actions.

        This function samples a batch of vectors from [-1, 1], and computes
        the Q value using the corresponding state. The action with the
        highest Q value is returned as the optimal action.

        Note that this function uses the :hidden: state representation. This
        lets us process a batch of state samples on the GPU together, then
        individually compute the optimal action for each.

        This version is a bit more memory intensive then the other one,
        but much more efficient due to batch processing.
        """

        # Repeat the samples along dim 1 so extracting action later is easy
        # (B, N, R, C) -> repeat (B, Unif, N, R, C) -> (B * Unif, N, R, C)
        hidden = hidden_state.unsqueeze(1)\
                             .repeat(1, self.num_uniform, 1, 1, 1)\
                             .view(-1, *hidden_state.size()[1:])

        # Sample uniform actions actions between environment bounds
        actions = torch.zeros((hidden.size(0), self.action_size),
                              device=DEVICE)\
                             .uniform_(*self.bounds)
        hidden_action = self.action_net(actions)

        q = self.qnet(hidden, hidden_action)

        # Reshape to (Batch, Uniform) to find max action per sample
        top1 = q.view(-1, self.num_uniform).argmax(1)

        # The max indices are along an independent dimension, so we need 
        # to do some book-keeping with the action vector
        top1 = top1.view(-1, 1, 1).expand(-1, 1, self.action_size)
        actions = actions.view(-1, self.num_uniform, self.action_size)

        actions = torch.gather(actions, 1, top1).squeeze(1)

        return actions

    def forward(self, image, time, action=None):
        """Calculates the Q-value for a given state and action.

        During training, the current policy will execute a recorded action and
        observe the output Q value, while target policy will perform a small
        optimization over random actions, and return the best for each sample.
        """

        hidden_state = self.state_net(image, time)

        # If no action is given, we need to first calculate an optimal action,
        # then return the Q-value associated with it. Having to re-compute
        # the Q-value is a tad inefficient, but for small networks is OK
        if action is None:
            with torch.no_grad():
                action = self._uniform_optimizer(hidden_state)

        hidden_action = self.action_net(action)
        return self.qnet(hidden_state, hidden_action)

    def optimal_action(self, image, time):
        """Uses the uniform optimizer to return an optimal action

        This is used in double Q-learning to select an action from the
        current policy, before being passed to the target policy
        """

        with torch.no_grad():

            hidden = self.state_net(image, time)
            return self._uniform_optimizer(hidden)

    def sample_action(self, image, time):
        """Uses the CEM optimizer to sample an action in the environment."""

        with torch.no_grad():

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).to(DEVICE)
            if isinstance(time, float):
                time = torch.tensor([time], device=DEVICE)

            hidden = self.state_net(image, time)
            return self._cem_optimizer(hidden).cpu().numpy().flatten()

