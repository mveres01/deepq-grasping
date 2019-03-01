import torch
import numpy as np


def _preprocess_inputs(image, timestep, device):
    """Ensures inputs are formatted as torch tensors."""

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).to(device)
    if isinstance(timestep, (int, float)):
        timestep = torch.tensor([timestep], dtype=torch.float32, device=device)
    return image, timestep


class CEMOptimizer:
    """Implements the cross entropy method.

    The main input when calling is the network. We assume that the network
    has the following components:

    * state_net: Computes a hidden representation from an image
    * action_net: Computes a hidden representation from an action
    * qnet: Computes an output q value from hidden states & actions

    As this is a sampling based method, it is much more efficient to only
    compute the hidden representations for states _once_, and re-use them
    on subsequent iterations.

    Notes
    -----
    Note that the CEM method is (generally) only run in evaluation mode,
    where a single sample is optimized at a time. However, this function
    supports batch sizes > 0

    See: https://en.wikipedia.org/wiki/Cross-entropy_method
    """

    def __init__(self, num_cem, cem_elite, cem_iter, action_size, bounds,
                 device, **kwargs):

        self.pop_size = num_cem
        self.elite = cem_elite
        self.iters = cem_iter
        self.action_size = action_size
        self.bounds = bounds
        self.device = device

    @torch.no_grad()
    def __call__(self, network, image, timestep):

        if not hasattr(network, 'state_net'):
            raise AttributeError('Network does not have \"state_net\" parameter.')
        elif not hasattr(network, 'action_net'):
            raise AttributeError('Network does not have \"action_net\" parameter.')
        elif not hasattr(network, 'qnet'):
            raise AttributeError('Network does not have \"qnet\" parameter.')

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


class SupervisedCEMOptimizer:
    """Implements the cross entropy method.

    The main input when calling is the network. We assume that the network
    has the following components:

    * state_net: Computes a hidden representation from an image
    * action_net: Computes a hidden representation from an action
    * qnet: Computes an output q value from hidden states & actions

    As this is a sampling based method, it is much more efficient to only
    compute the hidden representations for states _once_, and re-use them
    on subsequent iterations.

    Notes
    -----
    This optimizer takes into account that the initial policy is biased 
    to move downwards, and limits the search space along the z-dimension. 

    See: https://en.wikipedia.org/wiki/Cross-entropy_method
    """

    def __init__(self, num_cem, cem_elite, cem_iter, action_size, bounds,
                 device, **kwargs):

        self.pop_size = num_cem
        self.elite = cem_elite
        self.iters = cem_iter
        self.action_size = action_size
        self.bounds = bounds
        self.device = device

    @torch.no_grad()
    def __call__(self, network, image, timestep):

        if not hasattr(network, 'state_net'):
            raise AttributeError('Network does not have \"state_net\" parameter.')
        elif not hasattr(network, 'action_net'):
            raise AttributeError('Network does not have \"action_net\" parameter.')
        elif not hasattr(network, 'qnet'):
            raise AttributeError('Network does not have \"qnet\" parameter.')

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

            # Restrict search space
            action[:, 2] = action[:, 2].clamp(-1, -0.5)
            

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



class UniformOptimizer:
    """Used during training to find the most likely actions.

    The main input when calling is the network. We assume that the network
    has the following components:

    * state_net: Computes a hidden representation from an image
    * action_net: Computes a hidden representation from an action
    * qnet: Computes an output q value from hidden states & actions

    As this is a sampling based method, it is much more efficient to only
    compute the hidden representations for states _once_, and re-use them
    on subsequent iterations.

    Notes
    -----
    This function samples a batch of vectors from [-1, 1], and computes
    the Q value using the corresponding state. The action with the
    highest Q value is returned as the optimal action.
    """

    def __init__(self, num_uniform, action_size, bounds, device, **kwargs):

        self.pop_size = num_uniform
        self.action_size = action_size
        self.bounds = bounds
        self.device = device

    @torch.no_grad()
    def __call__(self, network, image, timestep):

        if not hasattr(network, 'state_net'):
            raise AttributeError('Network does not have \"state_net\" parameter.')
        elif not hasattr(network, 'action_net'):
            raise AttributeError('Network does not have \"action_net\" parameter.')
        elif not hasattr(network, 'qnet'):
            raise AttributeError('Network does not have \"qnet\" parameter.')

        network.eval()

        image, timestep = _preprocess_inputs(image, timestep, self.device)

        # We repeat the hidden state representation of the input (rather
        # then the raw input itself) to save some memory
        hstate = network.state_net(image, timestep)

        # (B, N, R, C) -> repeat (B, Unif, N, R, C) -> (B * Unif, N, R, C)
        hstate = hstate.unsqueeze(1)\
                       .repeat(1, self.pop_size, 1, 1, 1)\
                       .view(-1, *hstate.size()[1:])

        # Sample actions uniformly
        actions = torch.zeros((hstate.size(0), self.action_size),
                              device=self.device).uniform_(*self.bounds)

        q = network.qnet(hstate, network.action_net(actions))
        q = q.view(-1, self.pop_size)

        # Reshape to (Batch, Uniform) to find max action along dim=1
        topq, top1 = q.max(1)

        # Need to reshape the vectors to index the proper actions
        top1 = top1.view(-1, 1, 1).expand(-1, 1, self.action_size)
        actions = actions.view(-1, self.pop_size, self.action_size)
        actions = torch.gather(actions, 1, top1)

        network.train()
        return actions.squeeze(1), topq
