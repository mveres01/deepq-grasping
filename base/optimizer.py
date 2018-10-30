import torch
import numpy as np


class CEMOptimizer:
    """Implements the cross entropy method.

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

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], device=self.device)

        network.eval()
        state = network.state_net(image, timestep)

        mu = torch.zeros(state.size(0), 1, self.action_size, device=self.device)
        std = torch.ones_like(mu) * 0.5

        # Repeat the samples along dim 1 so extracting action later is easy
        # (B, N, R, C) -> repeat (B, Cem, N, R, C) -> (B * Cem, N, R, C)
        state = state.unsqueeze(1) \
                     .repeat(1, self.pop_size, 1, 1, 1) \
                     .view(-1, *state.size()[1:])

        for i in range(self.iters):

            # Sample actions from the Gaussian parameterized by (mu, std)
            action = torch.normal(mu.expand(-1, self.pop_size, -1),
                                  std.expand(-1, self.pop_size, -1))\
                                 .clamp(*self.bounds).to(self.device)
            action = action.view(-1, self.action_size)

            # Evaluate the actions using a forward pass through the network
            q = network.optim_forward(state, action).view(-1, self.pop_size)

            # Find the top actions and use them to update the sampling dist
            topq, topk = torch.topk(q, self.elite, dim=1)

            # Book-keeping to extract topk actions for each batch sample
            topk = topk.unsqueeze(2).expand(-1, -1, self.action_size)
            action = action.view(-1, self.pop_size, self.action_size)
            action = torch.gather(action, 1, topk)

            s = np.sqrt(max(2. - i / self.iters, 0))
            #s = np.sqrt(max(5. - i / 10., 0))

            mu = action.mean(dim=1, keepdim=True).detach()
            std = action.std(dim=1, keepdim=True).detach() + s

        network.train()
        return mu.squeeze(1), topq


class UniformOptimizer:
    """Used during training to find the most likely actions.

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

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], device=self.device)

        network.eval()
        state = network.state_net(image, timestep)

        # Repeat the samples along dim 1 so extracting action later is easy
        # (B, N, R, C) -> repeat (B, Unif, N, R, C) -> (B * Unif, N, R, C)
        state = state.unsqueeze(1)\
                     .repeat(1, self.pop_size, 1, 1, 1)\
                     .view(-1, *state.size()[1:])

        # Sample uniform actions actions between environment bounds
        actions = torch.zeros((state.size(0), self.action_size),
                              device=self.device).uniform_(*self.bounds)

        q = network.optim_forward(state, actions)

        # Reshape to (Batch, Uniform) to find max action along dim=1
        topq, top1 = q.view(-1, self.pop_size).max(1)

        # Need to reshape the vectors to index the proper actions
        top1 = top1.view(-1, 1, 1).expand(-1, 1, self.action_size)
        actions = actions.view(-1, self.pop_size, self.action_size)
        actions = torch.gather(actions, 1, top1)

        network.train()
        return actions.squeeze(1), topq
