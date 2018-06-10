import os
import copy
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from agent import Agent, StateNetwork
from utils import ReplayMemoryBuffer, collect_experience

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, in_channels, out_channels, action_size, use_cuda):
        super(Actor, self).__init__()

        self.use_cuda = use_cuda

        self.state_net = StateNetwork(in_channels, out_channels)
        self.fc1 = nn.Linear(7 * 7 * (out_channels + 1), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, cur_time):

        out = self.state_net(image, cur_time)
        out = out.view(image.shape[0], -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.tanh(self.fc3(out))
        return out


if __name__ == '__main__':

    torch.manual_see(1234)
    np.random.seed(1234)

    render_env = False
    remove_height_hack = True
    data_dir = 'data' if remove_height_hack else 'data_height_hack'

    max_num_episodes = 50000
    max_num_steps = 15
    max_buffer_size = 100000

    num_rows = 64
    num_cols = 64
    in_channels = 3
    out_channels = 32
    lrate = 1e-4
    batch_size = 64
    gamma = 0.96
    max_grad_norm = 100
    decay = 0
    q_update_iter = 50  # Every N iterations, make a copy of the current network

    action_size = 4 if remove_height_hack else 3
    state_space = (3, num_rows, num_cols)
    action_space = (action_size,)
    reward_queue = deque(maxlen=100)

    env = e.KukaDiverseObjectEnv(height=num_rows,
                                 width=num_cols,
                                 removeHeightHack=remove_height_hack,
                                 maxSteps=max_num_steps,
                                 renders=render_env,
                                 isDiscrete=False)

    actor = Actor(in_channels, out_channels, action_size).to(device)
    critic = Agent(in_channels, out_channels, 
                   action_size, 0, 0, 0, 0).to(device)

    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)

    actor_target.eval()
    critic_target.eval()

    actor_optim = optim.Adam(actor.parameters(), lrate, weight_decay=decay)
    critic_optim = optim.Adam(critic.parameters(), lrate, weight_decay=decay)

    memory = ReplayMemoryBuffer(max_buffer_size, state_space, action_space)

    if os.path.exists(data_dir):
        memory.load(data_dir, max_buffer_size)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(data_dir)

    print('Done initializing memory!')

    cur_iter = 0
    for episode in range(max_num_episodes):

        start = time.time()

        state = env.reset()
        state = state.transpose(2, 0, 1)[np.newaxis]

        for step in range(max_num_steps + 1):

            state_ = state.astype(np.float32) / 255.
            state_ = torch.from_numpy(state_).to(device)

            cur_step = float(step) / float(max_num_steps)
            cur_step = torch.tensor([cur_time, device=device])

            action = actor(state, cur_step).cpu().data.numpy().flatten()

            # Exploration
            action = action + np.random.normal(0, 0.05, action.shape)
            action = np.clip(action, -1., 1.)

            next_state, reward, terminal, _ = env.step(action)
            next_state = next_state.transpose(2, 0, 1)[np.newaxis]

            # Store the transition in the experience replay bank
            memory.add(state, action, reward, next_state, terminal, step)
            reward_queue.append(reward)

            # Train the networks;
            s0, act, r, s1, term, timestep = memory.sample(batch_size)

            s0 = torch.from_numpy(s0).to(device).requires_grad_(True)
            act = torch.from_numpy(act).to(device).requires_grad_(True)
            s1 = torch.from_numpy(s1).to(device).requires_grad_(False)
            r = torch.from_numpy(r).to(device).requires_grad_(False)
            term = torch.from_numpy(term).to(device).requires_grad_(False)

            t0 = timestep / float(max_num_steps)
            t0 = torch.from_numpy(t0).to(device).requires_grad_(True)

            t1 = (timestep + 1.) / float(max_num_steps)
            t1 = torch.from_numpy(t1).to(device).requires_grad_(False)

            # Training the critic is through Mean Squared Error
            with torch.no_grad():
                a_target = actor_target(s1, t1)
                q_target = critic_target(s1, t1, a_target).view(-1)
                y = r + (1. - term) * gamma * q_target

            q_pred = critic(s0, t0, act).view(-1)

            loss = torch.mean((y - q_pred) ** 2)

            critic_optim.zero_grad()
            loss.backward()
            critic_optim.step()
            critic_optim.zero_grad()

            # Update the actor network by following the policy gradient
            q_pred = -critic(s0, t0, actor(s0, t0)).mean()

            actor_optim.zero_grad()
            q_pred.backward()
            actor_optim.step()
            actor_optim.zero_grad()

            # Update the target network
            if cur_iter % q_update_iter == 0:
                actor_target = copy.deepcopy(actor)
                critic_target = copy.deepcopy(critic)

            cur_iter += 1
            if terminal:
                break
            state = next_state

        print('Episode: %d, Step: %d, Reward: %d, Took: %2.4f' %
              (episode, step, np.sum(reward_queue), time.time() - start))
