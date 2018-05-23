import os
import copy
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from agent import Agent, StateNetwork
from utils import ReplayMemoryBuffer, collect_experience


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
                nn.init.xavier_uniform(p)

    def forward(self, image, cur_time):

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

        out = self.state_net(image_var, time_var)
        out = out.view(image.shape[0], -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.tanh(self.fc3(out))
        return out


if __name__ == '__main__':

    render_env = False
    remove_height_hack = True
    data_dir = 'data' if remove_height_hack else 'data_height_hack'

    max_num_episodes = 50000
    max_num_steps = 15
    max_buffer_size = 100000

    use_cuda = torch.cuda.is_available()
    num_rows = 64
    num_cols = 64
    in_channels = 3
    out_channels = 32
    lrate = 1e-4
    batch_size = 64
    gamma = 0.96
    decay = 1e-2
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

    actor = Actor(in_channels, out_channels, action_size, use_cuda)
    critic = Agent(in_channels, out_channels, action_size, 0, 0, 0, 0, use_cuda)

    if use_cuda:
        actor.cuda()
        critic.cuda()

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

            cur_step = float(step) / float(max_num_steps)
            action = actor(state, cur_step).cpu().data.numpy().flatten()

            if episode % 5 == 0:
                np.set_printoptions(2)
                print(action)

            # Exploration
            action = action + np.random.normal(0, 0.05, action.shape)
            action = np.clip(action, -1., 1.)

            next_state, reward, terminal, _ = env.step(action)
            next_state = next_state.transpose(2, 0, 1)[np.newaxis]

            # Store the transition in the experience replay bank
            memory.add(state, action, reward, next_state, terminal, step)
            reward_queue.append(reward)

            # Update the current policy using experience from the memory buffer
            s0, act, r, s1, term, timestep = memory.sample(batch_size)

            t0 = timestep / max_num_steps
            t1 = (timestep + 1.) / max_num_steps

            # Training the critic is through Mean Squared Error
            a_target = actor_target(s1, t1)
            q_target = critic_target(s1, t1, a_target).view(-1)
            y = Variable(r + (1. - term) * gamma * q_target.data)

            if use_cuda:
                y = y.cuda()

            q_pred = critic(s0, t0, act).view(-1)

            critic_loss = torch.mean(torch.pow(y - q_pred, 2))

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            critic_optim.zero_grad()

            # Update the actor network
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
