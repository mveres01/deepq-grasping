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
from config import Config as conf
from utils import ReplayMemoryBuffer, collect_experience

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(conf.SEED)
torch.manual_seed(conf.SEED)

env = e.KukaDiverseObjectEnv(height=conf.NUM_ROWS,
                             width=conf.NUM_COLS,
                             removeHeightHack=conf.REMOVE_HEIGHT_HACK,
                             maxSteps=conf.MAX_NUM_STEPS,
                             renders=conf.RENDER_ENV,
                             isDiscrete=conf.IS_DISCRETE)


class Actor(nn.Module):
    def __init__(self, in_channels, out_channels, action_size):
        super(Actor, self).__init__()

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
        out = F.tanh(self.fc3(out)) # Constrain actions to [-1, 1]
        return out


if __name__ == '__main__':

    in_channels = 3
    state_space = (3, conf.NUM_ROWS, conf.NUM_COLS)
    action_space = (conf.ACTION_SIZE,)
    reward_queue = deque(maxlen=100)

    lrate = 1e-3
    decay = 0.
    batch_size = 32
    max_grad_norm = 100
    gamma = 0.96
    q_update_iter = 50

    out_channels = 32
    num_uniform = 64
    num_cem = 64
    cem_iter = 3
    cem_elite = 6

    actor = Actor(in_channels, out_channels, conf.ACTION_SIZE).to(device)
    critic = Agent(in_channels, out_channels, conf.ACTION_SIZE, 0, 0, 0, 0).to(device)

    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)

    memory = ReplayMemoryBuffer(conf.MAX_BUFFER_SIZE, state_space, action_space)

    if os.path.exists(conf.DATA_DIR):
        memory.load(conf.DATA_DIR, conf.MAX_BUFFER_SIZE)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(conf.DATA_DIR)

    actor_optim = optim.Adam(actor.parameters(), lrate, weight_decay=decay)
    critic_optim = optim.Adam(critic.parameters(), lrate, weight_decay=decay)

    cur_iter = 0
    for episode in range(conf.MAX_NUM_EPISODES):

        start = time.time()

        state = env.reset()
        for step in range(conf.MAX_NUM_STEPS + 1):

            state = state.transpose(2, 0, 1)[np.newaxis]
            state = state.astype(np.float32) / 255.
            state = torch.from_numpy(state).to(device)

            cur_step = float(step) / float(conf.MAX_NUM_STEPS)
            cur_step = torch.tensor([cur_step], device=device)

            with torch.no_grad():
                action = actor(state, cur_step).cpu().data.numpy().flatten()

            # Add some exploration noise
            action = action + np.random.normal(0, 0.05, action.shape)
            action = np.clip(action, -1., 1.)

            state, reward, terminal, _ = env.step(action)

            # Train the networks;
            s0, act, r, s1, term, timestep = memory.sample(batch_size)

            s0 = torch.from_numpy(s0).to(device).requires_grad_(True)
            act = torch.from_numpy(act).to(device).requires_grad_(True)
            s1 = torch.from_numpy(s1).to(device).requires_grad_(False)
            r = torch.from_numpy(r).to(device).requires_grad_(False)
            term = torch.from_numpy(term).to(device).requires_grad_(False)

            t0 = timestep / float(conf.MAX_NUM_STEPS)
            t1 = (timestep + 1.) / float(conf.MAX_NUM_STEPS)

            t0 = torch.from_numpy(t0).to(device).requires_grad_(True)
            t1 = torch.from_numpy(t1).to(device).requires_grad_(False)

            # Train the models
            pred = critic(s0, t0, act).view(-1)

            # Training the critic is through Mean Squared Error
            with torch.no_grad():
                a_target = actor_target(s1, t1)
                q_target = critic_target(s1, t1, a_target).view(-1)
                target = r + (1. - term) * gamma * q_target

            loss = torch.mean((pred - target) ** 2)

            critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()
            critic_optim.zero_grad()

            # Update the actor network by following the policy gradient
            q_pred = -critic(s0, t0, actor(s0, t0)).mean()

            actor_optim.zero_grad()
            q_pred.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()
            actor_optim.zero_grad()

            # Update the target network
            cur_iter += 1
            if cur_iter % q_update_iter == 0:
                actor_target = copy.deepcopy(actor)
                critic_target = copy.deepcopy(critic)

            if terminal:
                break

        reward_queue.append(reward)

        print('Episode: %d, Step: %2d, Reward: %1.2f, Took: %2.4f' %
              (episode, step, np.mean(reward_queue), time.time() - start))
