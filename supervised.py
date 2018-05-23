"""
# TODO Check whether we:
# Compute an action at the start, and follow a linear path to goal OR
# Compute an action at each timestep
# How was this implemented in the paper? Should match memory buffer
# TODO: Check whether balancing the dataset will yield better performance
# TODO: Check previous Levine grasping paper for details on how grasps are
# re-chosen / chosen at every time step (e.g.~for visual servoing).
# Maybe make a second supervised file and check whether predicting at
# multiple steps is better then as a straight line?
"""

import os
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from agent import Agent
from utils import ReplayMemoryBuffer, collect_experience


if __name__ == '__main__':

    render_env = False
    remove_height_hack = True
    data_dir = 'data' if remove_height_hack else 'data_height_hack'
    action_size = 4 if remove_height_hack else 3

    max_num_episodes = 50000
    max_num_steps = 15
    max_buffer_size = 10000
    q_update_iter = 50  # Every N iterations, make a copy of the current network

    use_cuda = torch.cuda.is_available()

    lrate = 1e-4
    decay = 0.
    batch_size = 64
    num_rows = 64
    num_cols = 64
    in_channels = 3
    out_channels = 32
    num_random = 32
    num_cem = 64
    cem_iter = 3
    cem_elite = 6
    is_test = False
    gamma = 0.96
    state_space = (3, num_rows, num_cols)
    action_space = (action_size,)
    reward_queue = deque(maxlen=100)

    env = e.KukaDiverseObjectEnv(height=num_rows,
                                 width=num_cols,
                                 removeHeightHack=remove_height_hack,
                                 maxSteps=max_num_steps,
                                 renders=render_env,
                                 isDiscrete=False,
                                 isTest=is_test)
    # env.render(mode='human')

    model = Agent(in_channels, out_channels, action_size,
                  num_random, num_cem, cem_iter, cem_elite, use_cuda)

    if use_cuda:
        model.cuda()

    # model.load_state_dict(torch.load('checkpoints/2000_model.pt'))
    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)

    memory = ReplayMemoryBuffer(max_buffer_size, state_space, action_space)

    if os.path.exists(data_dir):
        memory.load(data_dir, max_buffer_size)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(data_dir)

    memory.set_supervised()
    train_loader = DataLoader(memory, batch_size, True, num_workers=0)

    cur_iter = 0
    for episode in range(max_num_episodes):

        for batch in train_loader:

            start = time.time()

            state = env.reset()
            state = state.transpose(2, 0, 1)[np.newaxis]

            # Apply the linearly spaced action vector in the simulator
            sequence = []
            for step in range(max_num_steps + 1):

                # When we select an action to use in the simulator - use CEM
                # This action represents a vector from the current state to the end
                # state - divide the action into max_num_steps segments
                cur_time = float(step) / float(max_num_steps)
                cur_action = model.choose_action(state, cur_time=cur_time).flatten()


                next_state, reward, terminal, _ = env.step(cur_action)
                next_state = next_state.transpose(2, 0, 1)[np.newaxis]

                sequence.append([state, cur_action, reward, next_state, terminal, step])


                # Train the networks
                s0, act, reward_label, _, _, timestep = batch

                timestep /= float(max_num_steps)
                reward_label = Variable(reward_label.cuda() if use_cuda else
                                        reward_label)

                # Q_current predicts the Q value over the current state
                q_pred = model(s0.numpy(), timestep.numpy(), act.numpy()).view(-1)

                loss = F.binary_cross_entropy_with_logits(q_pred, reward_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if terminal:
                    break
                state = next_state

            action_sum = 0.
            for seq in sequence:

                state, action, _, next_state, terminal, step = seq

                action_sum = action_sum + action

                # Note that reward is the final reward of the episode
                memory.add(state, action_sum, reward, next_state, terminal, step)

            reward_queue.append(reward)

            print('Episode: %d, Step: %2d, Reward: %2.4f, Took: %2.4f' %
                (episode, step, np.mean(reward_queue), time.time() - start))

        if episode % 100 == 0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoints/%d_model.pt' % episode)

        
