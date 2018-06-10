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
# TODO: Does the replay memory buffer get updated at all? If not, we're 
# constraining the network predict outcome with a fixed distribution equal 
# to the split of (positive, negative) samples in the training set
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    torch.manual_seed(1234)
    np.random.seed(1234)

    render_env = False
    remove_height_hack = True
    data_dir = 'data' if remove_height_hack else 'data_height_hack'
    action_size = 4 if remove_height_hack else 3

    max_num_episodes = 50000
    max_num_steps = 15
    max_buffer_size = 100000
    q_update_iter = 50

    lrate = 1e-4
    decay = 0.
    batch_size = 128
    num_rows = 64
    num_cols = 64
    in_channels = 3
    out_channels = 32
    num_random = 64
    num_cem = 64
    cem_iter = 3
    cem_elite = 6
    max_grad_norm = 100
    gamma = 0.96
    state_space = (3, num_rows, num_cols)
    action_space = (action_size,)
    reward_queue = deque(maxlen=100)
    loss_queue = deque(maxlen=100)
    accuracies = deque(maxlen=100)

    env = e.KukaDiverseObjectEnv(height=num_rows,
                                 width=num_cols,
                                 removeHeightHack=remove_height_hack,
                                 maxSteps=max_num_steps,
                                 renders=render_env,
                                 isDiscrete=False)
    # env.render(mode='human')

    model = Agent(in_channels, out_channels, action_size,
                  num_random, num_cem, cem_iter, cem_elite).to(device)

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

        start = time.time()

        state = env.reset()
        state = state.transpose(2, 0, 1)[np.newaxis]

        # Apply the linearly spaced action vector in the simulator
        sequence = []
        for step in range(max_num_steps + 1):

            # When we select an action to use in the simulator - use CEM
            # This action represents a vector from the current state to the end
            # state - divide the action into max_num_steps segments
            state_ = state.astype(np.float32) / 255.
            cur_step = float(step) / float(max_num_steps)

            # NOTE: Action is currently limited to [-1, 1] in agent.py
            action = model.choose_action(state_, cur_step)

            action_step = action.cpu().numpy().flatten()

            next_state, reward, terminal, _ = env.step(action_step)
            next_state = next_state.transpose(2, 0, 1)[np.newaxis]

            sequence.append([state, action, reward, next_state, terminal, step])

            # Train the networks
            s0, act, r, _, _, timestep = next(iter(train_loader))

            s0 = s0.to(device).requires_grad_(True)
            act = act.to(device).requires_grad_(True)
            r = r.to(device).requires_grad_(True)

            t0 = timestep / float(max_num_steps)
            t0 = t0.to(device).requires_grad_(True)

            # Predict a binary outcome
            q_pred = model(s0, t0, act).view(-1)

            
            #if step == 0:
            #    np.set_printoptions(2)
            #    print('q_pred: \n', q_pred.cpu().data.numpy()[:10])
            loss = torch.nn.BCEWithLogitsLoss()(q_pred, r.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


            q_pred = q_pred.detach()
            q_pred[q_pred >= 0.5] = 1.
            q_pred[q_pred < 0.5] = 0.
            acc = (q_pred.cpu().data.numpy() == r.cpu().data.numpy()).mean()
            accuracies.append(acc)
            loss_queue.append(loss.detach().cpu().numpy())

            if terminal:
                break
            state = next_state

        # Do we update the memory buffer or not?
        '''
        action_sum = 0.
        for seq in sequence:

            state, action, _, next_state, terminal, step = seq

            action_sum = action_sum + action

            # Note that reward is the final reward of the episode
            memory.add(state, action_sum, reward, next_state, terminal, step)
        '''

        reward_queue.append(reward)

        print('Episode: %d, Step: %2d, Reward: %2.2f, Acc: %2.2f, Loss: %2.4f, Took: %2.4f' %
            (episode, step, np.mean(reward_queue), np.mean(accuracies), 
             np.mean(loss_queue), time.time() - start))

        if episode % 100 == 0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoints/%d_model.pt' % episode)

        
