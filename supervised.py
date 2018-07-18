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

from config import Config as conf
from agent import BaseNetwork
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


if __name__ == '__main__':

    in_channels = 3
    out_channels = 32
    bounds = (-10, 10)
    reward_queue = deque(maxlen=100)

    conf.MAX_NUM_EPISODES = 50000
    conf.MAX_NUM_STEPS = 15
    conf.MAX_BUFFER_SIZE = 100000

    lrate = 1e-3
    decay = 1e-3
    batch_size = 128
    max_grad_norm = 100
    gamma = 0.96
    q_update_iter = 10


    num_random = 64
    num_cem = 64
    cem_iter = 3
    cem_elite = 6
    reward_queue = deque(maxlen=100)
    loss_queue = deque(maxlen=100)
    accuracies = deque(maxlen=100)

    # env.render(mode='human')
    model = BaseNetwork(in_channels, out_channels, conf.ACTION_SIZE,
                        num_random, num_cem, cem_iter, cem_elite, bounds).to(device)

    # model.load_state_dict(torch.load('checkpoints/2000_model.pt'))
    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)

    memory = ReplayMemoryBuffer(conf.MAX_BUFFER_SIZE, 
                                conf.STATE_SPACE, 
                                conf.ACTION_SPACE)

    if os.path.exists(conf.DATA_DIR):
        memory.load(conf.DATA_DIR, conf.MAX_BUFFER_SIZE)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(conf.DATA_DIR)
    memory.set_supervised()

    train_loader = DataLoader(memory, batch_size, True, num_workers=0)

    cur_iter = 0
    for episode in range(conf.MAX_NUM_EPISODES):

        start = time.time()

        # Apply the linearly spaced action vector in the simulator
        state = env.reset()
        sequence = []
        for step in range(conf.MAX_NUM_STEPS + 1):

            # When we select an action to use in the simulator - use CEM
            # This action represents a vector from the current state to the end
            # state - divide the action into conf.MAX_NUM_STEPS segments
            state_ = state.transpose(2, 0, 1)[np.newaxis]
            state_ = state_.astype(np.float32) / 255. 

            cur_step = float(step) / float(conf.MAX_NUM_STEPS)

            # NOTE: Action is currently limited to [-1, 1] in agent.py
            action = model.choose_action(state_, cur_step)

            action_step = action.cpu().numpy().flatten()

            print(action_step)

            action_step = action_step / (conf.MAX_NUM_STEPS - step)

            #print(action_step)

            next_state, reward, terminal, _ = env.step(action_step)

            sequence.append([state.transpose(2, 0, 1), 
                             action, reward, 
                             next_state.transpose(2, 0, 1), 
                             terminal, step])
            state = next_state

            # Train the networks
            s0, act, r, _, _, timestep = next(iter(train_loader))


            s0 = s0.to(device).requires_grad_(True)
            act = act.to(device).requires_grad_(True)
            r = r.to(device).requires_grad_(True)

            t0 = timestep / float(conf.MAX_NUM_STEPS)
            t0 = t0.to(device).requires_grad_(True)

            # Predict a binary outcome
            pred = model(s0, t0, act).view(-1)

            loss = torch.nn.BCEWithLogitsLoss()(pred, r.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            q_pred = pred.detach()
            q_pred[q_pred >= 0.5] = 1.
            q_pred[q_pred < 0.5] = 0.
            acc = (q_pred.cpu().data.numpy() == r.cpu().data.numpy()).mean()
            accuracies.append(acc)
            loss_queue.append(loss.detach().cpu().numpy())

            if terminal:
                break

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

        
