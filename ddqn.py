import os
import copy
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from agent import Agent
from utils import ReplayMemoryBuffer, collect_experience


if __name__ == '__main__':

    render_env = False
    remove_height_hack = True
    use_precollected = True
    data_dir = 'data' if remove_height_hack else 'data_height_hack'
    action_size = 4 if remove_height_hack else 3

    max_num_episodes = 50000
    max_num_steps = 15
    max_buffer_size = 100000
    q_update_iter = 50  # Every N iterations, make a copy of the current network

    use_cuda = torch.cuda.is_available()

    lrate = 1e-3
    batch_size = 128
    num_rows = 64
    num_cols = 64
    in_channels = 3
    out_channels = 32
    num_random = 16
    num_cem = 64
    cem_iter = 3
    cem_elite = 3
    gamma = 0.96
    q_update_iter = 50  # Every N iterations, make a copy of the current network
    state_space = (3, num_rows, num_cols)
    action_space = (action_size,)
    reward_queue = deque(maxlen=100)

    env = e.KukaDiverseObjectEnv(height=num_rows,
                                 width=num_cols,
                                 removeHeightHack=remove_height_hack,
                                 maxSteps=max_num_steps,
                                 renders=render_env,
                                 isDiscrete=False)

    model = Agent(in_channels, out_channels, action_size,
                  num_random, num_cem, cem_iter, cem_elite, use_cuda)

    if use_cuda:
        model.cuda()

    q_target = copy.deepcopy(model)

    memory = ReplayMemoryBuffer(max_buffer_size, state_space, action_space)

    if os.path.exists(data_dir):
        memory.load(data_dir, max_buffer_size)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(data_dir)

    optimizer = optim.Adam(model.parameters(), lr=lrate)

    cur_iter = 0
    for episode in range(max_num_episodes):

        start = time.time()

        state = env.reset()
        state = state.transpose(2, 0, 1)[np.newaxis]

        for step in range(max_num_steps + 1):

            cur_step = float(step) / float(max_num_steps)

            # When we select an action to use in the simulaltor - use CEM
            action = model.choose_action(state, cur_step).flatten()

            next_state, reward, terminal, _ = env.step(action)
            next_state = next_state.transpose(2, 0, 1)[np.newaxis]

            # Store the transition in the experience replay bank
            memory.add(state, action, reward, next_state, terminal, step)


            # Update the current policy using experience from the memory buffer
            s0, act, r, s1, term, timestep = memory.sample(batch_size)
            timestep /= float(max_num_steps)

            # Q_current predicts the Q value over the current state
            q_pred = model(s0, timestep, act).view(-1)  # saved action

            # Q_targets predicts Q value over the next state, using the action that
            # maximizes the Q value of the current network
            timestep = timestep + 1. / float(max_num_steps)
            amax = model.choose_action(s1.copy(), timestep, use_cem=False)

            q_tgt = q_target(s1, timestep, amax).view(-1)  # random search

            y = Variable(r + (1. - term) * gamma * q_tgt.data)

            if use_cuda:
                y = y.cuda()

            loss = torch.sum(torch.pow(q_pred - y, 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target network
            if cur_iter % q_update_iter == 0:
                q_target = copy.deepcopy(model)

            cur_iter += 1

            if terminal:
                break
            state = next_state


        reward_queue.append(reward)

        print('Episode: %d, Step: %2d, Reward: %1.2f, Took: %2.4f' %
             (episode, step, np.mean(reward_queue), time.time() - start))

        if episode % 100 == 0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoints/%d_model.pt' % episode)
