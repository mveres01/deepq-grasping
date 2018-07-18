import os
import copy
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from agent import BaseNetwork
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

if __name__ == '__main__':

    in_channels = 3
    state_space = (3, conf.NUM_ROWS, conf.NUM_COLS)
    action_space = (conf.ACTION_SIZE,)
    reward_queue = deque(maxlen=100)

    lrate = 1e-4
    decay = 0.
    batch_size = 128
    max_grad_norm = 100
    gamma = 0.96
    q_update_iter = 50 
    
    out_channels = 32
    num_uniform = 64
    num_cem = 64
    cem_iter = 3
    cem_elite = 6

    model = BaseNetwork(in_channels, out_channels, conf.ACTION_SIZE,
                  num_uniform, num_cem, cem_iter, cem_elite).to(device)
    # model.load_state_dict(torch.load('checkpoints/2000_model.pt'))

    memory = ReplayMemoryBuffer(conf.MAX_BUFFER_SIZE, state_space, action_space)

    # Initialize memory with experience from disk, or collect new
    if os.path.exists(conf.DATA_DIR):
        memory.load(conf.DATA_DIR, conf.MAX_BUFFER_SIZE)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(conf.DATA_DIR)

    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)

    cur_iter = 0
    for episode in range(conf.MAX_NUM_EPISODES):

        start = time.time()

        state = env.reset()
        for step in range(conf.MAX_NUM_STEPS + 1):

            # When we select an action to use in the simulaltor - use CEM
            state = state.transpose(2, 0, 1)[np.newaxis]
            state = state.astype(np.float32) / 255. 

            cur_step = float(step) / float(conf.MAX_NUM_STEPS)

            action = model.choose_action(state, cur_step)
            action = action.cpu().numpy().flatten()

            state, reward, terminal, _ = env.step(action)

            # Sample data from the memory buffer & put on GPU
            loss = 0
            for batch in memory.sample_episode(batch_size):

                s0, act, r, _, _, timestep = batch

                s0 = torch.from_numpy(s0).to(device).requires_grad_(True)
                act = torch.from_numpy(act).to(device).requires_grad_(True)
                r = torch.from_numpy(r).to(device).requires_grad_(False)

                t0 = timestep / float(conf.MAX_NUM_STEPS)
                t0 = torch.from_numpy(t0).to(device).requires_grad_(True)

                # Train the models
                pred = model(s0, t0, act).view(-1)  # saved action
   
                # Since the rewards in this environment are sparse (and only 
                # occur at the final timestep of the episode), the loss
                # \sum_{t'=t}^T \gamma^{t' - t} r(s_t, a_t) will always just 
                # be the single-step reward
                loss = loss + torch.sum((pred - r) ** 2)

            loss = loss / batch_size

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if terminal:
                break

        reward_queue.append(reward)

        print('Episode: %d, Step: %2d, Reward: %1.2f, Took: %2.4f' %
              (episode, step, np.mean(reward_queue), time.time() - start))

        if episode % 100 == 0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoints/%d_model.pt' % episode)


        '''
        if episode % 100 != 0:
            continue


        # Check the learned policy
        start = time.time()
        test_rewards = deque(maxlen=100)
        for test_episode in range(10):

            state = test_env.reset()
            state = state.transpose(2, 0, 1)[np.newaxis]

            for step in range(conf.MAX_NUM_STEPS + 1):

                # When we select an action to use in the simulaltor - use CEM
                cur_step = float(step) / float(conf.MAX_NUM_STEPS)
                action = model.choose_action(state, cur_step).flatten()

                next_state, reward, terminal, _ = test_env.step(action)
                next_state = next_state.transpose(2, 0, 1)[np.newaxis]

                if terminal:
                    break
                state = next_state
            test_rewards.append(reward)

        print('\nTEST REWARD:  %1.2f, Took: %2.4f\n' %
              (np.mean(test_rewards), time.time() - start))
        '''
        
