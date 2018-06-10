import os
import copy
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from agent import Agent
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

    model = Agent(in_channels, out_channels, conf.ACTION_SIZE,
                  num_uniform, num_cem, cem_iter, cem_elite).to(device)
    # model.load_state_dict(torch.load('checkpoints/2000_model.pt'))

    q_target = copy.deepcopy(model)

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
            pred = model(s0, t0, act).view(-1)  # saved action

            # We don't calculate a gradient for the target network; these
            # weights instead get updated by copying the prediction network
            # weights every few training iterations
            with torch.no_grad():
                target = r + (1. - term) * gamma * q_target(s1, t1).view(-1)
   
            loss = torch.mean((pred - target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Update the target network
            cur_iter += 1
            if cur_iter % q_update_iter == 0:
                q_target = copy.deepcopy(model)

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
        
