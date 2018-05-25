import os
import copy
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from agent import Agent
from utils import ReplayMemoryBuffer, collect_experience

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

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
    num_uniform = 64
    num_cem = 64
    cem_iter = 3
    cem_elite = 6
    max_grad_norm = 100
    gamma = 0.96
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
                  num_uniform, num_cem, cem_iter, cem_elite).to(device)

    # model.load_state_dict(torch.load('checkpoints/2000_model.pt'))
    q_target = copy.deepcopy(model)

    memory = ReplayMemoryBuffer(max_buffer_size, state_space, action_space)

    if os.path.exists(data_dir):
        memory.load(data_dir, max_buffer_size)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(data_dir)

    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)

    cur_iter = 0
    for episode in range(max_num_episodes):

        start = time.time()

        state = env.reset()
        state = state.transpose(2, 0, 1)[np.newaxis]

        for step in range(max_num_steps + 1):

            # When we select an action to use in the simulaltor - use CEM
            state_ = state.astype(np.float32) / 255. 
            cur_step = float(step) / float(max_num_steps)
            
            action = model.choose_action(state_, cur_step).cpu().numpy()
            action = action.flatten()

            next_state, reward, terminal, _ = env.step(action)
            next_state = next_state.transpose(2, 0, 1)[np.newaxis]

            # Store the transition in the experience replay bank
            memory.add(state, action, reward, next_state, terminal, step)

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

            # Q_current predicts the Q value over the current state
            q_pred = model(s0, t0, act).view(-1)  # saved action

            # Q_targets predicts Q value over the next state
            with torch.no_grad():
                q_tgt = r + (1. - term) * gamma * q_target(s1, t1).view(-1)

            loss = torch.sum(torch.pow(q_pred - q_tgt, 2))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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


        '''
        if episode % 100 != 0:
            continue


        # Check the learned policy
        start = time.time()
        test_rewards = deque(maxlen=100)
        for test_episode in range(10):

            state = test_env.reset()
            state = state.transpose(2, 0, 1)[np.newaxis]

            for step in range(max_num_steps + 1):

                # When we select an action to use in the simulaltor - use CEM
                cur_step = float(step) / float(max_num_steps)
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
        
