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

    lrate = 5e-4
    decay = 1e-3
    batch_size = 64
    max_grad_norm = 1000
    gamma = 0.9
    q_update_iter = 50

    out_channels = 32
    num_uniform = 16
    num_cem = 64
    cem_iter = 3
    cem_elite = 6

    model = Agent(in_channels, out_channels, conf.ACTION_SIZE,
                  num_uniform, num_cem, cem_iter, cem_elite).to(device)

    q_target = copy.deepcopy(model)

    memory = ReplayMemoryBuffer(conf.MAX_BUFFER_SIZE, 
                                conf.STATE_SPACE,
                                conf.ACTION_SPACE)

    if os.path.exists(conf.DATA_DIR):
        memory.load(conf.DATA_DIR, conf.MAX_BUFFER_SIZE)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(conf.DATA_DIR)

    #optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)
    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=decay)

    cur_iter = 0
    for episode in range(conf.MAX_NUM_EPISODES):

        start = time.time()

        state = env.reset()
        for step in range(conf.MAX_NUM_STEPS + 1):

            # When we select an action to use in the simulaltor - use CEM
            state = state.transpose(2, 0, 1)[np.newaxis]
            state = state.astype(np.float32) / 255. - 0.5
            
            cur_step = float(step) / float(conf.MAX_NUM_STEPS)
            
            action = model.choose_action(state, cur_step)
            action = action.cpu().numpy().flatten()

            state, reward, terminal, _ = env.step(action)

            # Train the networks;
            s0, act, r, s1, term, timestep = memory.sample(batch_size)

            s0 = torch.from_numpy(s0).to(device).requires_grad_(True) - 0.5
            act = torch.from_numpy(act).to(device).requires_grad_(True)
            s1 = torch.from_numpy(s1).to(device).requires_grad_(False) - 0.5
            r = torch.from_numpy(r).to(device).requires_grad_(False)
            term = torch.from_numpy(term).to(device).requires_grad_(False)

            t0 = timestep / float(conf.MAX_NUM_STEPS)
            t1 = (timestep + 1.) / float(conf.MAX_NUM_STEPS)
            
            t0 = torch.from_numpy(t0).to(device).requires_grad_(True)
            t1 = torch.from_numpy(t1).to(device).requires_grad_(False)

            # Predicts the Q value over the current state
            pred = model(s0, t0, act).view(-1)

            # For the target, we find the action that maximizes the Q value for
            # the current policy but use the Q value from the target policy
            with torch.no_grad():
                best_action = model.choose_action(s1, t1, use_cem=False)

                target = q_target(s1, t1, best_action).view(-1)
                target = r + (1. - term) * gamma * target

            loss = torch.sum((pred - target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
            checkpoint_dir = 'checkpoints/ddqn'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            save_dir = os.path.join(checkpoint_dir, '%d_model.py'%episode)
            torch.save(model.state_dict(), save_dir)
