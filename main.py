import os
import time
import numpy as np
import torch
import argparse
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from collections import deque
from utils import ReplayBuffer, collect_experience


def test(env, model, memory, explore_prob=0., on_policy=False, **kwargs):
    """Performs a full grasping episode in the environment."""

    step = 0.
    done = False
    state = env.reset()

    state = state.transpose(2, 0, 1)[np.newaxis]

    for step in range(kwargs['max_steps']):

        state_ = state.astype(np.float32) / 255.

        #explore_prob = max((1000 - cur_iter) / 10000., 0.01)
        action = model.sample_action(state_, float(step), explore_prob)

        next_state, reward, done, _ = env.step(action)

        next_state = next_state.transpose(2, 0, 1)[np.newaxis]

        if on_policy:
            memory.add(state, action, reward, next_state, done, step)

        if done:
            return step, reward

        state = next_state


def main(**kwargs):

    warm_start = 5 # number of epochs to pre-train networks for
    warm_start_iters = int(warm_start * kwargs['buffer_size'] // kwargs['batch_size'])
    total_iters = 0 

    checkpoint_dir = os.path.join('checkpoints', kwargs['method'])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if kwargs['method'] == 'dqn':
        from dqn import DQN as Model
    elif kwargs['method'] == 'ddqn':
        from ddqn import DDQN as Model
    elif kwargs['method'] == 'ddpg':
        from ddpg import DDPG as Model
    else:
        raise ValueError('Model <%s> not understood'%kwargs['method'])

    model = Model(**kwargs)

    env = e.KukaDiverseObjectEnv(height=64,
                                 width=64,
                                 removeHeightHack=True,
                                 maxSteps=kwargs['max_steps'],
                                 renders=kwargs['render'],
                                 isDiscrete=False,
                                 isTest=not kwargs['is_train'])

    memory = ReplayBuffer(kwargs['buffer_size'], 
                          state_size=(3, 64, 64),
                          action_size=(kwargs['action_size'],))

    if os.path.exists(kwargs['data_dir']):
        memory.load(kwargs['data_dir'], kwargs['buffer_size'])
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(kwargs['data_dir'])


    # Do a warm start so we don't spend time acting in the environment early on
    for it in range(warm_start_iters): 
        model.train(memory, **kwargs)

        if it % kwargs['update_iter'] == 0:
            model.update()
            print('Warm start iter %d / %d'%(it, warm_start_iters))

    # Train the model & simultaneously test in the environment
    reward_queue = deque(maxlen=200)
    for episode in range(kwargs['max_episodes']):
       
        start = time.time()

        step, reward = test(env, model, memory, **kwargs)
        reward_queue.append(reward)

        if episode % 50 == 0:
            checkpoint = os.path.join(checkpoint_dir, '%d' % episode)
            model.save_checkpoint(checkpoint)

        for _ in range(int(step)): 
            model.train(memory, **kwargs)

            total_iters += 1
            if total_iters % kwargs['update_iter'] == 0:
                model.update()

        print('Episode: %d, Step: %2d, Reward: %1.2f, Took: %2.4f' %
              (episode, step, np.mean(reward_queue), time.time() - start))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Off Policy Deep Q-Learning')
  
    # Model parameters
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--model', dest='method', default='dqn')
    parser.add_argument('--on-policy', action='store_true', default=False)
    parser.add_argument('--epochs', dest='max_episodes', default=1500, type=int)
    parser.add_argument('--data-dir', default='data1M')
    parser.add_argument('--buffer-size', default=100000, type=int)
    parser.add_argument('--checkpoint', default=None)
    
    # Hyperparameters
    parser.add_argument('--channels', dest='num_features', default=32, type=int)
    parser.add_argument('--gamma', default=0.90, type=float)
    parser.add_argument('--decay', default=1e-3, type=float)
    parser.add_argument('--lr', dest='lrate', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=200, type=float)
    parser.add_argument('--update', dest='update_iter', default=50, type=int)
    parser.add_argument('--explore-prob', default=0., type=float)
   
    # Optimizer Parameters
    parser.add_argument('--uniform', dest='num_uniform', default=32, type=int)
    parser.add_argument('--cem', dest='num_cem', default=64, type=int)
    parser.add_argument('--cem-iter', default=3, type=int)
    parser.add_argument('--cem-elite', default=8, type=int)
   
    # Environment Parameters
    parser.add_argument('--max-steps', default=15, type=int)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--is-train', action='store_true', default=False)

    kwargs = vars(parser.parse_args())
    kwargs['action_size'] = 4
    kwargs['bounds'] = (-1, 1)
    kwargs['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
 
    main(**kwargs)
