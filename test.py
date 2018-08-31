import os
import time
import numpy as np
import torch
import argparse
from collections import deque
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from base import BaseNetwork
from utils import ReplayBuffer, collect_experience

import ray
ray.init()
time.sleep(2)


@ray.remote
class GymEnvironment(object):
    """Wrapper to run an environment on a remote CPU."""

    def __init__(self, model_creator, env_creator, seed=None):

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model = model_creator()
        self.env = env_creator()
        self.reset()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def rollout(self, weights, num_rollouts=5, explore_prob=0.):
        """Performs a full grasping episode in the environment."""

        self.model.set_weights(weights)

        steps, rewards = [], []
        for _ in range(num_rollouts):

            step, done = 0, False
            state = self.reset()

            while not done:

                state = state.transpose(2, 0, 1)[np.newaxis]
                state = state.astype(np.float32) / 255.

                # Sample and perform and action in the sim
                action = self.model.sample_action(state,
                                                  float(step),
                                                  explore_prob)
                state, reward, done, _ = self.env.step(action)

                step = step + 1

            steps.append(step - 1)
            rewards.append(reward)

        return (steps, rewards)


def make_env(env_config):
    """Makes a new environment given a config file."""

    def create():
        return e.KukaDiverseObjectEnv(**env_config)
    return create


def make_model(Model, network_config, **model_config):
    """Makes a new model given a config file."""

    network_creator = make_network(network_config)

    def create():
        return Model(network_creator, **model_config)
    return create


def make_network(network_config):
    """Creates a new network given a config file."""

    def create():
        return BaseNetwork(**network_config).to(network_config['device'])
    return create


def test(weights, envs, num_episodes=1, nr=5):

    # Train the model & simultaneously test in the environment
    steps, rewards = [], []

    for _ in range(num_episodes):

        start = time.time()

        results = [env.rollout.remote(weights, num_rollouts=nr) for env in envs]

        for item in ray.get(results):
            steps.extend(item[0])
            rewards.extend(item[1])
      
    return steps, rewards


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Off Policy Deep Q-Learning')

    # Model parameters
    parser.add_argument('--model', dest='method', default='dqn')
    parser.add_argument('--on-policy', action='store_true', default=False)
    parser.add_argument('--epochs', dest='max_episodes', default=10000, type=int)
    parser.add_argument('--data-dir', default='data100K2')
    parser.add_argument('--buffer-size', default=100000, type=int)
    parser.add_argument('--checkpoint', default=None)

    # Hyperparameters
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--channels', dest='num_features', default=32, type=int)
    parser.add_argument('--gamma', default=0.94, type=float)
    parser.add_argument('--decay', default=1e-5, type=float)
    parser.add_argument('--lr', dest='lrate', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--update', dest='update_iter', default=50, type=int)
    parser.add_argument('--explore-prob', default=0., type=float)
    parser.add_argument('--uniform', dest='num_uniform', default=16, type=int)

    # Optimization Parameters
    parser.add_argument('--cem', dest='num_cem', default=64, type=int)
    parser.add_argument('--cem-iter', default=3, type=int)
    parser.add_argument('--cem-elite', default=6, type=int)

    # Environment Parameters
    parser.add_argument('--max-steps', default=15, type=int)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--is-test', action='store_true', default=False)
    parser.add_argument('--block-random', default=0.3, type=float)

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.bounds = (-1, 1)
    args.action_size = 4
    args.device = torch.device('cpu')

    checkpoint_dir = os.path.join('checkpoints', args.method)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if args.method == 'dqn':
        from dqn_test import DQN as Model
    elif args.method == 'ddqn':
        from ddqn import DDQN as Model
    elif args.method == 'ddpg':
        from ddpg import DDPG as Model
    else:
        raise NotImplementedError('Model <%s> not implemented' % args.method)

    '''
    elif args.method == 'mcre':
        from mcre import MCRE as Model
        args.batch_size /= 8  # roughly average episode length
    elif args.method == 'supervised':
        from supervised import Supervised as Model
        memory.set_supervised()
        memory.action /= args.max_steps  # i.e. bound to [-1, 1]
        test = test_supervised
    '''

    # Defines parameters for network generator
    net_config = {'out_channels':args.num_features,
                  'action_size':args.action_size,
                  'num_uniform':args.num_uniform,
                  'num_cem':args.num_cem,
                  'cem_elite':args.cem_elite,
                  'cem_iter':args.cem_iter,
                  'device':args.device,
                  'bounds':args.bounds}

    model_creator = make_model(Model, net_config, **vars(args))

    # Defines parameters for distributed evaluation
    env_config = {'actionRepeat':80,
                  'isEnableSelfCollision':True,
                  'renders':args.render,
                  'isDiscrete':False,
                  'maxSteps':args.max_steps,
                  'dv':0.06,
                  'removeHeightHack':True,
                  'blockRandom':args.block_random,
                  'cameraRandom':0,
                  'width':64,
                  'height':64,
                  'numObjects':5,
                  'isTest':args.is_test}

    # Make the remote environments
    env_creator = make_env(env_config)
    envs = [GymEnvironment.remote(model_creator, env_creator) for _ in range(20)]

    # Create a trainable model
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_config['device'] = args.device
    model = make_model(Model, net_config, **vars(args))()

    if args.checkpoint is not None:
        model.load_checkpoint(args.checkpoint)

    if args.is_test:
        steps, rewards = test(model.get_weights(), envs, 5)

        print('Average across (%d) episodes: Step: %2.4f, Reward: %1.2f' %
              (args.max_episodes, np.mean(steps), np.mean(rewards)))
    else:

        # Initialize memory
        memory = ReplayBuffer(args.buffer_size,
                              state_size=(3, 64, 64),
                              action_size=(args.action_size,))

        if os.path.exists(args.data_dir):
            memory.load(args.data_dir, args.buffer_size)
        else:
            collect_experience(env_creator(), memory, print_status_every=100)
            memory.save(args.data_dir)

        # Train the model & simultaneously test in the environment
        step_queue = deque(maxlen=200)
        reward_queue = deque(maxlen=200)
        for episode in range(args.max_episodes):

            start = time.time()

            model.train(memory, **vars(args))

            if episode % args.update_iter == 0:
                model.update()

            if (episode > 0) and episode % (args.update_iter * 10) == 0:

                # Save checkpoint
                checkpoint = os.path.join(checkpoint_dir, '%d' % episode)
                model.save_checkpoint(checkpoint)

                # Test in sim
                steps, rewards = test(model.get_weights(), envs, 1, 5)

                for step, reward in zip(steps, rewards):
                    reward_queue.append(reward)
                    step_queue.append(step)

                print('Episode: %d, Step: %2.4f, Reward: %1.2f, Took: %2.4f' %
                      (episode, np.mean(step_queue), np.mean(reward_queue), 
                       time.time() - start))
