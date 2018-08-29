import os
import time
import numpy as np
import torch
import argparse
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from collections import deque

from base import BaseNetwork
from utils import ReplayBuffer, collect_experience


def test(env, model, memory, explore_prob=0., on_policy=False, **kwargs):
    """Performs a full grasping episode in the environment."""

    step = 0.
    done = False
    state = env.reset()

    state = state.transpose(2, 0, 1)[np.newaxis]

    for step in range(kwargs['max_steps']):

        state_ = state.astype(np.float32) / 255.
        action = model.sample_action(state_, float(step), explore_prob)

        next_state, reward, done, _ = env.step(action)

        next_state = next_state.transpose(2, 0, 1)[np.newaxis]

        if on_policy:
            memory.add(state, action, reward, next_state, done, step)

        if done:
            return step, reward

        state = next_state

    return step, reward


def test_supervised(env, model, memory, explore_prob=0., on_policy=False, **kwargs):
    """Performs a full grasping episode in the environment."""

    step = 0.
    done = False
    state = env.reset()

    state = state.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.

    action = model.sample_action(state, 0., explore_prob)

    # Repeat the predicted action until we reach terminal
    for step in range(kwargs['max_steps']):

        _, reward, done, _ = env.step(action)

        if done:
            break

    return step, reward


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = os.path.join('checkpoints', args.method)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    env = e.KukaDiverseObjectEnv(height=64,
                                 width=64,
                                 removeHeightHack=True,
                                 maxSteps=args.max_steps,
                                 renders=args.render,
                                 isDiscrete=False,
                                 blockRandom=args.block_random,
                                 isTest=not args.is_train)

    # Environment values
    args.bounds = (-1, 1)
    args.action_size = env.action_space.shape[0]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize memory
    memory = ReplayBuffer(args.buffer_size,
                          state_size=(3, 64, 64),
                          action_size=(args.action_size,))

    if os.path.exists(args.data_dir):
        memory.load(args.data_dir, args.buffer_size)
    else:
        collect_experience(env, memory, print_status_every=100)
        memory.save(args.data_dir)

    if args.method == 'dqn':
        from dqn import DQN as Model
    elif args.method == 'ddqn':
        from ddqn import DDQN as Model
    elif args.method == 'ddpg':
        from ddpg import DDPG as Model
    elif args.method == 'mcre':
        from mcre import MCRE as Model
        args.batch_size /= 8  # roughly average episode length
    elif args.method == 'supervised':
        from supervised import Supervised as Model
        memory.set_supervised()
        memory.action /= args.max_steps  # i.e. bound to [-1, 1]
        global test, test_supervised
        test = test_supervised
    else:
        raise NotImplementedError('Model <%s> not implemented' % args.method)

    network = BaseNetwork(args.num_features,
                          args.action_size,
                          args.num_uniform,
                          args.num_cem,
                          args.cem_iter,
                          args.cem_elite,
                          args.device,
                          args.bounds).to(args.device)

    # Load the model and an optional checkpoint
    model = Model(network, **vars(args))

    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)

    # Do a warm start so we don't spend time acting in the environment early on
    if args.checkpoint:
        warm_start_iters = 0
    else:
        warm_start_iters = int(10 * args.buffer_size // args.batch_size)

    losses = deque(maxlen=200)
    for it in range(warm_start_iters):

        loss = model.train(memory, **vars(args))
        losses.append(loss)

        if it % args.update_iter == 0:
            model.update()
            print('Warm start iter %d / %d, Loss: %2.4f' %
                  (it, warm_start_iters, np.mean(losses)))

        if it % 500 == 0:
            checkpoint = os.path.join(checkpoint_dir, '%d' % it)
            model.save_checkpoint(checkpoint)

            np.set_printoptions(2)
            for name, weight in model.model.named_parameters():
                w = torch.norm(weight).data.cpu().numpy()
                if hasattr(weight, 'grad'):
                    grad = torch.norm(weight.grad).data.cpu().numpy()
                else:
                    grad = None

                print(name, '\t', w, '\t', grad)
            print('')

    checkpoint = os.path.join(checkpoint_dir, 'final')
    model.save_checkpoint(checkpoint)

    # Train the model & simultaneously test in the environment
    reward_queue = deque(maxlen=200)
    for episode in range(args.max_episodes):

        start = time.time()

        '''
        model.train(memory, **kwargs)

        if episode % args.update_iter == 0:
            model.update()
        '''

        step, reward = test(env, model, memory, **vars(args))
        reward_queue.append(reward)

        print('Episode: %d, Step: %2d, Reward: %1.2f, Took: %2.4f' %
              (episode, step, np.mean(reward_queue), time.time() - start))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Off Policy Deep Q-Learning')

    # Model parameters
    parser.add_argument('--model', dest='method', default='dqn')
    parser.add_argument('--on-policy', action='store_true', default=False)
    parser.add_argument('--epochs', dest='max_episodes', default=1000, type=int)
    parser.add_argument('--data-dir', default='data100K2')
    parser.add_argument('--buffer-size', default=100000, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--block-random', default=0.3, type=float)

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
    parser.add_argument('--is-train', action='store_true', default=False)

    # Build model
    main(parser.parse_args())
