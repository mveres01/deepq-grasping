import os
import time
import argparse
from collections import deque
import numpy as np
import torch
from utils import make_env, make_model, make_memory


class EnvWrapper:
    """Interface to a Gym env that enables distributed evaluation.

    Parameters
    ----------
    env_creator: callable function that creates a regular Gym Environment
    model_creator: callable function that returns an actor
    seed: random seed to use
    """

    def __init__(self, env_creator, model_creator, seed=None):

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = env_creator()
        self.policy = model_creator()

    def step(self, action):
        """Takes a step in the environment."""
        if not isinstance(action, np.ndarray):
            action = action.cpu().numpy().flatten()
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def rollout(self, weights, num_episodes=5, explore_prob=0.):
        """Performs a full grasping episode in the environment."""

        self.policy.set_weights(weights)

        episodes = []
        for _ in range(num_episodes):

            state = self.reset().transpose(2, 0, 1)[np.newaxis]
            step = 0.
            done = False

            cur_episode = []
            while not done:

                # Note state is normalized to [0, 1]
                s0 = state.astype(np.float32) / 255.
                action = self.policy.sample_action(s0, step, explore_prob)

                next_state, reward, done, _ = self.step(action)

                next_state = next_state.transpose(2, 0, 1)[np.newaxis]
                cur_episode.append((state, action, reward, next_state, done, step))

                state = next_state
                step = step + 1.

            episodes.append(cur_episode)

        return episodes


def test(envs, weights, rollouts, explore):
    """Helper function for evaluating current policy in environments."""

    start = time.time()
    for w in weights:
        for k, v in w.items():
            w[k] = v.cpu()
    return [env.rollout(weights, rollouts, explore) for env in envs]


def main(args):
    """Main driver for evaluating different models.

    Can be used in both training and testing mode.
    """

    if args.seed is None:
        args.seed = np.random.randint(1234567890)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Make the remote environments; the models aren't very large and can
    # be run fairly quickly on the cpus. Save the GPUs for training
    env_creator = make_env(args.max_steps, args.is_test, args.render)
    model_creator = make_model(args, torch.device('cpu'))

    envs = []
    for _ in range(args.remotes):
        envs.append(EnvWrapper(env_creator, model_creator, args.seed_env))

    # We'll put the trainable model on the GPU if one's available
    device = torch.device('cpu' if args.no_cuda or not
                          torch.cuda.is_available() else 'cuda')
    model = make_model(args, device)()

    if args.checkpoint is not None:
        model.load_checkpoint(args.checkpoint)

    # Train
    if not args.is_test:

        checkpoint_dir = os.path.join('checkpoints', args.model)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Some methods have specialized memory implementations
        memory = make_memory(args.model, args.buffer_size)
        memory.load(**vars(args))

        # Perform a validation step every full pass through the data
        iters_per_epoch = args.buffer_size // args.batch_size

        # Keep a running average of n-epochs worth of rollouts
        step_queue = deque(maxlen=1 * args.rollouts * args.remotes)
        reward_queue = deque(maxlen=step_queue.maxlen)
        loss_queue = deque(maxlen=iters_per_epoch)

        start = time.time()
        for episode in range(args.max_epochs * iters_per_epoch):

            loss = model.train(memory, **vars(args))
            loss_queue.append(loss)

            if episode % args.update_iter == 0:
                model.update()

            # Validation step;
            # Here we take the weights from the current network, and distribute
            # them to all remote instances. While the network trains for another
            # epoch, these instances will run in parallel & evaluate the policy.
            # If an epoch finishes before remote instances, training will be
            # halted until outcomes are returned
            if (episode + 1) % iters_per_epoch == 0:

                cur_episode = '%d' % (episode // iters_per_epoch)
                model.save_checkpoint(os.path.join(checkpoint_dir, cur_episode))

                # Collect results from the previous epoch
                for device in test(envs, model.get_weights(), 
                                   args.rollouts, args.explore):
                    for ep in device:
                        # (s0, act, r, s1, terminal, timestep)
                        step_queue.append(ep[-1][-1])
                        reward_queue.append(ep[-1][2])

                print('Epoch: %s, Step: %2.4f, Reward: %1.2f, Loss: %2.4f, '\
                      'Took:%2.4fs' %
                      (cur_episode, np.mean(step_queue), np.mean(reward_queue),
                       np.mean(loss_queue), time.time() - start))

                start = time.time()

    print('---------- Testing ----------')

    steps, rewards = [], []
    for device in test(envs, model.get_weights(), args.rollouts, args.explore):
        for ep in device:
            # (s0, act, r, s1, terminal, timestep)
            steps.append(ep[-1][-1])
            rewards.append(ep[-1][2])

    print('Average across (%d) episodes: Step: %2.4f, Reward: %1.2f' %
          (args.rollouts * args.remotes, np.mean(steps),
                     np.mean(rewards)))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Off Policy Deep Q-Learning')

    # Model parameters
    parser.add_argument('--model', default='dqn',
                        choices=['dqn', 'ddqn', 'ddpg', 'supervised', 'mcre', 'cmcre'])
    parser.add_argument('--data-dir', default='data100K')
    parser.add_argument('--buffer-size', default=100000, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--epochs', dest='max_epochs', default=200, type=int)
    parser.add_argument('--explore', default=0.0, type=float)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    # Hyperparameters
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--seed-env', default=None, type=int)
    parser.add_argument('--channels', dest='out_channels', default=32, type=int)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--decay', default=1e-5, type=float)
    parser.add_argument('--lr', dest='lrate', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--update', dest='update_iter', default=50, type=int)
    parser.add_argument('--uniform', dest='num_uniform', default=16, type=int)
    parser.add_argument('--cem', dest='num_cem', default=64, type=int)
    parser.add_argument('--cem-iter', default=3, type=int)
    parser.add_argument('--cem-elite', default=10, type=int)

    # Environment Parameters
    parser.add_argument('--max-steps', default=15, type=int)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--test', dest='is_test', action='store_true', default=False)

    # Distributed Parameters
    parser.add_argument('--rollouts', default=8, type=int)
    parser.add_argument('--remotes', default=1, type=int, choices=[1])

    args = parser.parse_args()
    main(args)
