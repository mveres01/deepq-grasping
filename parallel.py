import os
import time
import argparse
from collections import deque
import numpy as np
import torch
import ray
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from utils import EnvWrapper, make_env, make_model, make_memory


def test(envs, weights, rollouts, explore):
    """Helper function for running remote experiments."""

    for w in weights:
        for k, v in w.items():
            w[k] = v.cpu()
    return [env.rollout.remote(weights, rollouts, explore) for env in envs]


def parallel(args):
    """Main driver for evaluating different models.

    Can be used in both training and testing mode.
    """

    if args.seed is None:
        args.seed = np.random.randint(1234567890)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create a wrapper around the Gym env to run distributed
    @ray.remote(num_cpus=1) 
    class RemoteEnvWrapper(EnvWrapper):
        pass

    # Make the remote environments; the models aren't very large and can
    # be run fairly quickly on the cpus. Save the GPUs for training
    env_creator = make_env(args.max_steps, args.is_test, args.render)
    model_creator = make_model(args, torch.device('cpu'))

    envs = []
    for _ in range(args.remotes):
        envs.append(RemoteEnvWrapper.remote(env_creator, model_creator, args.seed_env))

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

        # Keep a running average of n-epochs worth of rollouts
        step_queue = deque(maxlen=1 * args.rollouts * args.remotes)
        reward_queue = deque(maxlen=step_queue.maxlen)
        loss_queue = deque(maxlen=step_queue.maxlen)

        # Perform a validation step every full pass through the data
        iters_per_epoch = args.buffer_size // args.batch_size

        results = []
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
            if episode % (iters_per_epoch // 1) == 0:

                cur_episode = '%d' % (episode // iters_per_epoch)
                model.save_checkpoint(os.path.join(checkpoint_dir, cur_episode))

                # Collect results from the previous epoch
                for device in ray.get(results):
                    for ep in device:
                        # (s0, act, r, s1, terminal, timestep)
                        step_queue.append(ep[-1][-1])
                        reward_queue.append(ep[-1][2])

                # Update weights of remote network & perform rollouts
                results = test(envs, model.get_weights(),
                               args.rollouts, args.explore)

                print('Epoch: %s, Step: %2.4f, Reward: %1.2f, Loss: %2.4f, '\
                      'Took:%2.4fs' % (cur_episode, np.mean(step_queue),
                      np.mean(reward_queue), np.mean(loss_queue),
                      time.time() - start))

                start = time.time()

    print('---------- Testing ----------')
    results = test(envs, model.get_weights(), args.rollouts, args.explore)

    steps, rewards = [], []
    for device in ray.get(results):
        for ep in device:
            # (s0, act, r, s1, terminal, timestep)
            steps.append(ep[-1][-1])
            rewards.append(ep[-1][2])

    print('Average across (%d) episodes: Step: %2.4f, Reward: %1.2f' %
          (args.rollouts * args.remotes, np.mean(steps), np.mean(rewards)))
