import os
import time
import numpy as np
import torch
import argparse
from collections import deque
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from utils import ReplayBuffer, collect_experience

import ray
ray.init()
time.sleep(1)

from visdom import Visdom
vis = Visdom(port=8097)
vis.close(None)


@ray.remote(num_cpus=1)
class GymEnvironment(object):
    """Wrapper to run an environment on a remote CPU."""

    def __init__(self, model_creator, env_creator, seed=None):

        if seed is None:
            seed = np.random.randint(1234567890)
    
        np.random.seed(seed)
        torch.manual_seed(seed)

        os.environ['MKL_NUM_THREADS'] = '1'

        self.model = model_creator()
        self.env = env_creator()
        state = self.reset()

        self.vis = Visdom(port=8097)
        self.win_id = self.vis.image(state.transpose(2, 0, 1))

        for p in self.model.model.parameters():
             p.requires_grad_(False)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def rollout(self, weights, num_episodes=5, explore_prob=0.):
        """Performs a full grasping episode in the environment."""

        self.model.set_weights(weights)

        steps, rewards = [], []
        for _ in range(num_episodes):

            step, done = 0, False
            state = self.reset()

            while not done:

                state = state.transpose(2, 0, 1)[np.newaxis]
                state = state.astype(np.float32) / 255.

                self.vis.image(state[0], win=self.win_id)

                # Sample and perform and action in the sim
                action = self.model.sample_action(state,
                                                  float(step),
                                                  explore_prob)
                action = action.cpu().numpy().flatten()

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


def make_model(Model, network_config):
    """Makes a new model given a config file."""

    def create():
        return Model(network_config)
    return create


def test(weights, envs, num_episodes=1):
    """Used for evaluating network performance.

    This can be used in parallel with the training loop to collect data on
    remote processes while the main process trains the network.
    """

    return [env.rollout.remote(weights, num_episodes) for env in envs]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Off Policy Deep Q-Learning')

    # Model parameters
    parser.add_argument('--model', default='dqn')
    parser.add_argument('--data-dir', default='data100K2')
    parser.add_argument('--buffer-size', default=100000, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--epochs', dest='max_episodes', default=200000, type=int)
    parser.add_argument('--on-policy', action='store_true', default=False)
    parser.add_argument('--explore-prob', default=0., type=float)

    # Hyperparameters
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--channels', dest='num_features', default=32, type=int)
    parser.add_argument('--gamma', default=0.92, type=float)
    parser.add_argument('--decay', default=1e-6, type=float)
    parser.add_argument('--lr', dest='lrate', default=5e-4, type=float)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--update', dest='update_iter', default=50, type=int)
    parser.add_argument('--uniform', dest='num_uniform', default=16, type=int)
    parser.add_argument('--cem', dest='num_cem', default=64, type=int)
    parser.add_argument('--cem-iter', default=3, type=int)
    parser.add_argument('--cem-elite', default=6, type=int)

    # Environment Parameters
    parser.add_argument('--max-steps', default=15, type=int)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--is-test', action='store_true', default=False)
    parser.add_argument('--block-random', default=0.3, type=float)
    parser.add_argument('--rollouts', default=8, type=int)
    parser.add_argument('--remotes', default=10, type=int)

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 

    if args.model == 'dqn':
        from dqn import DQN as Model
    elif args.model == 'ddqn':
        from ddqn import DDQN as Model
    elif args.model == 'ddpg':
        from ddpg import DDPG as Model
    else:
        raise NotImplementedError('Model <%s> not implemented' % args.model)

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

    # Defines parameters for network generator
    net_config = {'out_channels':args.num_features,
                  'action_size':4 if env_config['removeHeightHack'] else 3,
                  'num_uniform':args.num_uniform,
                  'num_cem':args.num_cem,
                  'cem_elite':args.cem_elite,
                  'cem_iter':args.cem_iter,
                  'device':torch.device('cpu'),  # Use cpu for remote eval
                  'bounds':(-1, 1)}

    model_config = {'seed':args.seed,
                    'gamma':args.gamma,
                    'decay':args.decay,
                    'lrate':args.lrate}
    net_config.update(model_config)


    # Make the remote environments
    model_creator = make_model(Model, net_config)
    env_creator = make_env(env_config)

    envs = []
    for _ in range(args.remotes):
        envs.append(GymEnvironment.remote(model_creator, env_creator, seed=None))

    # Create a trainable model
    net_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(Model, net_config)()
       
    if args.checkpoint is not None:
        model.load_checkpoint(args.checkpoint)


    if args.is_test:
    
        # Only test the policy, no training
        results = ray.get(test(model.get_weights(), envs, args.rollouts))

        steps, rewards = [], []
        for out in results:
            steps.extend(out[0])
            rewards.extend(out[1])

        print('Average across (%d) episodes: Step: %2.4f, Reward: %1.2f' %
              (args.rollouts * args.remotes, np.mean(steps), np.mean(rewards)))

    else:

        checkpoint_dir = os.path.join('checkpoints', args.model)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Use a memory replay buffer for training
        memory = ReplayBuffer(args.buffer_size,
                              state_size=(3, 64, 64),
                              action_size=(net_config['action_size'],))

        if os.path.exists(args.data_dir):
            memory.load(args.data_dir, args.buffer_size)
        else:
            collect_experience(env_creator(), memory, print_status_every=100)
            memory.save(args.data_dir)

        # Train the model & simultaneously test in the environment
        iters_per_epoch = args.buffer_size // args.batch_size

        scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, 6, gamma=0.1)


        step_queue = deque(maxlen=max(200, args.rollouts * args.remotes))
        reward_queue = deque(maxlen=step_queue.maxlen)

        n_steps = 50
        loss_plot = vis.line(Y=np.array([0]), X=np.array([0]),
                        opts=dict(
                            title="Episodic loss per %d steps."%n_steps,
                            caption="Episodic loss per %d steps."%n_steps))

        reward_plot = vis.line(Y=np.array([0]), X=np.array([0]),
                        opts=dict(
                            title="Episodic reward per %d steps."%n_steps,
                            caption="Episodic reward per %d steps."%n_steps))


        results = []
        start = time.time()
        for episode in range(args.max_episodes):

            loss = model.train(memory, **vars(args))

            if episode % args.update_iter == 0:
                model.update()


            # Validation step
            if episode % (iters_per_epoch // 1) == 0:

                ep = '%d' % (episode // iters_per_epoch)
                checkpoint = os.path.join(checkpoint_dir, ep)
                model.save_checkpoint(checkpoint)

                # This setup allows us to evaluate performance in parallel
                # with training. Since we collect results from the previous
                # validation step, we only need to wait to collect results,
                # rather then the full execution
                for device in ray.get(results):
                    step_queue.extend(device[0])
                    reward_queue.extend(device[1])

                results = test(model.get_weights(), envs, args.rollouts)
                
                #scheduler.step(np.mean(reward_queue)) 
                scheduler.step() 
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'], 1e-7)
                    print('LR: ', param_group['lr'])

                print('\nEpoch: %s, Episode: %d, Step: %2.4f, Reward: %1.2f, Took: %2.4f' %
                      (ep, episode, np.mean(step_queue), np.mean(reward_queue),
                       time.time() - start))

                vis.line(X=np.array([episode]),
                         Y=np.array([np.mean(reward_queue)]),
                         win=reward_plot,
                         update='append')

                start = time.time()

                torch.cuda.empty_cache()

            if episode % 50 == 0:
                vis.line(X=np.array([episode]),
                         Y=np.array([loss]),
                         win=loss_plot,
                         update='append')