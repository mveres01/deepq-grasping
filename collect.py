import time
import argparse
import numpy as np
import ray
from gym import spaces
from main import EnvWrapper
from factory import make_env
from models.base.memory import BaseMemory


class ContinuousDownwardBiasPolicy:
    """Policy which takes continuous actions, and is biased to move down.

    Taken from:
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_kuka_diverse_object_grasping.py

    TODO: Fix requirements of get/set/load/save functions?
    """

    def __init__(self, height_hack_prob=0.9):
        """Initializes the DownwardBiasPolicy.

        Args:
            height_hack_prob: The probability of moving down at every move.
        """
        self._height_hack_prob = height_hack_prob
        self._action_space = spaces.Box(low=-1, high=1, shape=(4,))

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    def load_checkpoint(self, checkpoint_dir):
        pass

    def save_checkpoint(self, checkpoint_dir):
        pass

    def sample_action(self, obs, step, explore_prob):
        """Implements height hack and grasping threshold hack.
        """
        dx, dy, dz, da = self._action_space.sample()
        if np.random.random() < self._height_hack_prob:
            dz = -1
        return np.asarray([dx, dy, dz, da])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Initialize memory')
    parser.add_argument('--max-steps', default=15, type=int)
    parser.add_argument('--is-test', action='store_true', default=False)
    parser.add_argument('--buffer-size', default=100000, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--remotes', dest='num_remotes', default=1, type=int)
    parser.add_argument('--merge-every', default=5, type=int,
                        help='Gather rollouts every K episodes')
    parser.add_argument('--outdir', default='data', type=str)

    args = parser.parse_args()

    ray.init(num_cpus=args.num_remotes)
    time.sleep(1)

    # Defines parameters for distributed evaluation
    env_creator = make_env(args.max_steps, args.is_test, render=False)

    envs = []
    for _ in range(args.num_remotes):
        envs.append(EnvWrapper.remote(env_creator, 
                                      ContinuousDownwardBiasPolicy,
                                      seed=args.seed))

    memory = BaseMemory(args.buffer_size)

    while not memory.is_full:

        rollouts = [env.rollout.remote(args.merge_every) for env in envs]

        for cpu in ray.get(rollouts):
            for batch in cpu:
                for step in batch:
                    if memory.is_full:
                        break

                    memory.add(*step)  #(s0, act, r, s1, done, steps)

        print('Memory capacity: %d/%d' % (memory.cur_idx, memory.buffer_size))

    memory.save(args.outdir)
