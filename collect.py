import argparse
import numpy as np
import ray
from gym import spaces
from parallel import GymEnvironment, make_env
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

        del obs  # unused
        del step  # unused
        del explore_prob  # unused

        dx, dy, dz, da = self._action_space.sample()
        if np.random.random() < self._height_hack_prob:
            dz = -1
        return np.asarray([dx, dy, dz, da])


def collect_experience(max_steps, is_test, num_remotes, buffer_size,
                       seed, merge_every, outdir):

    # Defines parameters for distributed evaluation
    env_creator = make_env(max_steps, is_test, render=False)

    envs = []
    for _ in range(num_remotes):
        envs.append(GymEnvironment.remote(ContinuousDownwardBiasPolicy,
                                          env_creator, seed=seed))

    memory = BaseMemory(buffer_size)

    while not memory.is_full:

        rollouts = [env.rollout.remote(merge_every) for env in envs]

        for cpu in ray.get(rollouts):
            for batch in cpu:
                for step in batch:
                    if memory.is_full:
                        break

                    # batch = (s0, act, r, s1, done, steps)
                    memory.add(*step)

        print('Memory capacity: %d/%d' % (memory.cur_idx, memory.buffer_size))

    memory.save(outdir)


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
    collect_experience(**vars(args))
