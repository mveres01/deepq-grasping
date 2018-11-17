import os
import time
import numpy as np
from gym import spaces
import torch
from torch.utils.data import Dataset
import pybullet_envs.bullet.kuka_diverse_object_gym_env as e

from parallel import GymEnvironment
from models.base.memory import BaseMemory


import ray


class ContinuousDownwardBiasPolicy:
    """Policy which takes continuous actions, and is biased to move down.

    TODO: Fix gross requirements of get/set/load/save functions?

    Taken from:
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_kuka_diverse_object_grasping.py
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

        del step  # unused
        dx, dy, dz, da = self._action_space.sample()
        if np.random.random() < self._height_hack_prob:
            dz = -1
        return np.asarray([dx, dy, dz, da])


def make_env(env_config):
    """Makes a new environment given a config file."""

    def create():
        return e.KukaDiverseObjectEnv(**env_config)
    return create


def collect_experience(merge_every=25):

    SEED = None
    NUM_REMOTES = 2
    STATE_SIZE = (3, 64, 64)
    ACTION_SIZE = 4
    BUFFER_SIZE = 200_000

    # Defines parameters for distributed evaluation
    env_config = {'actionRepeat':80,
                  'isEnableSelfCollision':True,
                  'renders':False,
                  'isDiscrete':False,
                  'maxSteps':15,
                  'dv':0.06,
                  'removeHeightHack':True,
                  'blockRandom':0.3,
                  'cameraRandom':0,
                  'width':64,
                  'height':64,
                  'numObjects':5,
                  'isTest':False}

    env_creator = make_env(env_config)

    envs = []
    for _ in range(NUM_REMOTES):
        envs.append(GymEnvironment.remote(ContinuousDownwardBiasPolicy,
                                          env_creator, seed=SEED))


    memory = BaseMemory(BUFFER_SIZE)

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

    memory.save('data200K')


if __name__ == '__main__':

    collect_experience(merge_every=5)
