import os
import numpy as np
from gym import spaces
from torch.utils.data import Dataset


class ContinuousDownwardBiasPolicy(object):
    """Policy which takes continuous actions, and is biased to move down.

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

    def sample_action(self, obs, explore_prob):
        """Implements height hack and grasping threshold hack.
        """
        dx, dy, dz, da = self._action_space.sample()
        if np.random.random() < self._height_hack_prob:
            dz = -1
        return [dx, dy, dz, da]


def collect_experience(env, memory, print_status_every=25):

    # Initialize the experience replay buffer with memory
    policy = ContinuousDownwardBiasPolicy()

    total_step = 0
    while not memory.is_full:

        terminal = False
        state = env.reset()
        state = state.transpose(2, 0, 1)[np.newaxis]

        step = 0
        while not terminal and not memory.is_full:

            action = policy.sample_action(state, .1)

            next_state, reward, terminal, _ = env.step(action)
            next_state = next_state.transpose(2, 0, 1)[np.newaxis]

            memory.add(state, action, reward, next_state, terminal, step)
            state = next_state

            step = step + 1
            total_step = total_step + 1

            if total_step % print_status_every == 0:
                print('Memory capacity: %d/%d' % (memory.cur_idx, memory.buffer_size))


class BaseMemory(Dataset):

    def __init__(self, buffer_size, state_size, action_size, **kwargs):

        if not isinstance(state_size, tuple):
            raise Exception(':param state_size: must be type <tuple>')

        if not isinstance(action_size, tuple):
            action_size = (action_size, )

        self.cur_idx = 0
        self.is_full = False
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size

        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.terminal = None
        self.timestep = None

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return (np.float32(self.state[idx]) / 255.,
                np.float32(self.action[idx]),
                np.float32(self.reward[idx]),
                np.float32(self.next_state[idx]) / 255.,
                np.float32(self.terminal[idx]),
                np.float32(self.timestep[idx]))

    def add(self, state, action, reward, next_state, terminal, timestep):

        # In the off-policy setting, adding to a dataset means we're doing 
        # data collections. Initialize empty memory to hold necessary info
        # TODO: Find a more efficient way of doing this; memory map a file?
        if self.state is None:

            self.state = np.zeros((self.buffer_size,) + self.state_size, 
                                  dtype=np.uint8)
            self.action = np.zeros((self.buffer_size,) + self.action_size, 
                                   dtype=np.float32)
            self.reward = np.zeros((self.buffer_size,), dtype=np.float32)
            self.next_state = np.zeros((self.buffer_size,) + self.state_size, 
                                       dtype=np.uint8)
            self.terminal = np.zeros((self.buffer_size,), dtype=np.float32)
            self.timestep = np.zeros((self.buffer_size,), dtype=np.float32)

        batch_size = state.shape[0]
        store_idx = np.roll(np.arange(self.buffer_size), -self.cur_idx)[:batch_size]

        self.state[store_idx] = state
        self.action[store_idx] = action
        self.reward[store_idx] = reward
        self.next_state[store_idx] = next_state
        self.terminal[store_idx] = terminal
        self.timestep[store_idx] = timestep

        if self.cur_idx + batch_size >= self.buffer_size:
            self.is_full = True
        self.cur_idx = (self.cur_idx + batch_size) % self.buffer_size

    def load(self, data_dir, buffer_size=100000, **kwargs):
        """Loads a dataset using memory-mapping.

        In particular, the :state: and :next_state: variable of a dataset 
        take a large amount of space. To avoid OOM issues, load the entire 
        dataset as being memory-mapped that gets read when needed.
        """

        if not os.path.exists(data_dir):
            raise AssertionError('Data directory <%s> does not exist!' % data_dir)

        print('Memory-mapping state (mode=\'r\')... ')
        self.state = np.load(os.path.join(data_dir, 'state.npy'), mmap_mode='r')

        print('Memory-mapping action (mode=\'c\') ... ')
        self.action = np.load(os.path.join(data_dir, 'action.npy'), mmap_mode='c')

        print('Memory-mapping reward (mode=\'c\') ... ')
        self.reward = np.load(os.path.join(data_dir, 'reward.npy'), mmap_mode='c')

        print('Memory-mapping next_state (mode=\'r\')... ')
        self.next_state = np.load(os.path.join(data_dir, 'next_state.npy'), mmap_mode='r')

        print('Memory-mapping terminal (mode=\'r\')... ')
        self.terminal = np.load(os.path.join(data_dir, 'terminal.npy'), mmap_mode='r')

        print('Memory-mapping timestep (mode=\'r\')... ')
        self.timestep = np.load(os.path.join(data_dir, 'timestep.npy'), mmap_mode='r')

        print('Warning: Do not save memory unless you are aware of potential '
              'overwrites in the dataset.\n')

        if len(self.state) < self.buffer_size:
            raise ValueError('Requested %d samples, but dataset only has %d' % \
                             (self.buffer_size, len(self.state)))
       
        self.is_full = True
        self.cur_idx = self.buffer_size

    def sample(self, batch_size, balanced=False):

        # Dirty way to balance a minibatch by sampling an equal amount from
        # both positive and negative examples
        if balanced:
            neg = np.where(self.reward == 0)[0]
            neg = np.random.choice(neg, batch_size // 2, False)

            pos = np.where(self.reward == 1)[0]
            pos = np.random.choice(pos, batch_size // 2, False)

            batch_idx = np.hstack((pos, neg))
        else:
            upper_idx = self.buffer_size if self.is_full else self.cur_idx
            batch_idx = np.random.randint(0, upper_idx, batch_size)

        return self[batch_idx]

    def save(self, save_dir='.'):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('Saving state ... ')
        with open(os.path.join(save_dir, 'state.npy'), 'wb') as f:
            np.save(f, self.state, allow_pickle=False)

        print('Saving action ... ')
        with open(os.path.join(save_dir, 'action.npy'), 'wb') as f:
            np.save(f, self.action, allow_pickle=False)

        print('Saving reward ... ')
        with open(os.path.join(save_dir, 'reward.npy'), 'wb') as f:
            np.save(f, self.reward, allow_pickle=False)

        print('Saving next_state ... ')
        with open(os.path.join(save_dir, 'next_state.npy'), 'wb') as f:
            np.save(f, self.next_state, allow_pickle=False)

        print('Saving terminal ... ')
        with open(os.path.join(save_dir, 'terminal.npy'), 'wb') as f:
            np.save(f, self.terminal, allow_pickle=False)

        print('Saving timestep ... ')
        with open(os.path.join(save_dir, 'timestep.npy'), 'wb') as f:
            np.save(f, self.timestep, allow_pickle=False)
