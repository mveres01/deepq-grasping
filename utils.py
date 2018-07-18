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
    while True and not memory.is_full:

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
                print('Memory capacity: %d/%d' % (memory.cur_idx, memory.max_size))


class ReplayBuffer(Dataset):
    def __init__(self, max_size, state_size, action_size):
        
        if not isinstance(state_size, tuple):
            raise Exception(':param state_size: must be type <tuple>')
        if not isinstance(action_size, tuple):
            action_size = (action_size, )

        self.cur_idx = 0
        self.is_full = False
        self.max_size = max_size
        self.state = np.zeros((max_size,) + state_size, dtype=np.uint8)
        self.action = np.zeros((max_size,) + action_size, dtype=np.float32)
        self.reward = np.zeros((max_size,), dtype=np.float32)
        self.next_state = np.zeros((max_size,) + state_size, dtype=np.uint8)
        self.terminal = np.zeros((max_size,), dtype=np.float32)
        self.timestep = np.zeros((max_size,), dtype=np.float32)

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        return self.sample(1, idx)

    def add(self, state, action, reward, next_state, terminal, timestep):

        batch_size = state.shape[0]
        store_idx = np.roll(np.arange(self.max_size), -self.cur_idx)[:batch_size]

        self.state[store_idx] = state
        self.action[store_idx] = action
        self.reward[store_idx] = reward
        self.next_state[store_idx] = next_state
        self.terminal[store_idx] = terminal
        self.timestep[store_idx] = timestep

        if self.cur_idx + batch_size >= self.max_size:
            self.is_full = True
        self.cur_idx = (self.cur_idx + batch_size) % self.max_size

    def sample(self, batch_size, batch_idx=None):

        if batch_idx is None:
            upper_idx = self.max_size if self.is_full else self.cur_idx
            batch_idx = np.random.randint(0, upper_idx, batch_size)

        return (np.float32(self.state[batch_idx]) / 255.,
                np.float32(self.action[batch_idx]),
                np.float32(self.reward[batch_idx]),
                np.float32(self.next_state[batch_idx]) / 255.,
                np.float32(self.terminal[batch_idx]),
                np.float32(self.timestep[batch_idx]))

    def sample_episode(self, batch_size, batch_idx=None):
        """Samples grasping episodes rather then single timesteps.

        This function will sample :batch_size: episodes from memory, 
        and return a generator that can be used to iterate over all 
        sampled episodes one-by-one. This allows us to doo messy 
        specification for uneven episode lengths
        """
     
        # Find where each episode in the memory buffer starts
        starts = np.where(self.timestep == 0)[0]

        # Pick :batch_size: random episodes
        batch_idx = np.random.choice(len(starts) - 1, batch_size, replace=False)

        for idx in batch_idx:
            
            steps = starts[idx + 1] - starts[idx] 
        
            yield self[starts[idx]:starts[idx+1]]

    def load(self, load_dir, max_buffer_size=10000):

        print('Loading state ... ')
        with open(os.path.join(load_dir, 'state.npy'), 'rb') as f:
            self.state = np.load(f)[:max_buffer_size].astype(np.uint8)

        print('Loading action ... ')
        with open(os.path.join(load_dir, 'action.npy'), 'rb') as f:
            self.action = np.load(f)[:max_buffer_size].astype(np.float32)

        print('Loading reward ... ')
        with open(os.path.join(load_dir, 'reward.npy'), 'rb') as f:
            self.reward = np.load(f)[:max_buffer_size].astype(np.float32)

        print('Loading next_state ... ')
        with open(os.path.join(load_dir, 'next_state.npy'), 'rb') as f:
            self.next_state = np.load(f)[:max_buffer_size].astype(np.uint8)

        print('Loading terminal ... ')
        with open(os.path.join(load_dir, 'terminal.npy'), 'rb') as f:
            self.terminal = np.load(f)[:max_buffer_size].astype(np.float32)

        print('Loading timestep ... ')
        with open(os.path.join(load_dir, 'timestep.npy'), 'rb') as f:
            self.timestep = np.load(f)[:max_buffer_size].astype(np.float32)

        self.is_full = True
        self.cur_idx = self.state.shape[0]
        self.max_buffer_size = self.state.shape[0]

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

    def set_supervised(self):
        """Converts independent actions and rewards to sequence-based.

        This is used for learning to grasp in the supervised learning case.
        + Action labels for each trial is given by: a_t = p_T - p_t
        + Episode labels for timestep in the episode is the episode reward
        """

        start = 0
        end = 1

        while end < self.state.shape[0]:

            while self.timestep[end] > self.timestep[start]:
                if end >= self.state.shape[0] - 1:
                    break
                end = end + 1

            # Convert the label for each element to be the straight-line
            # path from the action at the current time step & final action.
            # As each action is represented by [dx, dy, dz, d_theta], we can
            # just sum the actions for the episode.
            path = np.sum(self.action[start:end], axis=0)

            cur_path = 0
            for i in range(end - start):

                action = self.action[start + i].copy()

                self.action[start + i] = path - cur_path
               
                cur_path = cur_path + action

            assert np.all(cur_path - path == 0)
            assert np.all(self.reward[start:end - 1] == 0)

            # Set the reward for each timestep as the episode reward
            self.reward[start:end] = self.reward[end - 1]
            start = end
            end = end + 1
