from abc import ABC, abstractmethod

class GenericPolicy:
    """Generic policy class used when some functions aren't needed."""

    def get_weights(self):
        """Returns the weights of the model as a tuple."""
        pass

    def set_weights(self, weights):
        """Sets the weights of the model."""
        pass

    def load_checkpoint(self, checkpoint_dir):
        """Loads a model given path to a checkpoint dir."""
        pass

    def save_checkpoint(self, checkpoint_dir):
        """Saves a model to a checkpoint dir."""
        pass

    def sample_action(self, obs, step, explore_prob):
        """Samples an action from the policy."""
        pass

    def train(self, memory, gamma, batch_size, **kwargs):
        """Updates the model using a batch of samples from memory."""
        pass

    def update(self):
        """Updates target networks if they exist."""
        pass


class BasePolicy(ABC):
    """Implements a generic policy class.

    Algorithms (e.g. DQN, DDQN, DDPG ... ) must override all functions
    defined below.
    """

    @abstractmethod
    def get_weights(self):
        """Returns the weights of the model as a tuple."""
        pass

    @abstractmethod
    def set_weights(self, weights):
        """Sets the weights of the model."""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir):
        """Loads a model given path to a checkpoint dir."""
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_dir):
        """Saves a model to a checkpoint dir."""
        pass

    @abstractmethod
    def sample_action(self, obs, step, explore_prob):
        """Samples an action from the policy."""
        pass

    @abstractmethod
    def train(self, memory, gamma, batch_size, **kwargs):
        """Updates the model using a batch of samples from memory."""
        pass

    @abstractmethod
    def update(self):
        """Updates target networks if they exist."""
        pass
