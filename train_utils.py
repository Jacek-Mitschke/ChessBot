import random
from collections import deque

class ReplayBuffer:
    """
    An experience replay buffer for DQN training.
    Stores transitions of the form (state, action, reward, next_state, done),
    allowing the agent to sample minibatches for training.
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_index, reward, next_state, done):
        """
        Adds a transition to the buffer.

        Args:
            state: Current state tensor.
            action_index (int): Encoded action index.
            reward (float): Reward received after taking the action.
            next_state: Next state tensor.
            done (bool): Whether the episode ended after the action.
        """
        self.buffer.append((state, action_index, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current number of stored transitions.
        """
        return len(self.buffer)

def update_target_model(model, target_model):
    """
    Copies weights from the online model to the target model.

    This is typically done periodically in DQN to stabilize training.

    Args:
        model (nn.Module): The online DQN.
        target_model (nn.Module): The target DQN to be updated.
    """
    target_model.load_state_dict(model.state_dict())
