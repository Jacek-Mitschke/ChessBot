import torch
import random
import numpy as np
from torch.nn import functional as F

class DQNAgent:
    def __init__(self, model, target_model, optimizer, replay_buffer, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=10000):
        """
                Initializes the DQN agent with the given model, target model, optimizer,
                and replay buffer. Also sets epsilon-greedy exploration parameters.

                Args:
                    gamma: Discount factor for future rewards.
                    epsilon_start: Initial exploration rate.
                    epsilon_final: Minimum exploration rate.
                    epsilon_decay: Number of steps over which epsilon decays exponentially.
                """
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps = 0

    def select_action(self, state_tensor, legal_moves, env):
        """
        Selects an action using epsilon-greedy strategy. If exploring, picks a random legal move.
        Otherwise, selects the move with the highest predicted Q-value among legal actions.

        Returns:
            (move, q_chosen, q_max, q_mean) for logging/debugging.
        """
        self.steps += 1
        eps_threshold = self.epsilon_final + (self.epsilon - self.epsilon_final) * \
                        np.exp(-1. * self.steps / self.epsilon_decay)

        if random.random() < eps_threshold:
            # Random move (exploration)
            move = random.choice(legal_moves)
            return move, -1.0, -1.0, -1.0 # dummy Q-values
        else:
            # Exploitation - choose move with highest q value
            with torch.no_grad():
                q_values, _ = self.model(state_tensor)  # shape: [1, 4672]
                q_values = q_values.squeeze().cpu().numpy()

            legal_indices = [idx for idx in [env.encode_action(m) for m in legal_moves] if 0 <= idx < 4672]
            best_index = max(legal_indices, key=lambda idx: q_values[idx])

            q_legal = [q_values[idx] for idx in legal_indices]
            q_chosen = q_values[best_index]

            best_move = legal_moves[legal_indices.index(best_index)]
            return best_move, q_chosen, max(q_legal), np.mean(q_legal)

    def train_step(self, batch_size=64):
        """
        Performs a single training step using a mini-batch sampled from the replay buffer.

        Uses Double DQN logic:
            - Action selection from online network
            - Action evaluation from target network
        """

        if len(self.replay_buffer) < batch_size:
            return # not enough samples to train

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.cat(states).to("cuda")
        next_states = torch.cat(next_states).to("cuda")
        actions = torch.tensor(actions, dtype=torch.long).to("cuda")
        rewards = torch.tensor(rewards, dtype=torch.float32).to("cuda")
        dones = torch.tensor(dones, dtype=torch.bool).to("cuda")

        # Get predicted q values for chosen actions
        q_values, _ = self.model(states)  # shape: [batch, 4672]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: use online net to select action, target net to evaluate it
            next_q_values_online, _ = self.model(next_states)
            next_q_values_target, _ = self.target_model(next_states)
            best_actions = next_q_values_online.argmax(1)
            max_next_q_values = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # Q learning target
            target_q = rewards + self.gamma * max_next_q_values * (~dones)

        # Standard MSE loss between predicted and target Q-values
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
