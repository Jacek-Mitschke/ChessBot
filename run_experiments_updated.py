import os

import torch
from torch.optim import Adam

from chess_net import StrongChessNet
from dqn_agent import DQNAgent
from evaluation import evaluate_agents
from mcts_agent import MCTSAgent
from minimax_agent import MinimaxAgent
from train_utils import ReplayBuffer


def load_dqn_agent(model_path):
    """
    Loads a trained DQN model and returns an initialized DQNAgent with frozen weights for evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrongChessNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_model = StrongChessNet().to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = Adam(model.parameters(), lr=1e-4)
    buffer = ReplayBuffer()

    agent = DQNAgent(model, target_model, optimizer, buffer, epsilon_start=0.0)
    return agent

def load_mcts_agent(model_path, simulations=500, c_puct=1.5):
    """
    Loads a trained model for use within an MCTS-guided agent.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrongChessNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    agent = MCTSAgent(model, simulations=simulations, c_puct=c_puct, device=device)
    return agent

def main():
    os.makedirs("experiment_results", exist_ok=True)

    # Paths
    model_path = "saved_models/dqn_model_curriculum_final.pt"

    # Load agents
    dqn_agent = load_dqn_agent(model_path)
    mcts_agent = load_mcts_agent(model_path)
    minimax_agent = MinimaxAgent(depth=3)

    # Choose which experiments to run
    experiments_to_run = [4]

    # Experiment 1: DQN vs Minimax (Standard Start)
    if 1 in experiments_to_run:
        print("\n=== Experiment 1: DQN vs Minimax from standard starting position ===")
        evaluate_agents(
            agent1=dqn_agent,
            agent2=minimax_agent,
            num_games=100,
            agent1_name="dqn",
            agent2_name="minimax",
            start_random_positions=False
        )

    # Experiment 2: DQN vs Minimax (Random Start)
    if 2 in experiments_to_run:
        print("\n=== Experiment 2: DQN vs Minimax from randomized midgame positions ===")
        evaluate_agents(
            agent1=dqn_agent,
            agent2=minimax_agent,
            num_games=100,
            agent1_name="dqn",
            agent2_name="minimax",
            start_random_positions=True
        )

    # Experiment 3: DQN vs MCTS (Standard Start)
    if 3 in experiments_to_run:
        print("\n=== Experiment 3: DQN vs MCTS from standard starting position ===")
        evaluate_agents(
            agent1=dqn_agent,
            agent2=mcts_agent,
            num_games=100,
            agent1_name="dqn",
            agent2_name="mcts",
            start_random_positions=False
        )

    # Experiment 4: MCTS vs Minimax (Standard Start)
    if 4 in experiments_to_run:
        print("\n=== Experiment 4: MCTS vs Minimax from standard starting position ===")
        evaluate_agents(
            agent1=mcts_agent,
            agent2=minimax_agent,
            num_games=100,
            agent1_name="mcts",
            agent2_name="minimax",
            start_random_positions=False
        )

if __name__ == "__main__":
    main()
