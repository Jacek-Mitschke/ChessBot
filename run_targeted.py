import os
import csv
import time
import torch
import random
from tqdm import tqdm
from chess_env import ChessEnv
from dqn_agent import DQNAgent
from chess_net import StrongChessNet
from train_utils import ReplayBuffer
from torch.optim import Adam
from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent
import chess

# Patch DQNAgent with select_move so it works in self-play
def dqn_select_move_wrapper(self, board):
    """
    Allows the DQN agent to be used in place of a select_move-style agent.
    Converts the board to a tensor state and returns the best legal move.
    """
    env = ChessEnv()
    env.board = board.copy(stack=False)
    state = env.get_tensor()
    legal_moves = list(board.legal_moves)
    move, _, _, _ = self.select_action(state, legal_moves, env)
    return move

DQNAgent.select_move = dqn_select_move_wrapper


def run_targeted_dqn_evaluation(model_path="saved_models_non_stochastic/dqn_model_curriculum_final.pt", output_dir="targeted_experiments", num_games=10):
    """
    Runs a set of targeted evaluation experiments for the DQN agent.
    Records Q-value confidence, performance, move times, and ASCII logs.

    Evaluations:
    - DQN vs Minimax
    - DQN vs itself (self-play)
    - DQN vs MCTS (with midgame positions)
    - DQN vs Random (scripted human benchmark)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load DQN agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrongChessNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    agent = DQNAgent(model, model, Adam(model.parameters()), ReplayBuffer(), epsilon_start=0.0)
    DQNAgent.select_move = dqn_select_move_wrapper

    # Define a simple scripted opponent (random move)
    class RandomAgent:
        def select_move(self, board):
            return random.choice(list(board.legal_moves))

    # Opponents to test against
    opponents = {
        "minimax_qval": MinimaxAgent(depth=3),
        "dqn_selfplay": agent,
        "mcts_midgame": MCTSAgent(model=model, simulations=300, c_puct=1.5, device=device),
        "scripted_human": RandomAgent()
    }

    for name, opponent in opponents.items():
        print(f"\n=== Running {name} evaluation ===")
        csv_path = os.path.join(output_dir, f"dqn_vs_{name}.csv")

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Game", "Winner", "Moves",
                "QChosen_Mean", "QMax_Mean", "QMean_Mean",
                "DQN_Avg_Time", "Opponent_Avg_Time"
            ])

            for game_idx in tqdm(range(num_games), desc=f"DQN vs {name}"):
                env = ChessEnv()
                board = env.board
                state = env.reset()
                done = False
                dqn_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK

                q_vals = {"chosen": [], "max": [], "mean": []}
                dqn_times, opp_times = [], []
                ascii_frames = []

                # Optional midgame noise for adaptability test
                if "midgame" in name:
                    for _ in range(random.randint(2, 6)):
                        board.push(random.choice(list(board.legal_moves)))

                while not done:
                    is_dqn_turn = board.turn == dqn_color
                    move_start = time.time()

                    if is_dqn_turn:
                        move, q_c, q_max, q_mean = agent.select_action(state, list(board.legal_moves), env)
                        q_vals["chosen"].append(q_c)
                        q_vals["max"].append(q_max)
                        q_vals["mean"].append(q_mean)
                        dqn_times.append(time.time() - move_start)
                    else:
                        move = opponent.select_move(board)
                        opp_times.append(time.time() - move_start)

                    ascii_frames.append(board.unicode())
                    idx = env.encode_action(move)
                    state, done, _ = env.step(idx)

                result = board.result()
                if result == "1-0":
                    winner = "dqn" if dqn_color == chess.WHITE else name
                elif result == "0-1":
                    winner = "dqn" if dqn_color == chess.BLACK else name
                else:
                    winner = "draw"

                writer.writerow([
                    game_idx + 1, winner, board.fullmove_number,
                    round(sum(q_vals["chosen"]) / len(q_vals["chosen"]), 4) if q_vals["chosen"] else 0,
                    round(sum(q_vals["max"]) / len(q_vals["max"]), 4) if q_vals["max"] else 0,
                    round(sum(q_vals["mean"]) / len(q_vals["mean"]), 4) if q_vals["mean"] else 0,
                    round(sum(dqn_times) / len(dqn_times), 4) if dqn_times else 0,
                    round(sum(opp_times) / len(opp_times), 4) if opp_times else 0
                ])

                # Save ASCII game log
                ascii_path = os.path.join(output_dir, f"game_{name}_{game_idx+1}.txt")
                with open(ascii_path, "w", encoding="utf-8") as log_file:
                    log_file.write("\n\n".join(ascii_frames))

    print("\n Targeted DQN evaluation complete. Results saved to", output_dir)

if __name__ == "__main__":
    run_targeted_dqn_evaluation()