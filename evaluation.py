import csv
import os
import random
import time

import chess
from tqdm import tqdm  # Add this

from chess_env import ChessEnv
from dqn_agent import DQNAgent


def evaluate_agents(agent1, agent2, num_games=100, agent1_name="agent1", agent2_name="agent2", start_random_positions=False):
    """
    Runs a head-to-head evaluation between two agents over a specified number of games.
    Alternates colours, logs outcomes, timing, and move stats.

    Parameters:
    - agent1, agent2: Agents that implement either select_action (DQN) or select_move (Minimax/MCTS).
    - num_games: Number of games to simulate.
    - agent1_name, agent2_name: Strings used for naming and CSV logging.
    - start_random_positions: If True, random midgame states are sampled at start.
    """
    print(f"Evaluating {agent1_name.upper()} vs {agent2_name.upper()} for {num_games} games...")

    results = {
        f"{agent1_name}_wins": 0,
        f"{agent2_name}_wins": 0,
        "draws": 0,
        "fallbacks": 0,
        "total_moves": [],
        "agent1_times": [],
        "agent2_times": [],
    }

    os.makedirs("experiment_results", exist_ok=True)
    csv_path = f"experiment_results/evaluation_{agent1_name}_vs_{agent2_name}.csv"

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Game", "Winner", "Moves", "Agent1_Avg_Time", "Agent2_Avg_Time"])

        # Add tqdm for the game loop
        for game_idx in tqdm(range(num_games), desc=f"{agent1_name} vs {agent2_name}"):
            ...
            # (Rest of your code stays identical inside the loop)

            env = ChessEnv()
            board = env.board
            state = env.reset()
            done = False

            # Optional: start from random midgame position
            if start_random_positions:
                for _ in range(random.randint(2, 8)):
                    move = random.choice(list(board.legal_moves))
                    board.push(move)

            agent1_times = []
            agent2_times = []

            # Alternate colors
            if game_idx % 2 == 0:
                white_player = agent1
                black_player = agent2
                white_name = agent1_name
                black_name = agent2_name
            else:
                white_player = agent2
                black_player = agent1
                white_name = agent2_name
                black_name = agent1_name

            while not done:
                current_player = white_player if board.turn == chess.WHITE else black_player
                move_start = time.time()

                if isinstance(current_player, DQNAgent):
                    move, _, _, _ = current_player.select_action(state, list(board.legal_moves), env)
                else:
                    move = current_player.select_move(board)

                move_time = time.time() - move_start
                if board.turn == chess.WHITE:
                    agent1_times.append(move_time if white_player == agent1 else move_time)
                else:
                    agent2_times.append(move_time if black_player == agent2 else move_time)

                idx = env.encode_action(move)
                state, done, fallback = env.step(idx)

                if fallback:
                    results["fallbacks"] += 1

            result = board.result()
            if result == "1-0":
                winner = white_name
            elif result == "0-1":
                winner = black_name
            else:
                winner = "draw"

            if winner == agent1_name:
                results[f"{agent1_name}_wins"] += 1
            elif winner == agent2_name:
                results[f"{agent2_name}_wins"] += 1
            else:
                results["draws"] += 1

            results["total_moves"].append(board.fullmove_number)
            results["agent1_times"].append(sum(agent1_times) / len(agent1_times) if agent1_times else 0)
            results["agent2_times"].append(sum(agent2_times) / len(agent2_times) if agent2_times else 0)

            writer.writerow([game_idx+1, winner, board.fullmove_number,
                             round(sum(agent1_times) / max(1, len(agent1_times)), 4),
                             round(sum(agent2_times) / max(1, len(agent2_times)), 4)])

            print(f"Game {game_idx+1}/{num_games}: Winner - {winner}")

    print("\n=== Evaluation Summary ===")
    print(f"{agent1_name} Wins: {results[f'{agent1_name}_wins']}")
    print(f"{agent2_name} Wins: {results[f'{agent2_name}_wins']}")
    print(f"Draws: {results['draws']}")
    print(f"Fallbacks: {results['fallbacks']}")
    return results
