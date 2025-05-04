import os
import random
import copy
import csv
import torch
import chess
from chess_net import StrongChessNet
from dqn_agent import DQNAgent
from chess_env import ChessEnv
from train_utils import ReplayBuffer, update_target_model
from minimax_agent import MinimaxAgent
from tqdm import tqdm

SAVE_DIR = "saved_models"
LOG_DIR = "logs"
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "dqn_model_curriculum_final.pt")
TARGET_UPDATE_EVERY = 100
SAVE_EVERY = 100

# Create folders if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_dqn_selfplay_curriculum(episodes=1000):
    """
    Trains a DQN chess agent via curriculum-based self-play.
    The curriculum gradually introduces stronger opponents (random → self → past versions → Minimax).
    Rewards are shaped for strategic behaviour, and logs are written for later analysis.

    Args:
        episodes (int): Number of self-play training episodes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ChessEnv()
    model = StrongChessNet().to(device)
    target_model = StrongChessNet().to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=50000)
    agent = DQNAgent(model, target_model, optimizer, replay_buffer)

    past_agents = [] # Stores snapshots of past agents
    central_squares = [chess.D4, chess.E4, chess.D5, chess.E5, chess.C4, chess.F4, chess.C5, chess.F5]

    # Open log file once
    log_file = os.path.join(LOG_DIR, "training_log_curriculum.csv")
    with open(log_file, mode="w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["Episode", "TotalReward", "Steps", "DQNColor", "OpponentType",
                         "QChosen", "QMax", "QMean", "FinalResult", "UsedFallback"])

        for episode in tqdm(range(episodes), desc="Training Progress"):
            state_tensor = env.reset()
            done = False
            total_reward = 0
            q_chosen = q_max = q_mean = steps = 0
            fallback_used = False
            dqn_color = "White" if env.board.turn == chess.WHITE else "Black"

            # Curriculum opponent selection
            if episode < 300:
                opponent_type = "random"
            elif episode < 700:
                opponent_type = random.choices(["random", "self", "past"], [0.2, 0.5, 0.3])[0]
            else:
                opponent_type = random.choices(["self", "past", "minimax"], [0.3, 0.3, 0.4])[0]

            if episode % 100 == 0 and episode > 0:
                past_agents.append(copy.deepcopy(agent))

            position_counts = {}

            while not done:
                position_key = env.board.fen()
                count = position_counts.get(position_key, 0)
                position_counts[position_key] = count + 1

                is_dqn_turn = (env.board.turn == (chess.WHITE if dqn_color == "White" else chess.BLACK))

                soft_penalty = -0.05 * count if count >= 1 and is_dqn_turn else 0

                # Action selection
                if is_dqn_turn:
                    action, q_chosen, q_max, q_mean = agent.select_action(state_tensor, list(env.board.legal_moves), env)
                else:
                    legal_moves = list(env.board.legal_moves)
                    if opponent_type == "random":
                        action = random.choice(legal_moves)
                    elif opponent_type == "self":
                        action, _, _, _ = agent.select_action(state_tensor, legal_moves, env)
                    elif opponent_type == "past" and past_agents:
                        past_agent = random.choice(past_agents)
                        action, _, _, _ = past_agent.select_action(state_tensor, legal_moves, env)
                    elif opponent_type == "minimax":
                        action = MinimaxAgent(depth=3).select_move(env.board)
                    else:
                        action = random.choice(legal_moves)

                action_idx = env.encode_action(action)
                next_state_tensor, done, fallback_used = env.step(action_idx)

                # Reward shaping for strategic behaviour
                reward = soft_penalty
                piece = env.board.piece_at(action.from_square)
                if piece:
                    if piece.piece_type == chess.PAWN:
                        rank_advancement = chess.square_rank(action.to_square) - chess.square_rank(action.from_square)
                        if (env.board.turn == chess.WHITE and rank_advancement > 0) or \
                           (env.board.turn == chess.BLACK and rank_advancement < 0):
                            reward += 0.05

                    if env.board.is_capture(action):
                        reward += 0.2

                    if action.to_square in central_squares:
                        reward += 0.1

                    if env.board.is_castling(action):
                        reward += 0.15

                if done:
                    result = env.board.result()
                    if env.board.is_repetition(3):
                        reward = -0.9
                    elif result == "1-0":
                        reward = 1 if dqn_color == "White" else -1
                    elif result == "0-1":
                        reward = 1 if dqn_color == "Black" else -1
                    else:
                        reward = min(0.0, -0.8 + 0.8 * (episode / episodes))

                if is_dqn_turn:
                    replay_buffer.push(state_tensor, action_idx, reward, next_state_tensor, done)
                    agent.train_step()

                total_reward += reward
                state_tensor = next_state_tensor
                steps += 1

            # Save model and update target periodically
            if episode % TARGET_UPDATE_EVERY == 0 and episode > 0:
                update_target_model(model, target_model)
                print(f"[Episode {episode}] Target network updated.")

            if episode % SAVE_EVERY == 0 and episode > 0:
                checkpoint_path = os.path.join(SAVE_DIR, f"dqn_model_curriculum_{episode}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[Episode {episode}] Checkpoint saved at {checkpoint_path}")

            # Log training progress
            result = env.board.result()
            writer.writerow([
                episode,
                round(total_reward, 4),
                steps,
                dqn_color,
                opponent_type,
                round(q_chosen, 4),
                round(q_max, 4),
                round(q_mean, 4),
                result,
                int(fallback_used)
            ])

    # Final model save
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Training complete. Final model saved at {FINAL_MODEL_PATH}.")

if __name__ == "__main__":
    train_dqn_selfplay_curriculum()
