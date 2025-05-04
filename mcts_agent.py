import math
import chess
import torch
from collections import defaultdict
import numpy as np

class MCTSNode:
    def __init__(self, board: chess.Board):
        """
        A node in the Monte Carlo Tree Search.

        Attributes:
            board: A copy of the current board state at this node.
            children: A dictionary mapping moves to child nodes.
            visits: A count of visits for each legal move.
            q_values: The average value estimate (Q) for each move.
            priors: The prior probabilities (P) from the policy network for each move.
        """
        self.board = board.copy()
        self.children = {}  # move: MCTSNode
        self.visits = defaultdict(int)  # move: N(s, a)
        self.q_values = defaultdict(float)  # move: Q(s, a)
        self.priors = {}  # move: P(s, a)

    def is_leaf(self):
        return len(self.children) == 0


class MCTSAgent:
    def __init__(self, model, simulations=100, c_puct=1.5, device="cuda"):
        """
        Monte Carlo Tree Search agent with neural guidance (PUCT).

        Args:
            model: A policy-value neural network.
            simulations: Number of MCTS rollouts per move selection.
            c_puct: Exploration constant for balancing exploitation/exploration.
            device: "cuda" or "cpu".
        """
        self.model = model.to(device)
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device

    def select_move(self, board):
        """
        Chooses a move from the current board using MCTS guided by the neural net.
        """
        root = MCTSNode(board)

        # Initial policy + value from ANN
        state_tensor = self.board_to_tensor(board).unsqueeze(0).to(self.device)
        policy_logits, _ = self.model(state_tensor)
        priors = torch.softmax(policy_logits, dim=1).squeeze().detach().cpu().numpy()

        for move in board.legal_moves:
            idx = self.encode_move(move)
            root.priors[move] = priors[idx]

        for _ in range(self.simulations):
            self.simulate(root)

        # Dirichlet Noise for Exploration at Root
        alpha = 0.3
        epsilon = 0.25
        legal_moves_list = list(board.legal_moves)
        noise = np.random.dirichlet([alpha] * len(legal_moves_list))
        for i, move in enumerate(legal_moves_list):
            root.priors[move] = (1 - epsilon) * root.priors[move] + epsilon * noise[i]

        # Pick most visited move
        move = max(root.visits, key=root.visits.get)
        return move

    def simulate(self, node):
        """
        Performs a single MCTS simulation from the given node.

        Returns:
            The estimated value of the board state (from the perspective of the current player).
        """
        board = node.board

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1
            elif result == "0-1":
                return -1
            else:
                return 0

        if node.is_leaf():
            # Expand
            state_tensor = self.board_to_tensor(board).unsqueeze(0).to(self.device)
            policy_logits, value = self.model(state_tensor)
            policy = torch.softmax(policy_logits, dim=1).squeeze().detach().cpu().numpy()
            value = value.item()

            for move in board.legal_moves:
                new_board = board.copy(stack=False)
                new_board.push(move)
                node.children[move] = MCTSNode(new_board)  # push move here
                idx = self.encode_move(move)
                node.priors[move] = policy[idx]

            return value

        # Select move using PUCT
        total_visits = sum(node.visits[m] for m in board.legal_moves)
        best_score = -float('inf')
        best_move = None

        for move in board.legal_moves:
            q = node.q_values[move]
            p = node.priors.get(move, 1e-5)
            n = node.visits[move]
            u = self.c_puct * p * math.sqrt(total_visits + 1) / (1 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            return 0  # fallback

        next_node = node.children[best_move]
        value = -self.simulate(next_node)

        # Update visit count and q value with an incremental average
        node.visits[best_move] += 1
        node.q_values[best_move] += (value - node.q_values[best_move]) / node.visits[best_move]
        return value

    def board_to_tensor(self, board):
        """
        Converts a chess board to a tensor of shape (18, 8, 8) for NN input.
        """
        planes = torch.zeros((18, 8, 8), dtype=torch.float32)
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            channel = self._piece_to_channel(piece)
            planes[channel, row, col] = 1

        if board.has_kingside_castling_rights(chess.WHITE): planes[12, :, :] = 1
        if board.has_queenside_castling_rights(chess.WHITE): planes[13, :, :] = 1
        if board.has_kingside_castling_rights(chess.BLACK): planes[14, :, :] = 1
        if board.has_queenside_castling_rights(chess.BLACK): planes[15, :, :] = 1
        if board.ep_square is not None:
            row, col = divmod(board.ep_square, 8)
            planes[16, row, col] = 1
        planes[17, :, :] = int(board.turn)
        return planes

    def _piece_to_channel(self, piece):
        """
        Maps a piece to the appropriate channel index in the input tensor.
        """
        piece_type = piece.piece_type - 1  # 0-indexed: P=0, ..., K=5
        return piece_type + (0 if piece.color == chess.WHITE else 6)

    def encode_move(self, move):
        """
        Encodes a chess move into an index for the neural network output vector.

        Returns:
            int: An index in [0, 4672) representing the move.
        """
        from_square = move.from_square
        to_square = move.to_square

        if move.promotion:
            promo_map = {'q': 0, 'r': 1, 'b': 2, 'n': 3}
            promo_type = promo_map[chess.piece_symbol(move.promotion)]
            index = 4096 + (64 * promo_type) + to_square
        else:
            index = 64 * from_square + to_square

        return index

