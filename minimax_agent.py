import chess
import chess.engine
import random

class MinimaxAgent:
    def __init__(self, depth=2, stochastic=False):
        """
        Minimax agent with optional stochastic tie-breaking.

        Args:
            depth (int): The search depth for minimax evaluation.
            stochastic (bool): Whether to break ties randomly among best moves.
        """
        self.depth = depth
        self.stochastic = stochastic

    def select_move(self, board: chess.Board):
        """
        Selects the best move from the current board state using minimax search.

        If stochastic mode is enabled, randomly chooses between equally scored best moves.
        """
        if self.stochastic:
            maximizing = board.turn  # True for White, False for Black
            best_score = float('-inf') if maximizing else float('inf')
            best_moves = []

            for move in board.legal_moves:
                board.push(move)
                score = self._minimax(board, self.depth - 1, float('-inf'), float('inf'), not board.turn)
                board.pop()

                if (maximizing and score > best_score) or (not maximizing and score < best_score):
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

            return random.choice(best_moves) if best_moves else random.choice(list(board.legal_moves))

        else:
            best_score = float('-inf') if board.turn else float('inf')
            best_move = None

            for move in board.legal_moves:
                board.push(move)
                score = self._minimax(board, self.depth - 1, float('-inf'), float('inf'), not board.turn)
                board.pop()

                if board.turn and score > best_score:
                    best_score = score
                    best_move = move
                elif not board.turn and score < best_score:
                    best_score = score
                    best_move = move

            return best_move or random.choice(list(board.legal_moves))

    def _minimax(self, board, depth, alpha, beta, is_maximizing):
        """
        Recursive minimax algorithm with alpha-beta pruning.

        Args:
            board (chess.Board): The current board state.
            depth (int): Remaining search depth.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            is_maximizing (bool): True if maximizing player's turn.

        Returns:
            float: Evaluation score of the position.
        """
        if depth == 0 or board.is_game_over():
            return self._evaluate(board)

        if is_maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate(self, board):
        """
        Evaluates the board based on material count.

        Args:
            board (chess.Board): The board to evaluate.

        Returns:
            float: Material difference (positive for White advantage).
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        score = 0
        for piece_type in piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        return score
