import chess
import numpy as np
import torch

class ChessEnv:
    """
    A lightweight chess environment for interaction with DQN agents.
    Handles board state, tensor conversion, legal move management,
    and action encoding/decoding between chess.Move and flat index space.
    """

    def __init__(self):
        self.board = chess.Board()
        self.last_promotion_fallback = False

    def reset(self):
        """
        Resets the board to the starting position and returns the encoded state tensor.
        """
        self.board.reset()
        return self.get_tensor()

    def get_tensor(self):
        """
        Encodes the current board into a 4D tensor with 18 feature planes:
        - 12 piece planes (6 types × 2 colors)
        - 4 castling rights planes
        - 1 en passant plane
        - 1 side-to-move plane
        Returns a tensor of shape [1, 18, 8, 8] for model input.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 18-channel encoding: 12 piece planes + castling + en passant + side to move
        planes = np.zeros((18, 8, 8), dtype=np.float32)

        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            channel = self._piece_to_channel(piece)
            planes[channel, row, col] = 1

        # Castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE): planes[12, :, :] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE): planes[13, :, :] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK): planes[14, :, :] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK): planes[15, :, :] = 1

        # En passant
        if self.board.ep_square is not None:
            row, col = divmod(self.board.ep_square, 8)
            planes[16, row, col] = 1

        # Side to move
        planes[17, :, :] = int(self.board.turn)

        return torch.tensor(planes).unsqueeze(0).to(device)  # shape: [1, 18, 8, 8]

    def _piece_to_channel(self, piece):
        """
        Maps a piece to its corresponding channel index.
        Channels 0–5 are white pieces, 6–11 are black.
        """
        piece_type = piece.piece_type - 1  # 0-indexed: P=0, ..., K=5
        return piece_type + (0 if piece.color == chess.WHITE else 6)

    def legal_moves(self):
        """
        Returns a list of legal chess.Move objects in the current position.
        """
        return list(self.board.legal_moves)

    def encode_action(self, move):
        """
        Encodes a chess.Move into a flat action index (0–4671).
        - Regular moves: 0–4095 (64 from-squares × 64 to-squares)
        - Promotions: 4096–4671 (64 to-squares × 4 promotion types)
        """
        from_sq = move.from_square
        to_sq = move.to_square

        # Promotion encoding (only 256 slots: 64 to-squares × 4 promo types)
        if move.promotion:
            promo_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
            promo_index = promo_map[move.promotion]
            return 4096 + (64 * promo_index) + to_sq

        # Basic move
        return 64 * from_sq + to_sq

    def decode_action(self, index):
        """
        Decodes an action index into a chess.Move.
        If the index encodes a promotion, searches for a matching legal move.
        If decoding fails, triggers fallback and returns a null move.
        """
        if index >= 4096:
            promo_index = (index - 4096) // 64
            to_sq = (index - 4096) % 64
            from_sq = None

            # Search for all legal promotions to find from_square
            promo_type_map = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            promo_type = promo_type_map[promo_index]

            for move in self.board.legal_moves:
                if move.to_square == to_sq and move.promotion == promo_type:
                    return move

            # Fallback (invalid decoded promotion)
            self.last_promotion_fallback = True
            return chess.Move.null()
        else:
            from_sq = index // 64
            to_sq = index % 64
            return chess.Move(from_sq, to_sq)

    def step(self, action_index):
        """
        Applies the decoded action to the board. Returns:
        - the new encoded state tensor
        - whether the game is done
        - whether a fallback (invalid promotion) was used
        """
        move = self.decode_action(action_index)

        fallback_triggered = self.last_promotion_fallback
        self.last_promotion_fallback = False

        if move not in self.board.legal_moves:
            self.board.push(chess.Move.null())  # still mutate the board state
            return self.get_tensor(), True, True  # done = True, fallback = True

        self.board.push(move)
        done = self.board.is_game_over()
        return self.get_tensor(), done, fallback_triggered

