import numpy as np

from .base import Agent

class ExpectimaxAgent(Agent):
    """
    Agent that uses Expectimax search to find optimal moves in 2048.
    Accounts for randomness in tile placement.
    """
    
    def __init__(self, name="Expectimax", max_depth=3):
        super().__init__(name)
        self.max_depth = max_depth
    
    def get_move(self, game):
        """Entry point for the agent. Loop over all possible moves (up/down/left/right),
         simulate the result, and run _expectimax on each. Return the direction with
         the best expected score."""
        best_score = float('-inf')
        best_move = None

        for move in game.get_available_moves():
            next_state = game.copy()
            next_state.step(move)
            score = self._expectimax(next_state, self.max_depth - 1, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


    
    def _expectimax(self, game, depth, is_max_player):
        if depth == 0 or game.is_game_over():
             return self._evaluate_board(game.board.grid)

        if is_max_player: # At a max node; try all valid moves
            best_value = float('inf')
            for move in game.get_available_moves():
                next_state = game.copy()
                next_state.step(move) # Applies a move and adds a random tile
                value = self._expectimax(next_state, depth - 1, False)
                best_value = max(best_value, value)
            return best_value

        else: # Chance node (2 with 90% chance, 4 with 10% chance)
            empty = []
            for i in range(game.board.size):
                for j in range(game.board.size):
                    if game.board.grid[i,j] == 0:
                        empty.append((i,j))

            if len(empty) == 0: # No empty tiles
                return self._evaluate_board(game.board.grid)

            expected_value = 0
            for (i, j) in empty:
                for tile_value, probability in [(2,0.9),(4,0.1)]:
                    next_game = game.copy()
                    next_game.board.grid[i,j] = tile_value
                    value = self._expectimax(next_game, depth - 1, True)
                    expected_value += value * probability
            expected_value /= len(empty) # Take average val across all empty spots
            return expected_value


    
    def _evaluate_board(self, board):
        """ Evaluation of tiles. Empty tiles award 10 points each.
        Highest tile awards log(tile + 1) * 2 """
        empty_tiles = np.count_nonzero(board == 0)


        def num_empty_tiles(board):
            """More empty tiles = more flexibility and less risk.
            Reward boards with more 0s."""

            # MULTIPLIER: 10X
            return np.count_nonzero(board == 0) * 10

        def monotonicity(board):
            """Measures whether values increase or decrease consistently across rows
            or columns. Boards with decreasing rows/cols (e.g. [128, 64, 32, 16]) are
             easier to merge and build up."""
            def score_line(line):
                inc = 0
                dec = 0
                for i in range(3):
                    if line[i] <= line[i+1]:
                        inc += line[i+1] - line[i]
                    else:
                        dec += line[i] - line[i + 1]
                return -min(inc, dec)

            score = 0
            for row in board:
                score += score_line(row)
            for col in board.T:
                score += score_line(col)
            return score

        def smoothness(board):
            """Penalizes abrupt changes between adjacent tiles. Ideal boards have
            neighboring tiles that are similar in value (e.g. [16, 16] or [128, 64])."""
            penalty = 0
            for i in range(4):
                for j in range(4):
                    if board[i][j] == 0:
                        continue
                    val = np.log2(board[i][j]) # Use log to dilute penalty
                    for dx, dy in [(0,1),(1,0)]:
                        ni, nj = i + dx, j + dy
                        if ni < 4 and nj < 4 and board[ni][nj] != 0:
                            # If in range of board and not empty cell
                            neighbor_val = np.log2(board[ni][nj])
                            # Differnece between values
                            penalty -= abs(val - neighbor_val)

            return penalty

        def corner_max_tile(board):
            """Reward boards where the highest tile is in a corner (e.g. top-left).
            Strategic stacking tends to place the max tile there."""

            max_val = max_tile(board)
            corners = [board[0][0], board[0][-1], board[-1][0], board[-1][-1]]
            return 1 if max_val in corners else 0

        def max_tile(board):
            """Encourage growth toward the goal tile."""

            # MULTIPLIER: log(max) * 2
            return np.max(board)


        empty = num_empty_tiles(board)
        smooth = smoothness(board)
        mono = monotonicity(board)
        corner = corner_max_tile(board)
        max_val = np.log2(max_tile(board))

        # Weighted sum of heuristics
        return (
            2.5 * empty +
            1.0 * smooth +
            1.0 * mono +
            10.0 * corner +
            1.5 * max_val
        )
    def get_config(self):
        return {
            'max_depth': self.max_depth
        }
