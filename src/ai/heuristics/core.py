import numpy as np
from .base import Heuristic

class EmptyTilesHeuristic(Heuristic):
    """Evaluates board based on number of empty tiles."""

    def __init__(self, name="EmptyTiles"):
        super().__init__(name)
        
    def evaluate(self, board):
        return np.count_nonzero(board == 0)


class MonotonicityHeuristic(Heuristic):
    """Evaluates how monotonic (ordered) the board is."""
    
    def __init__(self, name="Monotonicity"):
        super().__init__(name)
        
    def evaluate(self, board):
        monotonicity = 0
        
        # Check rows (left to right and right to left)
        for i in range(4):
            # Left to right
            row = board[i]
            if np.all(row[:-1] <= row[1:]):  # Non-decreasing
                monotonicity += 1
                
            # Right to left
            if np.all(row[:-1] >= row[1:]):  # Non-increasing
                monotonicity += 1
                
        # Check columns (top to bottom and bottom to top)
        for j in range(4):
            # Top to bottom
            col = board[:, j]
            if np.all(col[:-1] <= col[1:]):  # Non-decreasing
                monotonicity += 1
                
            # Bottom to top
            if np.all(col[:-1] >= col[1:]):  # Non-increasing
                monotonicity += 1
                
        return monotonicity


class SmoothnessHeuristic(Heuristic):
    """Evaluates smoothness: difference between adjacent tiles."""
    
    def __init__(self, name="Smoothness"):
        super().__init__(name)
        
    def evaluate(self, board):
        smoothness = 0
        
        # Horizontal smoothness
        for i in range(4):
            for j in range(3):
                if board[i, j] > 0 and board[i, j+1] > 0:
                    smoothness -= abs(np.log2(board[i, j]) - np.log2(board[i, j+1]))
                    
        # Vertical smoothness
        for i in range(3):
            for j in range(4):
                if board[i, j] > 0 and board[i+1, j] > 0:
                    smoothness -= abs(np.log2(board[i, j]) - np.log2(board[i+1, j]))
                    
        return smoothness


class MaxValueHeuristic(Heuristic):
    """Evaluates board based on maximum tile value."""
    
    def __init__(self, name="MaxValue"):
        super().__init__(name)
        
    def evaluate(self, board):
        return np.log2(np.max(board))


class CornerMaxHeuristic(Heuristic):
    """Advanced corner placement with graduated rewards/penalties."""
    
    def __init__(self, name="CornerMax"):
        super().__init__(name)
        
    def evaluate(self, board):
        max_tile = np.max(board)
        if max_tile == 0:
            return 0
            
        # Find position of max tile
        max_pos = np.unravel_index(np.argmax(board), board.shape)
        row, col = max_pos
        
        # Corner positions (best)
        if (row, col) in [(0,0), (0,3), (3,0), (3,3)]:
            return max_tile * 2.0
            
        # Edge positions (okay)
        elif row == 0 or row == 3 or col == 0 or col == 3:
            return max_tile * 0.5
            
        # Center positions (bad)
        else:
            return -max_tile * 1.0


class MergePotentialHeuristic(Heuristic):
    """Evaluates potential for merging adjacent tiles."""
    
    def __init__(self, name="MergePotential"):
        super().__init__(name)
        
    def evaluate(self, board):
        merges = 0
        
        # Check horizontal merges
        for i in range(4):
            for j in range(3):
                if board[i,j] == board[i,j+1] and board[i,j] > 0:
                    merges += board[i,j]
                    
        # Check vertical merges  
        for i in range(3):
            for j in range(4):
                if board[i,j] == board[i+1,j] and board[i,j] > 0:
                    merges += board[i,j]
                    
        return merges
