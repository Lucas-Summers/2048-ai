import numpy as np
from .base import Heuristic

class EmptyTilesHeuristic(Heuristic):
    """Evaluates board based on number of empty tiles."""
    
    def __init__(self, name="EmptyTiles"):
        super().__init__(name)
        
    def evaluate(self, board):
        """More empty tiles is better."""
        return np.count_nonzero(board == 0)


class MonotonicityHeuristic(Heuristic):
    """Evaluates how monotonic (ordered) the board is."""
    
    def __init__(self, name="Monotonicity"):
        super().__init__(name)
        
    def evaluate(self, board):
        """Higher score for more monotonic boards."""
        # Check increasing/decreasing patterns in rows and columns
        monotonicity_score = 0
        
        # Check rows (left to right and right to left)
        for i in range(4):
            # Left to right
            row = board[i]
            if np.all(row[:-1] <= row[1:]):  # Non-decreasing
                monotonicity_score += 1
                
            # Right to left
            if np.all(row[:-1] >= row[1:]):  # Non-increasing
                monotonicity_score += 1
                
        # Check columns (top to bottom and bottom to top)
        for j in range(4):
            # Top to bottom
            col = board[:, j]
            if np.all(col[:-1] <= col[1:]):  # Non-decreasing
                monotonicity_score += 1
                
            # Bottom to top
            if np.all(col[:-1] >= col[1:]):  # Non-increasing
                monotonicity_score += 1
                
        return monotonicity_score


class SmoothnesHeuristic(Heuristic):
    """Evaluates smoothness - differences between adjacent tiles."""
    
    def __init__(self, name="Smoothness"):
        super().__init__(name)
        
    def evaluate(self, board):
        """Higher score for smoother boards (smaller differences between neighbors)."""
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
        """Higher maximum value is better."""
        return np.max(board)
