import numpy as np
from .base import Heuristic

class EmptyTilesHeuristic(Heuristic):
    """Evaluates board based on number of empty tiles."""

    def __init__(self, name="EmptyTiles"):
        super().__init__(name)
        
    def evaluate(self, board):
        return np.count_nonzero(board == 0)

class PositionalEmptyTilesHeuristic(Heuristic):
    """Empty tiles heuristic with positional weights"""
    
    def __init__(self, name="PositionalEmptyTiles"):
        super().__init__(name)
        
    def evaluate(self, board):
        score = 0
        for i in range(4):
            for j in range(4):
                if board[i,j] == 0:
                    if (i,j) in [(0,0), (0,3), (3,0), (3,3)]:  # Corners
                        score += 12  # High value for corner empties
                    elif i == 0 or i == 3 or j == 0 or j == 3:  # Edges
                        score += 8   # Medium value for edge empties  
                    else:  # Center
                        score += 4   # Lower value for center empties
        return score


class MonotonicityHeuristic(Heuristic):
    """Evaluates how monotonic (ordered) the board is."""
    
    def __init__(self, name="AjrMonotonicity"):
        super().__init__(name)
        
    def evaluate(self, board):
        mono_score = 0
        
        # Check each row and column (aj-r style)
        for i in range(4):
            # Row monotonicity
            row = [board[i,j] for j in range(4)]
            mono_score += self._calculate_line_monotonicity(row)
            
            # Column monotonicity  
            col = [board[j,i] for j in range(4)]
            mono_score += self._calculate_line_monotonicity(col)
        
        return mono_score
    
    def _calculate_line_monotonicity(self, line):
        """Rewards partial ordering."""
        inc_score = 0
        dec_score = 0
        prev_value = -1
        
        for value in line:
            if value == 0:
                value = 0  # Treat empty as 0
            
            inc_score += value
            if value <= prev_value or prev_value == -1:
                dec_score += value
                if value < prev_value and prev_value != -1:
                    # Penalty for breaking increasing order
                    inc_score -= prev_value
            prev_value = value
        
        # Return the better of increasing or decreasing direction
        return max(inc_score, dec_score)

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


class QualityStabilityHeuristic(Heuristic):
    """Measures board stability and resistance to quality loss."""
    
    def __init__(self, name="QualityStability"):
        super().__init__(name)
        
    def evaluate(self, board):
        stability = 0
        
        max_tile = np.max(board)
        max_pos = np.unravel_index(np.argmax(board), board.shape)
        row, col = max_pos
        
        # Check if max tile is protected (fewer exposed sides = more stable)
        protected_sides = 0
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        for di, dj in directions:
            ni, nj = row + di, col + dj
            if ni < 0 or ni >= 4 or nj < 0 or nj >= 4:
                protected_sides += 1  # Protected by wall
            elif board[ni,nj] != 0:
                protected_sides += 1  # Protected by another tile
        
        stability += protected_sides * max_tile * 0.1
        
        # Reward having second-highest tiles near max tile
        second_max = np.partition(board.flatten(), -2)[-2]
        if second_max > 0:
            for di, dj in directions:
                ni, nj = row + di, col + dj
                if 0 <= ni < 4 and 0 <= nj < 4:
                    if board[ni,nj] == second_max:
                        stability += second_max * 0.2
        
        return stability
