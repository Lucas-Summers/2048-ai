# game.py
import numpy as np
from .board import Board

class Game2048:
    def __init__(self):
        self.board = Board()
        self.score = 0
        self.moves = 0
        self.game_over = False
        self.movement_history = []  # To track tile movements for animation
        
    def step(self, direction):
        """Make a move and update the game state, return movement info for animations"""
        # Store previous board state
        prev_grid = self.board.grid.copy()
        
        # Make the move
        moved = self.board.move(direction)
        
        if not moved:
            return {
                'moved': False,
                'movements': [],
                'merges': [],
                'new_tile': None
            }
        
        # Calculate what tiles moved where
        movements = []
        merges = []
        
        # Track which new positions have been filled
        filled_positions = set()
        
        # Check the board for tile movements and merges
        for row in range(4):
            for col in range(4):
                # If the current cell had a value before
                if prev_grid[row][col] != 0:
                    # Find where this tile went
                    found = False
                    for new_row in range(4):
                        for new_col in range(4):
                            # If this is a valid movement location
                            if (new_row, new_col) not in filled_positions:
                                if direction == 0:  # Up
                                    valid = new_row <= row and new_col == col
                                elif direction == 1:  # Right
                                    valid = new_row == row and new_col >= col
                                elif direction == 2:  # Down
                                    valid = new_row >= row and new_col == col
                                elif direction == 3:  # Left
                                    valid = new_row == row and new_col <= col
                                else:
                                    valid = False
                                
                                # If it's a valid destination and has the same value or is twice the value (merge)
                                if valid and self.board.grid[new_row][new_col] != 0:
                                    if self.board.grid[new_row][new_col] == prev_grid[row][col]:
                                        # This tile moved to this position
                                        movements.append({
                                            'from': (row, col),
                                            'to': (new_row, new_col),
                                            'value': prev_grid[row][col]
                                        })
                                        filled_positions.add((new_row, new_col))
                                        found = True
                                        break
                                    elif self.board.grid[new_row][new_col] == prev_grid[row][col] * 2:
                                        # This tile merged at this position
                                        movements.append({
                                            'from': (row, col),
                                            'to': (new_row, new_col),
                                            'value': prev_grid[row][col]
                                        })
                                        merges.append({
                                            'position': (new_row, new_col),
                                            'value': self.board.grid[new_row][new_col]
                                        })
                                        filled_positions.add((new_row, new_col))
                                        found = True
                                        break
                            
                        if found:
                            break
        
        # Find new tile
        new_tile = None
        for row in range(4):
            for col in range(4):
                if self.board.grid[row][col] != 0 and prev_grid[row][col] == 0:
                    new_tile = {
                        'position': (row, col),
                        'value': self.board.grid[row][col]
                    }
        
        # Update score and moves
        self.moves += 1
        self.score = np.sum(self.board.grid)  # Simple scoring
        
        # Check for game over
        self.game_over = self.board.is_game_over()
        
        return {
            'moved': True,
            'movements': movements,
            'merges': merges,
            'new_tile': new_tile
        }
        
    def is_game_over(self):
        return self.game_over
