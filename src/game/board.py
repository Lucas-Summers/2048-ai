# board.py
import numpy as np
import random

class Board:
    def __init__(self, size=4):
        """Initialize a new 2048 board with the specified size."""
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        # Add initial tiles
        self.add_random_tile()
        self.add_random_tile()
        
    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell."""
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i, j] = 2 if random.random() < 0.9 else 4
            return True
        return False
            
    def get_empty_cells_count(self):
        """Return the number of empty cells."""
        return np.count_nonzero(self.grid == 0)
    
    def _slide_row(self, row):
        """Slide and merge tiles in a row to the left."""
        # Remove zeros
        row = row[row != 0]
        
        # Merge identical adjacent tiles
        i = 0
        while i < len(row) - 1:
            if row[i] == row[i + 1]:
                row[i] *= 2
                row[i + 1] = 0
                i += 2
            else:
                i += 1
        
        # Remove zeros again
        row = row[row != 0]
        
        # Pad with zeros to maintain size
        result = np.zeros(self.size, dtype=int)
        result[:len(row)] = row
        
        return result
    
    def move(self, direction):
        """Apply a move (0:up, 1:right, 2:down, 3:left) to the board.
        Returns True if the move changed the board, False otherwise."""
        # Save the original grid to check if the move changed anything
        original_grid = self.grid.copy()
        
        # Process the grid based on direction
        if direction == 0:  # Up
            for j in range(self.size):
                column = self.grid[:, j]
                column = self._slide_row(column)
                self.grid[:, j] = column
        elif direction == 1:  # Right
            for i in range(self.size):
                row = self.grid[i, :][::-1]  # Reverse row
                row = self._slide_row(row)
                self.grid[i, :] = row[::-1]  # Reverse back
        elif direction == 2:  # Down
            for j in range(self.size):
                column = self.grid[:, j][::-1]  # Reverse column
                column = self._slide_row(column)
                self.grid[:, j] = column[::-1]  # Reverse back
        elif direction == 3:  # Left
            for i in range(self.size):
                row = self.grid[i, :]
                row = self._slide_row(row)
                self.grid[i, :] = row
        
        # Check if the grid changed
        if not np.array_equal(original_grid, self.grid):
            self.add_random_tile()
            return True
        return False
    
    def get_available_moves(self):
        """Return a list of available moves (those that change the board)."""
        available_moves = []
        
        for direction in range(4):
            # Create a copy of the grid to test the move
            grid_copy = self.grid.copy()
            board_copy = Board(self.size)
            board_copy.grid = grid_copy
            
            if board_copy.move(direction):
                available_moves.append(direction)
        
        return available_moves
    
    def is_game_over(self):
        """Check if the game is over (no valid moves)."""
        return len(self.get_available_moves()) == 0
    
    def get_max_tile(self):
        """Return the value of the highest tile on the board."""
        return np.max(self.grid)
    
    def get_tile(self, row, col):
        """Get the value of a specific tile."""
        return self.grid[row, col]
    
    def __str__(self):
        """String representation of the board."""
        result = ""
        for i in range(self.size):
            for j in range(self.size):
                value = self.grid[i, j]
                if value == 0:
                    result += ".\t"
                else:
                    result += f"{value}\t"
            result += "\n"
        return result
