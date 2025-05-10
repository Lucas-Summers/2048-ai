import numpy as np
import random

class Board:
    def __init__(self, size=4):
        """Initialize a new 2048 board with the specified size."""
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.last_move_info = None  # To store info about the last move

        # Add initial tiles
        self.add_random_tile()
        self.add_random_tile()
        
    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell."""
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            value = 2 if random.random() < 0.9 else 4
            self.grid[i, j] = value
            
            # Return the new tile information
            return {'position': (i, j), 'value': value}
        return None
            
    def get_empty_cells_count(self):
        """Return the number of empty cells."""
        return np.count_nonzero(self.grid == 0)
    
    def _slide_row(self, row):
        """
        Slide and merge tiles in a row to the left, tracking movements and merges.
        
        Args:
            row: Row to slide and merge
            
        Returns:
            tuple: (new_row, score, movements, merges)
        """
        # Create copies for comparison
        original_row = row.copy()
        new_row = np.zeros_like(row)
        score = 0
        movements = []
        merges = []
        
        # Remove zeros and track positions
        non_zero_indices = [i for i, val in enumerate(row) if val != 0]
        non_zero_values = row[non_zero_indices]
        
        # Place non-zero values at the beginning of the new row
        write_idx = 0
        i = 0
        while i < len(non_zero_values):
            val = non_zero_values[i]
            
            # Check if we should merge with the next value
            if i + 1 < len(non_zero_values) and val == non_zero_values[i + 1]:
                # Merge
                new_row[write_idx] = val * 2
                score += val * 2
                
                # Record original positions of both merged tiles
                merges.append({
                    'position': write_idx,
                    'value': val * 2,
                    'merged_from': [non_zero_indices[i], non_zero_indices[i + 1]]
                })
                
                # Skip the next value by incrementing i by 2
                i += 2
            else:
                # No merge, just move
                new_row[write_idx] = val
                
                # Record movement if needed
                if write_idx != non_zero_indices[i]:
                    movements.append({
                        'from': non_zero_indices[i],
                        'to': write_idx,
                        'value': val
                    })
                
                # Move to the next value
                i += 1
            
            # Advance write position
            write_idx += 1
        
        return new_row, score, movements, merges

    def _apply_move(self, grid, direction):
        """
        Apply move logic to a grid and return detailed information.
        Does not modify the original grid.
        
        Args:
            grid: The grid to analyze
            direction: Direction (0:up, 1:right, 2:down, 3:left)
            
        Returns:
            tuple: (moved, score, new_grid, movements, merges)
        """
        # Create a copy of the grid to modify
        new_grid = grid.copy()
        score = 0
        all_movements = []
        all_merges = []
        
        # Transform row/column indices based on direction
        if direction == 0:  # Up
            for col in range(self.size):
                # Extract column
                column = new_grid[:, col].copy()
                # Process as if sliding left
                new_column, col_score, col_movements, col_merges = self._slide_row(column)
                # Update grid with new column
                new_grid[:, col] = new_column
                score += col_score
                
                # Adjust movement indices to reflect column orientation
                for move in col_movements:
                    all_movements.append({
                        'from': (move['from'], col),
                        'to': (move['to'], col),
                        'value': move['value']
                    })
                
                # Adjust merge indices to reflect column orientation
                for merge in col_merges:
                    adjusted_merge = {
                        'position': (merge['position'], col),
                        'value': merge['value'],
                        'merged_from': [(pos, col) for pos in merge['merged_from']]
                    }
                    all_merges.append(adjusted_merge)
                
        elif direction == 1:  # Right
            for row in range(self.size):
                # Extract row and reverse it
                r = new_grid[row, :].copy()[::-1]
                # Process as if sliding left
                new_r, row_score, row_movements, row_merges = self._slide_row(r)
                # Update grid with new row (reversed back)
                new_grid[row, :] = new_r[::-1]
                score += row_score
                
                # Adjust movement indices to reflect row orientation and reversal
                for move in row_movements:
                    from_col = self.size - 1 - move['from']
                    to_col = self.size - 1 - move['to']
                    all_movements.append({
                        'from': (row, from_col),
                        'to': (row, to_col),
                        'value': move['value']
                    })
                
                # Adjust merge indices to reflect row orientation and reversal
                for merge in row_merges:
                    position_col = self.size - 1 - merge['position']
                    merged_from = [(row, self.size - 1 - pos) for pos in merge['merged_from']]
                    adjusted_merge = {
                        'position': (row, position_col),
                        'value': merge['value'],
                        'merged_from': merged_from
                    }
                    all_merges.append(adjusted_merge)
                
        elif direction == 2:  # Down
            for col in range(self.size):
                # Extract column and reverse it
                column = new_grid[:, col].copy()[::-1]
                # Process as if sliding left
                new_column, col_score, col_movements, col_merges = self._slide_row(column)
                # Update grid with new column (reversed back)
                new_grid[:, col] = new_column[::-1]
                score += col_score
                
                # Adjust movement indices to reflect column orientation and reversal
                for move in col_movements:
                    from_row = self.size - 1 - move['from']
                    to_row = self.size - 1 - move['to']
                    all_movements.append({
                        'from': (from_row, col),
                        'to': (to_row, col),
                        'value': move['value']
                    })
                
                # Adjust merge indices to reflect column orientation and reversal
                for merge in col_merges:
                    position_row = self.size - 1 - merge['position']
                    merged_from = [(self.size - 1 - pos, col) for pos in merge['merged_from']]
                    adjusted_merge = {
                        'position': (position_row, col),
                        'value': merge['value'],
                        'merged_from': merged_from
                    }
                    all_merges.append(adjusted_merge)
                
        elif direction == 3:  # Left
            for row in range(self.size):
                # Extract row
                r = new_grid[row, :].copy()
                # Process as if sliding left (natural direction)
                new_r, row_score, row_movements, row_merges = self._slide_row(r)
                # Update grid with new row
                new_grid[row, :] = new_r
                score += row_score
                
                # Adjust movement indices to reflect row orientation
                for move in row_movements:
                    all_movements.append({
                        'from': (row, move['from']),
                        'to': (row, move['to']),
                        'value': move['value']
                    })
                
                # Adjust merge indices to reflect row orientation
                for merge in row_merges:
                    adjusted_merge = {
                        'position': (row, merge['position']),
                        'value': merge['value'],
                        'merged_from': [(row, pos) for pos in merge['merged_from']]
                    }
                    all_merges.append(adjusted_merge)
        
        # Check if the grid changed
        moved = not np.array_equal(grid, new_grid)
        
        return moved, score, new_grid, all_movements, all_merges

    def move(self, direction, return_info=False):
        """
        Apply a move and add a random tile if the board changes.
        
        Args:
            direction: Direction to move (0:up, 1:right, 2:down, 3:left)
            return_info: Whether to return detailed move information
            
        Returns:
            If return_info is False: bool indicating if the move changed the board
            If return_info is True: dict with detailed move information
        """
        # Apply move logic
        moved, score, new_grid, movements, merges = self._apply_move(self.grid, direction)
        
        if moved:
            # Update the grid
            self.grid = new_grid
            
            # Add a random tile
            new_tile = self.add_random_tile()
            
            # Store last move info
            self.last_move_info = {
                'moved': moved,
                'score': score,
                'movements': movements,
                'merges': merges,
                'new_tile': new_tile
            }
        else:
            # No movement occurred
            self.last_move_info = {
                'moved': False,
                'score': 0,
                'movements': [],
                'merges': [],
                'new_tile': None
            }
        
        return self.last_move_info if return_info else moved

    def is_valid_move(self, direction):
        """
        Check if a move is valid (would change the board).
        
        Args:
            direction: Direction to check (0:up, 1:right, 2:down, 3:left)
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        # Use _apply_move to check if the move would change the grid
        moved, _, _, _, _ = self._apply_move(self.grid, direction)
        return moved

    def get_available_moves(self):
        """Return a list of available moves (those that change the board)."""
        return [direction for direction in range(4) if self.is_valid_move(direction)]
    
    def is_game_over(self):
        """Check if the game is over (no valid moves)."""
        return len(self.get_available_moves()) == 0
    
    def get_max_tile(self):
        """Return the value of the highest tile on the board."""
        return np.max(self.grid)
    
    def get_tile(self, row, col):
        """Get the value of a specific tile."""
        return self.grid[row, col]

    def copy(self):
        """Create a deep copy of the board."""
        new_board = Board(self.size)
        new_board.grid = self.grid.copy()
        return new_board
    
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
