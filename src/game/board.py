import numpy as np
import random

class Board:
    def __init__(self, size=4):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.last_move_info = None

        # Add initial two tiles (2 or 4)
        self.add_random_tile()
        self.add_random_tile()
        
    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell."""
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            value = 2 if random.random() < 0.9 else 4
            self.grid[i, j] = value
            return {'position': (i, j), 'value': value}
        return None
            
    def get_empty_cells_count(self):
        """Return the number of empty cells."""
        return np.count_nonzero(self.grid == 0)
    
    def _slide_row(self, row):
        """Slide and merge tiles in a row to the left, tracking movements and merges."""
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
                
                # Skip the next value
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

            write_idx += 1
        
        return new_row, score, movements, merges

    def _apply_move(self, grid, direction):
        """Apply move logic to a grid and return detailed info without modifying the original grid."""
        new_grid = grid.copy()
        score = 0
        all_movements = []
        all_merges = []
        
        # Transform row/column indices based on direction
        if direction == 0:  # Up
            for col in range(self.size):
                column = new_grid[:, col].copy()
                new_column, col_score, col_movements, col_merges = self._slide_row(column)
                new_grid[:, col] = new_column
                score += col_score
                
                # Adjust movement indices to reflect column orientation
                for move in col_movements:
                    all_movements.append({
                        'from': (int(move['from']), int(col)),
                        'to': (int(move['to']), int(col)),
                        'value': int(move['value'])
                    })
                
                # Adjust merge indices to reflect column orientation
                for merge in col_merges:
                    adjusted_merge = {
                        'position': (int(merge['position']), int(col)),
                        'value': int(merge['value']),
                        'merged_from': [(int(pos), int(col)) for pos in merge['merged_from']]
                    }
                    all_merges.append(adjusted_merge)
                
        elif direction == 1:  # Right
            for r in range(self.size):
                row = new_grid[r, :].copy()[::-1]
                new_row, row_score, row_movements, row_merges = self._slide_row(row)
                new_grid[r, :] = new_row[::-1]
                score += row_score
                
                # Adjust movement indices to reflect row orientation
                for move in row_movements:
                    from_col = self.size - 1 - move['from']
                    to_col = self.size - 1 - move['to']
                    all_movements.append({
                        'from': (int(r), int(from_col)),
                        'to': (int(r), int(to_col)),
                        'value': int(move['value'])
                    })
                
                # Adjust merge indices to reflect row orientation
                for merge in row_merges:
                    position_col = self.size - 1 - merge['position']
                    merged_from = [(int(r), int(self.size - 1 - pos)) for pos in merge['merged_from']]
                    adjusted_merge = {
                        'position': (int(r), int(position_col)),
                        'value': int(merge['value']),
                        'merged_from': merged_from
                    }
                    all_merges.append(adjusted_merge)
                
        elif direction == 2:  # Down
            for col in range(self.size):
                column = new_grid[:, col].copy()[::-1]
                new_column, col_score, col_movements, col_merges = self._slide_row(column)
                new_grid[:, col] = new_column[::-1]
                score += col_score
                
                # Adjust movement indices to reflect column orientation
                for move in col_movements:
                    from_row = self.size - 1 - move['from']
                    to_row = self.size - 1 - move['to']
                    all_movements.append({
                        'from': (int(from_row), int(col)),
                        'to': (int(to_row), int(col)),
                        'value': int(move['value'])
                    })
                
                # Adjust merge indices to reflect column orientation
                for merge in col_merges:
                    position_row = self.size - 1 - merge['position']
                    merged_from = [(int(self.size - 1 - pos), int(col)) for pos in merge['merged_from']]
                    adjusted_merge = {
                        'position': (int(position_row), int(col)),
                        'value': int(merge['value']),
                        'merged_from': merged_from
                    }
                    all_merges.append(adjusted_merge)
                
        elif direction == 3:  # Left
            for r in range(self.size):
                row = new_grid[r, :].copy()
                new_row, row_score, row_movements, row_merges = self._slide_row(row)
                new_grid[r, :] = new_row
                score += row_score
                
                # Adjust movement indices to reflect row orientation
                for move in row_movements:
                    all_movements.append({
                        'from': (int(r), int(move['from'])),
                        'to': (int(r), int(move['to'])),
                        'value': int(move['value'])
                    })
                
                # Adjust merge indices to reflect row orientation
                for merge in row_merges:
                    adjusted_merge = {
                        'position': (int(r), int(merge['position'])),
                        'value': int(merge['value']),
                        'merged_from': [(int(r), int(pos)) for pos in merge['merged_from']]
                    }
                    all_merges.append(adjusted_merge)
        
        # Check if the grid changed
        moved = not np.array_equal(grid, new_grid)
        
        return moved, score, new_grid, all_movements, all_merges

    def move(self, direction, return_info=False):
        """Apply a move and add a random tile if the board changes."""
        moved, score, new_grid, movements, merges = self._apply_move(self.grid, direction)
        
        if moved:
            self.grid = new_grid
            new_tile = self.add_random_tile()
            self.last_move_info = {
                'moved': moved,
                'score': score,
                'movements': movements,
                'merges': merges,
                'new_tile': new_tile
            }
        else:
            self.last_move_info = {
                'moved': False,
                'score': 0,
                'movements': [],
                'merges': [],
                'new_tile': None
            }
        
        return self.last_move_info if return_info else moved

    def is_valid_move(self, direction):
        """Check if a move is valid (would change the board)."""
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

    def __hash__(self):
        return hash(tuple(map(tuple, self.grid)))

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return np.array_equal(self.grid, other.grid)