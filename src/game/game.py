import numpy as np
from .board import Board

class Game2048:
    def __init__(self):
        self.board = Board()
        self.score = 0
        self.moves = 0
        self.game_over = False
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = Board()
        self.score = 0
        self.moves = 0
        self.game_over = False
    
    def step(self, direction):
        """Make a move in the specified direction (0:up, 1:right, 2:down, 3:left)."""
        move_info = self.board.move(direction, return_info=True)
        if not move_info['moved']:
            return move_info
        
        self.score += move_info['score']
        self.moves += 1
        self.game_over = self.board.is_game_over()
        
        # Add extra game info to the result
        move_info.update({
            'total_score': self.score,
            'moves': self.moves,
            'game_over': self.game_over
        })
        
        return move_info
    
    def is_game_over(self):
        """Check if the game is over (no valid moves)."""
        return self.game_over
    
    def get_available_moves(self):
        """Get a list of valid moves in the current state."""
        return self.board.get_available_moves()
    
    def is_valid_move(self, direction):
        """Check if a move is valid without making it."""
        return self.board.is_valid_move(direction)
    
    def copy(self):
        """Create a deep copy of the game state."""
        game_copy = Game2048()
        game_copy.board = self.board.copy()
        game_copy.score = self.score
        game_copy.moves = self.moves
        game_copy.game_over = self.game_over
        return game_copy
    
    def get_state(self):
        """Get a serializable representation of the game state."""
        return {
            'board': self.board.grid.tolist(),
            'score': self.score,
            'moves': self.moves,
            'game_over': self.game_over
        }
    
    def set_state(self, state):
        """Restore the game from a serialized state."""
        self.board.grid = np.array(state['board'])
        self.score = state['score']
        self.moves = state['moves']
        self.game_over = state['game_over']
