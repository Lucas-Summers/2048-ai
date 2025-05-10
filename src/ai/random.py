import random
from .base import Agent

class RandomAgent(Agent):
    """
    Agent that selects random valid moves for the 2048 game.
    Serves as a baseline for comparing more sophisticated algorithms.
    """
    
    def __init__(self, name="Random", seed=None):
        super().__init__(name)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def get_move(self, game):
        """
        Select a random valid move from the current game state.
        
        Args:
            game: Game2048 instance
            
        Returns:
            move: Integer representing direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
                 or -1 if no valid moves are available
        """
        # Get valid moves from the game's board
        valid_moves = game.board.get_available_moves()
        
        # If no valid moves, return -1
        if not valid_moves:
            return -1
        
        # Choose a random valid move
        selected_move = random.choice(valid_moves)
        
        return selected_move
    
    def get_config(self):
        """Return agent configuration."""
        return {
            'seed': self.seed
        }
