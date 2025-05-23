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
        
        # Stats tracking
        self.moves_made = 0
        self.valid_moves_history = []
    
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
        
        # Track stats
        self.valid_moves_history.append(len(valid_moves))
        self.moves_made += 1
        
        # If no valid moves, return -1
        if not valid_moves:
            return -1
        
        # Choose a random valid move
        selected_move = random.choice(valid_moves)
        
        return selected_move
    
    def get_stats(self):
        """Return agent statistics."""
        # Calculate average number of valid moves available
        avg_valid_moves = 0
        if self.valid_moves_history:
            avg_valid_moves = sum(self.valid_moves_history) / len(self.valid_moves_history)
        
        return {
            'moves_made': self.moves_made,
            'avg_valid_moves': round(avg_valid_moves, 2),
            'strategy': 'Pure random selection'
        }
    
    def get_config(self):
        """Return agent configuration."""
        return {
            'seed': self.seed
        }
