import random
from .base import Agent

class RandomAgent(Agent):
    def __init__(self, name="Random", seed=None):
        super().__init__(name)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        self.stats = {
            "search_iterations": 0,     # Move evaluations (always 1 for random)
            "max_depth_reached": 0,     # No depth for random (0)
            "avg_reward": 0.0           # No reward calculation for random
        }
        
    def get_move(self, game):
        """Select a random valid move from the current game state."""
        
        # Reset stats
        self.stats = {
            "search_iterations": 1,     # One "evaluation" - picking randomly
            "max_depth_reached": 0,     # No depth for random
            "avg_reward": 0.0           # No meaningful reward for random
        }
        
        valid_moves = game.board.get_available_moves()
        if not valid_moves:
            return -1
        
        return random.choice(valid_moves)
    
    def get_stats(self):
        return self.stats.copy()
    
    def get_config(self):
        """Return agent configuration."""
        return {
            'seed': self.seed
        }