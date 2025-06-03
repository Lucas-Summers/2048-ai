import random
from .base import Agent

class RandomAgent(Agent):
    def __init__(self, name="Random", seed=None):
        super().__init__(name)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        self.stats = {
            "move_distribution": [0, 0, 0, 0],  # Count of [up, down, left, right] moves chosen
            "efficiency": 0.0,                  # Score per move (total score / moves made)
            "merge_frequency": 0.0,             # Percentage of moves that resulted in merges
            "game_duration": 0                  # Total number of moves made (survival metric)
        }
        
        # Tracking variables for cumulative stats
        self.total_moves_made = 0
        self.total_score = 0
        self.total_merges = 0
        
    def get_move(self, game):
        valid_moves = game.board.get_available_moves()
        if not valid_moves:
            return -1
        
        chosen_move = random.choice(valid_moves)
        
        self.total_moves_made += 1
        self.stats["move_distribution"][chosen_move] += 1
        game_copy = game.copy()
        move_result = game_copy.step(chosen_move)
        score_gain = move_result.get('score', 0)
        self.total_score += score_gain
        if score_gain > 0:
            self.total_merges += 1
        self.stats["efficiency"] = self.total_score / self.total_moves_made if self.total_moves_made > 0 else 0.0
        self.stats["merge_frequency"] = self.total_merges / self.total_moves_made if self.total_moves_made > 0 else 0.0
        self.stats["game_duration"] = self.total_moves_made
        
        return chosen_move
    
    def get_stats(self):
        return self.stats.copy()
    
    def get_config(self):
        return {
            'seed': self.seed
        }