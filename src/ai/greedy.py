import numpy as np
from .base import Agent

class GreedyAgent(Agent):
    def __init__(self, name="Greedy", tile_weight=1.0, score_weight=0.1):
        super().__init__(name)
        self.tile_weight = tile_weight
        self.score_weight = score_weight
        
        self.stats = {
            "search_iterations": 0,     # Move evaluations performed
            "max_depth_reached": 1,     # Always 1 for greedy (single-step lookahead)
            "avg_reward": 0.0           # Average evaluation score
        }
        
    def get_move(self, game):
        """Select the move that maximizes combination of highest tile and score gain."""
        
        # Reset stats
        self.stats = {
            "search_iterations": 0,
            "max_depth_reached": 1,
            "avg_reward": 0.0
        }
        
        valid_moves = game.board.get_available_moves()
        if not valid_moves:
            return -1
        
        best_move = None
        best_score = float('-inf')
        total_score = 0.0
        
        for move in valid_moves:
            game_copy = game.copy()
            move_result = game_copy.step(move)
            
            score_gain = move_result.get('score', 0)
            new_max_tile = game_copy.board.get_max_tile()
            
            # Combined evaluation
            tile_component = np.log2(max(new_max_tile, 1)) * self.tile_weight
            score_component = score_gain * self.score_weight
            combined_score = tile_component + score_component
            
            total_score += combined_score
            self.stats["search_iterations"] += 1
            
            if combined_score > best_score:
                best_score = combined_score
                best_move = move
        
        # Calculate average reward
        if self.stats["search_iterations"] > 0:
            self.stats["avg_reward"] = total_score / self.stats["search_iterations"]
        
        return best_move if best_move is not None else valid_moves[0]
    
    def get_stats(self):
        return self.stats.copy()
    
    def get_config(self):
        return {
            'tile_weight': self.tile_weight,
            'score_weight': self.score_weight
        } 