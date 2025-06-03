import numpy as np
from .base import Agent

class GreedyAgent(Agent):
    def __init__(self, name="Greedy", tile_weight=1.0, score_weight=0.1):
        super().__init__(name)
        self.tile_weight = tile_weight
        self.score_weight = score_weight
        
        self.stats = {
            "efficiency": 0.0,          # Score per move (total score / moves made)
            "game_duration": 0,         # Total number of moves made
            "avg_move_score": 0.0,      # Average evaluation score of all moves
            "score_variance": 0.0       # How much move scores varied (decision difficulty)
        }
        
        # Tracking variables for cumulative stats
        self.total_moves_made = 0
        
    def get_move(self, game):
        # Reset move-specific stats
        self.stats["avg_move_score"] = 0.0
        self.stats["score_variance"] = 0.0
        
        valid_moves = game.board.get_available_moves()
        if not valid_moves:
            return -1
        
        best_move = None
        best_score = float('-inf')
        total_score = 0.0
        move_scores = []
        
        for move in valid_moves:
            game_copy = game.copy()
            move_result = game_copy.step(move)
            
            score_gain = move_result.get('score', 0)
            new_max_tile = game_copy.board.get_max_tile()
            
            tile_component = np.log2(max(new_max_tile, 1)) * self.tile_weight
            score_component = score_gain * self.score_weight
            combined_score = tile_component + score_component
            
            move_scores.append(combined_score)
            total_score += combined_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_move = move
        
        # Update tracking stats
        self.total_moves_made += 1
        self.stats["game_duration"] = self.total_moves_made
        self.stats["efficiency"] = game.score / self.total_moves_made if self.total_moves_made > 0 else 0.0
        
        if valid_moves:
            self.stats["avg_move_score"] = total_score / len(valid_moves)
            
            if len(move_scores) > 1:
                mean_score = self.stats["avg_move_score"]
                variance = sum((score - mean_score) ** 2 for score in move_scores) / len(move_scores)
                self.stats["score_variance"] = variance
        
        return best_move if best_move is not None else valid_moves[0]
    
    def get_stats(self):
        return self.stats.copy()
    
    def get_config(self):
        return {
            'tile_weight': self.tile_weight,
            'score_weight': self.score_weight
        } 