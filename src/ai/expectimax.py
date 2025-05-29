import time
from .base import Agent
from .heuristics.base import CompositeHeuristic
from .heuristics.core import EmptyTilesHeuristic, MonotonicityHeuristic, SmoothnessHeuristic, CornerMaxHeuristic, MergePotentialHeuristic, MaxValueHeuristic

class ExpectimaxAgent(Agent):
    def __init__(self, name="Expectimax", thinking_time=0.5):
        super().__init__(name)
        self.thinking_time = thinking_time
        
        # Create composite heuristic for evaluation
        self.heuristic = CompositeHeuristic()
        self.heuristic.add_heuristic(EmptyTilesHeuristic(), 3.0)
        self.heuristic.add_heuristic(MonotonicityHeuristic(), 1.5) 
        self.heuristic.add_heuristic(CornerMaxHeuristic(), 5.0)
        self.heuristic.add_heuristic(SmoothnessHeuristic(), 2.0)
        self.heuristic.add_heuristic(MergePotentialHeuristic(), 2.5)
        self.heuristic.add_heuristic(MaxValueHeuristic(), 3.0)

        self.stats = {
            "search_iterations": 0,     # Total move evaluations performed
            "max_depth_reached": 0,     # Maximum search depth reached
            "avg_reward": 0.0           # Average heuristic value of evaluated positions
        }
    
    def get_move(self, game):
        """Returns the best move using Expectimax with progressive deepening."""
        
        # Reset stats each move
        self.stats = {
            "search_iterations": 0,
            "max_depth_reached": 0,
            "avg_reward": 0.0
        }
        
        available_moves = game.get_available_moves()
        if not available_moves:
            return -1
        if len(available_moves) == 1:
            return available_moves[0]
        
        total_evaluations = 0
        total_reward = 0.0
        start_time = time.time()
        best_move = None
        depth = 1
        while time.time() - start_time < self.thinking_time:
            current_best_move = None
            current_best_score = float('-inf')
            depth_completed = True
            
            for move in available_moves:
                # Check if we still have time for this move
                if time.time() - start_time >= self.thinking_time:
                    depth_completed = False
                    break
                
                next_state = game.copy()
                next_state.step(move)
                
                score = self._expectimax(next_state, depth - 1, False, start_time)
                
                total_evaluations += 1
                total_reward += score
                self.stats["search_iterations"] += 1
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
            
            if not depth_completed:
                break
                
            if depth_completed and current_best_move is not None:
                best_move = current_best_move
            
            depth += 1
            
        if total_evaluations > 0:
            self.stats["avg_reward"] = total_reward / total_evaluations
            
        return best_move if best_move is not None else available_moves[0]

    def _expectimax(self, game, depth, is_max_player, start_time):
        """Complete Expectimax search with time checks."""
        
        # Time check before any evaluation
        if time.time() - start_time >= self.thinking_time:
            return self.heuristic.evaluate(game.board.grid)
            
        # Base cases
        if depth == 0 or game.is_game_over():
            return self.heuristic.evaluate(game.board.grid)
        
        self.stats["max_depth_reached"] = max(self.stats["max_depth_reached"], depth)

        if is_max_player:
            # Max node: try all valid moves
            best_value = float('-inf')
            moves = game.get_available_moves()
            
            for move in moves:
                # Time check before processing each move
                if time.time() - start_time >= self.thinking_time:
                    break
                    
                next_state = game.copy()
                next_state.step(move)
                value = self._expectimax(next_state, depth - 1, False, start_time)
                best_value = max(best_value, value)
            
            return best_value

        else:
            # Chance node: evaluate all empty cells
            empty_cells = []
            for i in range(game.board.size):
                for j in range(game.board.size):
                    if game.board.grid[i,j] == 0:
                        empty_cells.append((i,j))

            if len(empty_cells) == 0:
                return self.heuristic.evaluate(game.board.grid)

            expected_value = 0
            for (i, j) in empty_cells:
                # Time check before processing each empty cell
                if time.time() - start_time >= self.thinking_time:
                    break
                    
                for tile_value, probability in [(2, 0.9), (4, 0.1)]:
                    next_game = game.copy()
                    next_game.board.grid[i,j] = tile_value
                    value = self._expectimax(next_game, depth - 1, True, start_time)
                    expected_value += value * probability
                    
            # Average across all empty positions
            expected_value /= len(empty_cells)
            return expected_value

    def get_stats(self):
        return self.stats.copy()
    
    def get_config(self):
        return {
            'thinking_time': self.thinking_time,
        } 