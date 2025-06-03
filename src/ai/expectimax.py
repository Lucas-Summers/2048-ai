import time
from .base import Agent
from .heuristics.base import CompositeHeuristic
from .heuristics.core import PositionalEmptyTilesHeuristic, MonotonicityHeuristic, SmoothnessHeuristic, CornerMaxHeuristic, MergePotentialHeuristic, MaxValueHeuristic, QualityStabilityHeuristic

class ExpectimaxAgent(Agent):
    def __init__(self, name="Expectimax", thinking_time=0.5):
        super().__init__(name)
        self.thinking_time = thinking_time
        
        # Composite heuristic for evaluation
        self.heuristic = CompositeHeuristic()
        self.heuristic.add_heuristic(CornerMaxHeuristic(), 5.0)
        self.heuristic.add_heuristic(MonotonicityHeuristic(), 3.0)
        self.heuristic.add_heuristic(PositionalEmptyTilesHeuristic(), 2.5)
        self.heuristic.add_heuristic(MergePotentialHeuristic(), 2.5)
        self.heuristic.add_heuristic(QualityStabilityHeuristic(), 2.0)
        self.heuristic.add_heuristic(SmoothnessHeuristic(), 1.5)
        self.heuristic.add_heuristic(MaxValueHeuristic(), 1.0)
        
        self.stats = {
            "efficiency": 0.0,          # Score per move (total score / moves made)
            "game_duration": 0,         # Total number of moves made
            "max_depth_reached": 0,     # Maximum tree depth explored
            "avg_search_iterations": 0.0 # Average search iterations per move
        }
        
        # Tracking variables for cumulative stats
        self.total_moves_made = 0
        self.total_iterations = 0

    def get_move(self, game):
        """Returns the best move using Expectimax with quality loss minimization."""
        
        # Reset move-specific stats
        self.stats["max_depth_reached"] = 0
        
        available_moves = game.get_available_moves()
        if not available_moves:
            return -1
        if len(available_moves) == 1:
            # Still count this as a move
            self.total_moves_made += 1
            self.stats["game_duration"] = self.total_moves_made
            self.stats["efficiency"] = game.score / self.total_moves_made if self.total_moves_made > 0 else 0.0
            self.stats["avg_search_iterations"] = self.total_iterations / self.total_moves_made if self.total_moves_made > 0 else 0.0
            return available_moves[0]
        
        original_quality = self.heuristic.evaluate(game.board.grid)
        start_time = time.time()
        
        best_move = None
        best_quality_loss = float('inf')
        best_expected_quality = float('-inf')
        depth = 1
        total_evaluations = 0
        total_reward = 0.0
        
        while time.time() - start_time < self.thinking_time:
            depth_completed = True
            
            for move in available_moves:
                if time.time() - start_time >= self.thinking_time:
                    depth_completed = False
                    break
                    
                next_state = game.copy()
                next_state.step(move)
                
                result = self._analyze_move_quality(next_state, depth-1, original_quality, start_time)
                
                total_evaluations += 1
                total_reward += result['expected_quality']
                
                # minimize the quality loss first, then maximize the expected quality
                if (result['quality_loss'] < best_quality_loss or 
                    (result['quality_loss'] == best_quality_loss and 
                     result['expected_quality'] > best_expected_quality)):
                    
                    best_quality_loss = result['quality_loss']
                    best_expected_quality = result['expected_quality']
                    best_move = move
            
            if not depth_completed:
                break
                
            depth += 1
        
        chosen_move = best_move if best_move is not None else available_moves[0]
        
        self.total_iterations += total_evaluations
        self.total_moves_made += 1
        self.stats["game_duration"] = self.total_moves_made
        self.stats["efficiency"] = game.score / self.total_moves_made if self.total_moves_made > 0 else 0.0
        self.stats["avg_search_iterations"] = self.total_iterations / self.total_moves_made if self.total_moves_made > 0 else 0.0
        
        return chosen_move

    def _get_strategic_empty_cells(self, board):
        """Only test tile spawns adjacent to existing tiles."""
        strategic_cells = []
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        
        for i in range(4):
            for j in range(4):
                if board[i,j] == 0:  # Empty cell
                    # Check if adjacent to any tile
                    has_adjacent_tile = False
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 4 and 0 <= nj < 4:
                            if board[ni,nj] != 0:
                                has_adjacent_tile = True
                                break
                    
                    if has_adjacent_tile:
                        strategic_cells.append((i, j))
        
        # if no strategic cells, use all empty cells
        if not strategic_cells:
            for i in range(4):
                for j in range(4):
                    if board[i,j] == 0:
                        strategic_cells.append((i, j))
        
        return strategic_cells

    def _analyze_move_quality(self, game, depth, original_quality, start_time):
        """Move quality analysis with worst-case tracking."""
        if time.time() - start_time >= self.thinking_time or depth == 0:
            current_quality = self.heuristic.evaluate(game.board.grid)
            return {
                'expected_quality': current_quality,
                'quality_loss': max(0, original_quality - current_quality)
            }
        
        empty_cells = self._get_strategic_empty_cells(game.board.grid)
        if not empty_cells:
            current_quality = self.heuristic.evaluate(game.board.grid)
            return {
                'expected_quality': current_quality,
                'quality_loss': max(0, original_quality - current_quality)
            }
        
        total_expected_quality = 0
        worst_quality = float('inf')
        for (i, j) in empty_cells:
            for tile_value, probability in [(2, 0.9), (4, 0.1)]:
                next_game = game.copy()
                next_game.board.grid[i,j] = tile_value
                
                sub_quality = self._expectimax(next_game, depth-1, True, start_time)
                total_expected_quality += sub_quality * probability
                worst_quality = min(worst_quality, sub_quality)
        
        expected_quality = total_expected_quality / len(empty_cells)
        # Use worst case for quality loss calculation
        quality_loss = max(0, original_quality - worst_quality)
        
        return {
            'expected_quality': expected_quality,
            'quality_loss': quality_loss
        }

    def _expectimax(self, game, depth, is_max_player, start_time):
        """Standard expectimax with pruning optimizations."""
        if time.time() - start_time >= self.thinking_time:
            return self.heuristic.evaluate(game.board.grid)
            
        if depth == 0 or game.is_game_over():
            return self.heuristic.evaluate(game.board.grid)

        self.stats["max_depth_reached"] = max(self.stats["max_depth_reached"], depth)

        if is_max_player:
            best_value = float('-inf')
            for move in game.get_available_moves():
                next_state = game.copy()
                next_state.step(move)
                value = self._expectimax(next_state, depth-1, False, start_time)
                best_value = max(best_value, value)
            return best_value
        else:
            # Use strategic empty cells instead of all empty cells
            empty_cells = self._get_strategic_empty_cells(game.board.grid)
            if not empty_cells:
                return self.heuristic.evaluate(game.board.grid)

            expected_value = 0
            for (i, j) in empty_cells:
                for tile_value, probability in [(2, 0.9), (4, 0.1)]:
                    next_game = game.copy()
                    next_game.board.grid[i,j] = tile_value
                    value = self._expectimax(next_game, depth-1, True, start_time)
                    expected_value += value * probability
                    
            return expected_value / len(empty_cells)
    
    def get_stats(self):
        return self.stats.copy()
    
    def get_config(self):
        return {
            'thinking_time': self.thinking_time
        }