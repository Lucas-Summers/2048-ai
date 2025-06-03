import numpy as np
import time
import random
from .base import Agent
from .heuristics.base import CompositeHeuristic
from .heuristics.core import EmptyTilesHeuristic, MonotonicityHeuristic, SmoothnessHeuristic, CornerMaxHeuristic, MergePotentialHeuristic, MaxValueHeuristic

class ExpectimaxAgent(Agent):
    def __init__(self, name="Expectimax", thinking_time=0.5, max_chance_samples=8, 
                 min_prob_threshold=0.01, max_moves=4):
        super().__init__(name)
        self.thinking_time = thinking_time
        self.max_chance_samples = max_chance_samples
        self.min_prob_threshold = min_prob_threshold
        self.max_moves = max_moves
        
        # Create composite heuristic for evaluation
        self.heuristic = CompositeHeuristic()
        self.heuristic.add_heuristic(EmptyTilesHeuristic(), 3.0)
        self.heuristic.add_heuristic(MonotonicityHeuristic(), 1.5) 
        self.heuristic.add_heuristic(CornerMaxHeuristic(), 5.0)
        self.heuristic.add_heuristic(SmoothnessHeuristic(), 2.0)
        self.heuristic.add_heuristic(MergePotentialHeuristic(), 2.5)
        self.heuristic.add_heuristic(MaxValueHeuristic(), 3.0)

        self.stats = {
            "nodes_evaluated": 0,     # Total number of game states evaluated
            "max_depth_reached": 0,   # Maximum search depth reached
            "chance_nodes_pruned": 0, # Number of chance nodes pruned via sampling
            "cache_hits": 0,          # Number of times evaluation cache was used
            "moves_pruned": 0,        # Number of moves pruned via move ordering
            "depths_completed": 0     # Number of complete depth iterations finished
        }
        
        # Cache for board evaluations
        self.eval_cache = {}
    
    def get_move(self, game):
        """Returns the best move for the given game state using Expectimax and progressive deepening."""
        
        # Reset stats with each move
        self.stats = {
            "nodes_evaluated": 0,
            "max_depth_reached": 0,
            "chance_nodes_pruned": 0,
            "cache_hits": 0,
            "moves_pruned": 0,
            "depths_completed": 0
        }
        
        # Clear cache periodically to prevent memory buildup
        if len(self.eval_cache) > 10000:
            self.eval_cache.clear()
        
        start_time = time.time()
        best_move = None
        available_moves = game.get_available_moves()
        
        if not available_moves:
            return -1
        if len(available_moves) == 1:
            return available_moves[0]
        
        # Progressive deepening: keep searching deeper until time runs out
        depth = 1
        while time.time() - start_time < self.thinking_time:
            current_best_move = None
            current_best_score = float('-inf')
            depth_completed = True
            
            # Order and prune moves for this depth iteration
            pruned_moves = self.get_pruned_moves(game, available_moves)
            for move in pruned_moves:
                # Check if we still have time for this move
                if time.time() - start_time >= self.thinking_time:
                    depth_completed = False
                    break
                
                next_state = game.copy()
                next_state.step(move)
                
                # Search with remaining time budget
                remaining_time = self.thinking_time - (time.time() - start_time)
                score = self._expectimax_fixed_depth(next_state, depth - 1, True, 
                                                   start_time, remaining_time)
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
            
            if not depth_completed:
                break
                
            if depth_completed and current_best_move is not None:
                best_move = current_best_move
                self.stats["depths_completed"] = depth
            depth += 1
            
        return best_move if best_move is not None else available_moves[0]

    def _expectimax_fixed_depth(self, game, depth, is_max_player, start_time, time_limit):
        """Fixed-depth Expectimax search with time checks."""

        if time.time() - start_time > time_limit:
            return self.heuristic.evaluate(game.board.grid)
        if depth == 0:
            return self.heuristic.evaluate(game.board.grid)
        if game.is_game_over():
            return self.heuristic.evaluate(game.board.grid)
        
        # check cache for an already evaluated board state at this depth
        cache_key = (hash(game.board), depth)
        if cache_key in self.eval_cache:
            self.stats["cache_hits"] += 1
            return self.eval_cache[cache_key]
        
        self.stats["nodes_evaluated"] += 1
        self.stats["max_depth_reached"] = max(self.stats["max_depth_reached"], depth)

        if is_max_player:
            best_value = float('-inf')
            moves = game.get_available_moves()
            
            for move in moves:
                # Check time limit before processing
                if time.time() - start_time > time_limit:
                    break
                    
                next_state = game.copy()
                next_state.step(move)
                value = self._expectimax_fixed_depth(next_state, depth - 1, False, start_time, time_limit)
                best_value = max(best_value, value)
            
            # Cache the result
            self.eval_cache[cache_key] = best_value
            return best_value

        else:  # Chance node
            empty_cells = []
            for i in range(game.board.size):
                for j in range(game.board.size):
                    if game.board.grid[i,j] == 0:
                        empty_cells.append((i,j))

            if len(empty_cells) == 0:
                result = self.heuristic.evaluate(game.board.grid)
                self.eval_cache[cache_key] = result
                return result

            # Sparse sampling: limit number of empty cells to sample
            og_count = len(empty_cells)
            if len(empty_cells) > self.max_chance_samples:
                empty_cells = self._select_best_empty_cells(game.board.grid, empty_cells, self.max_chance_samples)
                self.stats["chance_nodes_pruned"] += og_count - len(empty_cells)

            expected_value = 0
            total_probability = 0
            
            for (i, j) in empty_cells:
                # Check time limit again
                if time.time() - start_time > time_limit:
                    break
                    
                for tile_value, tile_prob in [(2, 0.9), (4, 0.1)]:
                    # skip processingextremely unlikely outcomes
                    cell_probability = tile_prob / len(empty_cells)
                    if cell_probability < self.min_prob_threshold:
                        continue
                        
                    next_game = game.copy()
                    next_game.board.grid[i,j] = tile_value
                    value = self._expectimax_fixed_depth(next_game, depth - 1, True, start_time, time_limit)
                    expected_value += value * cell_probability
                    total_probability += cell_probability

            # Normalize by total probability
            if total_probability > 0:
                expected_value /= total_probability
            else:
                expected_value = self.heuristic.evaluate(game.board.grid)
                
            self.eval_cache[cache_key] = expected_value
            return expected_value

    def _select_best_empty_cells(self, board, empty_cells, max_samples):
        """Select the most promising empty cells for sparse sampling based on position heuristics."""
        if len(empty_cells) <= max_samples:
            return empty_cells
            
        cell_scores = []
        for i, j in empty_cells:
            score = 0
            
            # Prefer corners and edges
            if (i, j) in [(0,0), (0,3), (3,0), (3,3)]:
                score += 10
            elif i == 0 or i == 3 or j == 0 or j == 3:
                score += 5
                
            # Prefer positions near high-value tiles
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 4 and 0 <= nj < 4 and board[ni][nj] > 0:
                        score += np.log2(board[ni][nj]) * 0.1
                        
            # Add some randomness to avoid deterministic patterns
            score += random.random() * 0.1
            
            cell_scores.append(((i, j), score))
        
        cell_scores.sort(key=lambda x: x[1], reverse=True)
        return [cell for cell, _ in cell_scores[:max_samples]]

    def get_pruned_moves(self, game, available_moves):
        """Order moves by quick heuristic evaluation and prune to top max_moves."""
        if len(available_moves) <= self.max_moves:
            return available_moves
        
        move_scores = []
        for move in available_moves:
            next_state = game.copy()
            next_state.step(move)
            
            # Check cache first
            board_hash = hash(next_state.board)
            cache_key = (board_hash, 0) # Use depth 0?
            if cache_key in self.eval_cache:
                heuristic_value = self.eval_cache[cache_key]
                self.stats["cache_hits"] += 1
            else:
                heuristic_value = self.heuristic.evaluate(next_state.board.grid)
                self.eval_cache[cache_key] = heuristic_value
            
            # Combine heuristic with immediate score gain
            score_gain = next_state.score - game.score
            total_score = heuristic_value + score_gain * 0.1  # Weight score less to avoid greediness
            move_scores.append((move, total_score))
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        pruned_moves = [move for move, _ in move_scores[:self.max_moves]]
        self.stats["moves_pruned"] += len(available_moves) - len(pruned_moves)
        
        return pruned_moves

    def get_stats(self):
        return self.stats.copy()
    
    def get_config(self):
        return {
            'thinking_time': self.thinking_time,
            'max_chance_samples': self.max_chance_samples,
            'min_prob_threshold': self.min_prob_threshold,
            'max_moves': self.max_moves
        }
