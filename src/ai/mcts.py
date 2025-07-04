import time
import random
import math
from .base import Agent
from .heuristics.base import CompositeHeuristic
from .heuristics.core import PositionalEmptyTilesHeuristic, MonotonicityHeuristic, SmoothnessHeuristic, CornerMaxHeuristic, MergePotentialHeuristic, MaxValueHeuristic, QualityStabilityHeuristic
from .expectimax import ExpectimaxAgent
from .rl import RLAgent

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.total_score = 0
        self.is_terminal = game_state.is_game_over()
        
    def add_child(self, move, child_node):
        """Add a child node for the given move."""
        self.children[move] = child_node
        
    def is_fully_expanded(self):
        """Check if all possible moves have been explored."""
        if self.is_terminal:
            return True
        available_moves = self.game_state.get_available_moves()
        return len(self.children) == len(available_moves)
    
    def best_child(self, exploration_weight=1.414):
        """Explore tree using standard UCB1 formula."""
        if not self.children:
            return None
            
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                return child  # Prioritize unvisited children
                
            # Standard UCB1 formula
            exploitation = child.total_score / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                
        return best_child


class MctsAgent(Agent):
    def __init__(self, name="MCTS", thinking_time=0.5, exploration_weight=1.414,
                 rollout_type="expectimax", rl_model_path=None):
        super().__init__(name)
        self.thinking_time = thinking_time
        self.exploration_weight = exploration_weight
        self.rollout_type = rollout_type  # "random", "expectimax", or "rl"
        self.rl_model_path = rl_model_path
        self.rollout_agent = None
        
        # Composite heuristic for evaluation
        self.heuristic = CompositeHeuristic()
        self.heuristic.add_heuristic(CornerMaxHeuristic(), 5.0)
        self.heuristic.add_heuristic(MonotonicityHeuristic(), 3.0)
        self.heuristic.add_heuristic(PositionalEmptyTilesHeuristic(), 2.5)
        self.heuristic.add_heuristic(MergePotentialHeuristic(), 2.5)
        self.heuristic.add_heuristic(QualityStabilityHeuristic(), 2.0)
        self.heuristic.add_heuristic(SmoothnessHeuristic(), 1.5)
        self.heuristic.add_heuristic(MaxValueHeuristic(), 1.0)

        if self.rollout_type == "expectimax":
            self.expectimax_agent = ExpectimaxAgent(thinking_time=0.1)  # Fast rollouts
        else:
            self.expectimax_agent = None

        if self.rollout_type == "rl":
            if not self.rl_model_path:
                raise ValueError("rl_model_path must be provided for RL rollout.")
            self.rollout_agent = RLAgent.load_model(self.rl_model_path, training=False, name="RL_Rollout")

        self.stats = {
            "efficiency": 0.0,          # Score per move (total score / moves made)
            "game_duration": 0,         # Total number of moves made
            "max_depth_reached": 0,     # Maximum tree depth explored
            "avg_search_iterations": 0.0 # Average MCTS iterations per move
        }
        
        # Tracking variables for cumulative stats
        self.total_moves_made = 0
        self.total_iterations = 0
    
    def get_move(self, game):
        """Returns the best move using MCTS."""
        
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
        
        root = MCTSNode(game.copy())
        
        start_time = time.time()
        iterations = 0
        total_reward = 0
        while time.time() - start_time < self.thinking_time:
            node = self._select(root)
            
            if not node.is_terminal and node.visits > 0:
                node = self._expand(node)

            reward = self._simulate(node)

            self._backpropagate(node, reward)
            
            iterations += 1
            total_reward += reward
        
        chosen_move = self._best_move(root)
        
        self.total_iterations += iterations
        self.total_moves_made += 1
        self.stats["game_duration"] = self.total_moves_made
        self.stats["efficiency"] = game.score / self.total_moves_made if self.total_moves_made > 0 else 0.0
        self.stats["avg_search_iterations"] = self.total_iterations / self.total_moves_made if self.total_moves_made > 0 else 0.0
        
        return chosen_move
    
    def _select(self, node):
        """Selection phase: traverse down using standard UCB1."""
        depth = 0
        while not node.is_terminal and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)  # Standard UCB1
            if node is None:
                break
            depth += 1
        
        self.stats["max_depth_reached"] = max(self.stats["max_depth_reached"], depth)
        return node
    
    def _expand(self, node):
        """Expansion phase: smart expansion using quality loss analysis."""
        available_moves = node.game_state.get_available_moves()
        unexplored_moves = [move for move in available_moves if move not in node.children]
        if not unexplored_moves:
            return node
        
        original_quality = self.heuristic.evaluate(node.game_state.board.grid)
        best_move = unexplored_moves[0]
        min_predicted_loss = float('inf')
        for move in unexplored_moves:
            test_game = node.game_state.copy()
            test_game.step(move)
            
            new_quality = self.heuristic.evaluate(test_game.board.grid)
            predicted_loss = max(0, original_quality - new_quality)
            
            # Pick move with least predicted quality loss
            if predicted_loss < min_predicted_loss:
                min_predicted_loss = predicted_loss
                best_move = move
        
        new_game = node.game_state.copy()
        new_game.step(best_move)
        
        child = MCTSNode(new_game, parent=node, move=best_move)
        node.add_child(best_move, child)
        
        return child
    
    def _simulate(self, node):
        """Simulation phase: simulate the game using the rollout type."""
        sim_game = node.game_state.copy()
        if self.rollout_type == "expectimax":
            final_score = self.expectimax_agent._expectimax(
                sim_game,
                3,  # Shallow depth for performance
                True,  # is_max_player
                time.time()  # start_time
            )
        elif self.rollout_type == "random":
            steps = 0
            max_steps = 50  # Prevent infinite games
            while not sim_game.is_game_over() and steps < max_steps:
                moves = sim_game.get_available_moves()
                if not moves:
                    break
                
                move = random.choice(moves)
                sim_game.step(move)
                steps += 1
            final_score = self.heuristic.evaluate(sim_game.board.grid)        
        elif self.rollout_type == "rl" and self.rollout_agent is not None:
            steps = 0
            max_steps = 50  # Prevent infinite games
            while not sim_game.is_game_over() and steps < max_steps:
                move = self.rollout_agent.get_move(sim_game)
                sim_game.step(move)
                steps += 1
            final_score = self.heuristic.evaluate(sim_game.board.grid)
        else:
            raise ValueError(f"Invalid rollout_type: {self.rollout_type}. Must be 'random', 'expectimax', or 'rl' with a valid rollout_agent.")
        
        return final_score
    
    def _backpropagate(self, node, reward):
        """Backpropagation phase: update visit counts and rewards."""
        current_node = node
        
        while current_node is not None:
            current_node.visits += 1
            current_node.total_score += reward
            current_node = current_node.parent
    
    def _best_move(self, root):
        """Select most visited child (standard MCTS approach)."""
        if not root.children:
            return random.choice(root.game_state.get_available_moves())
        
        best_move = None
        max_visits = -1
        
        for move, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_move = move
        
        return best_move if best_move is not None else random.choice(list(root.children.keys()))
    
    def get_stats(self):
        return self.stats.copy()
    
    def get_config(self):
        config = {
            'thinking_time': self.thinking_time,
            'exploration_weight': self.exploration_weight,
            'rollout_type': self.rollout_type
        }
        if self.rollout_type == "rl":
            config['rl_model_path'] = self.rl_model_path
        return config