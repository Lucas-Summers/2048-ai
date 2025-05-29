import time
import random
import math
from .base import Agent
from .heuristics.base import CompositeHeuristic
from .heuristics.core import EmptyTilesHeuristic, MonotonicityHeuristic, SmoothnessHeuristic, CornerMaxHeuristic, MergePotentialHeuristic, MaxValueHeuristic
from .expectimax import ExpectimaxAgent

class SimpleMCTSNode:
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
        """Select the best child using UCB1 formula."""
        if not self.children:
            return None
            
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                return child  # Prioritize unvisited children
                
            # UCB1 formula
            exploitation = child.total_score / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                
        return best_child

class MctsAgent(Agent):
    def __init__(self, name="MCTS", thinking_time=0.5, exploration_weight=1.414, 
                 rollout_type="expectimax"):
        super().__init__(name)
        self.thinking_time = thinking_time
        self.exploration_weight = exploration_weight
        self.rollout_type = rollout_type  # "random" or "expectimax"
        
        # Composite heuristic for evaluation
        self.heuristic = CompositeHeuristic()
        self.heuristic.add_heuristic(EmptyTilesHeuristic(), 3.0)
        self.heuristic.add_heuristic(MonotonicityHeuristic(), 1.5) 
        self.heuristic.add_heuristic(CornerMaxHeuristic(), 5.0)
        self.heuristic.add_heuristic(SmoothnessHeuristic(), 2.0)
        self.heuristic.add_heuristic(MergePotentialHeuristic(), 2.5)
        self.heuristic.add_heuristic(MaxValueHeuristic(), 3.0)

        if self.rollout_type == "expectimax":
            self.expectimax_agent = ExpectimaxAgent(thinking_time=999.0)  # Effectively disable time limits
        else:
            self.expectimax_agent = None

        self.stats = {
            "search_iterations": 0,     # MCTS iterations
            "max_depth_reached": 0,     # Maximum tree depth explored
            "avg_reward": 0.0           # Average reward from simulations
        }
    
    def get_move(self, game):
        """Returns the best move using MCTS."""

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
        
        root = SimpleMCTSNode(game.copy())
        
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
        
        self.stats["search_iterations"] = iterations
        if iterations > 0:
            self.stats["avg_reward"] = total_reward / iterations
        
        return self._best_move(root)
    
    def _select(self, node):
        """Selection phase: traverse down using UCB1 until we reach a leaf."""
        depth = 0
        while not node.is_terminal and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
            if node is None:
                break
            depth += 1
        
        self.stats["max_depth_reached"] = max(self.stats["max_depth_reached"], depth)
        return node
    
    def _expand(self, node):
        """Expansion phase: add a new child for an unexplored move."""
        available_moves = node.game_state.get_available_moves()
        unexplored_moves = [move for move in available_moves if move not in node.children]
        if not unexplored_moves:
            return node
        move = random.choice(unexplored_moves)
        
        new_game = node.game_state.copy()
        new_game.step(move)
        
        child = SimpleMCTSNode(new_game, parent=node, move=move)
        node.add_child(move, child)
        
        return child
    
    def _simulate(self, node):
        """Simulation phase: use expectimax (hybrid model) or random rollout."""
        
        if self.rollout_type == "expectimax":
            # Use expectimax evaluation with fixed depth of 2
            final_score = self.expectimax_agent._expectimax(
                node.game_state,
                2,  # Fixed depth
                True,  # is_max_player
                time.time()  # thinking_time is 999s so won't timeout
            )
        
        elif self.rollout_type == "random":
            sim_game = node.game_state.copy()
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
        
        else:
            raise ValueError(f"Invalid rollout_type: {self.rollout_type}. Must be 'random' or 'expectimax'")
        
        return final_score
    
    def _backpropagate(self, node, reward):
        """Backpropagation phase: update visit counts and scores up the tree."""
        while node is not None:
            node.visits += 1
            node.total_score += reward
            node = node.parent
    
    def _best_move(self, root):
        """Select the best move (most visited child)."""
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
        return {
            'thinking_time': self.thinking_time,
            'exploration_weight': self.exploration_weight,
            'rollout_type': self.rollout_type
        } 