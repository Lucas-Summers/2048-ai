import time
import random
import math
from copy import deepcopy
from .base import Agent
from .heuristics.base import CompositeHeuristic
from .heuristics.basic import EmptyTilesHeuristic, MonotonicityHeuristic, SmoothnessHeuristic, CornerMaxHeuristic, MergePotentialHeuristic

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, depth=0, heuristic=None, eval_cache=None, max_branching_factor=4, ):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.depth = depth
    
        self.children = {}
        self.visits = 0
        self.score = 0
        self.is_terminal = game_state.is_game_over()

        # For pruning
        self.heuristic = heuristic
        self.eval_cache = eval_cache if eval_cache is not None else {}
        self.max_branching_factor = max_branching_factor

        self._available_moves = None
        self._pruned_moves = None
        
    def add_child(self, move, child_node):
        """
        Add a child node for the given move
        """
        self.children[move] = child_node
        
    def is_fully_expanded(self):
        """
        Check if all possible moves from this state have been explored
        """
        if self.is_terminal:
            return True
            
        return len(self.children) == len(self.get_pruned_moves())

    def get_pruned_moves(self):
        """
        Get filtered list of moves based on heuristic evaluation
        """
        if self._pruned_moves is None:
            valid_moves = self.get_available_moves()
            
            # If fewer moves than our limit, use all of them
            if len(valid_moves) <= self.max_branching_factor:
                self._pruned_moves = valid_moves.copy()  # Copy to be safe
            else:
                move_scores = []
                for move in valid_moves:
                    temp_game = deepcopy(self.game_state)
                    temp_game.step(move)
                    
                    board_hash = hash(temp_game.board)
                    
                    if board_hash in self.eval_cache:
                        heuristic_value = self.eval_cache[board_hash]
                    else:
                        heuristic_value = self.heuristic.evaluate(temp_game.board.grid)
                        self.eval_cache[board_hash] = heuristic_value
                    
                    score_component = temp_game.score / 10000.0
                    move_scores.append((move, heuristic_value + score_component))
                
                move_scores.sort(key=lambda x: x[1], reverse=True)
                self._pruned_moves = [move for move, _ in move_scores[:self.max_branching_factor]]
                
        return self._pruned_moves

    def get_available_moves(self):
        """
        Get all available moves
        """
        if self._available_moves is None:
            self._available_moves = self.game_state.board.get_available_moves()
        return self._available_moves
        
    def best_child(self, exploration_weight=1.0):
        """
        Select the best child node according to the UCB1 formula
        """
        if not self.children:
            return None
            
        scores = {}
        for move, child in self.children.items():
            # Avoid division by zero
            if child.visits == 0:
                scores[move] = float('inf')
            else:
                exploitation = child.score / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                scores[move] = exploitation + exploration
                
        return self.children[max(scores, key=scores.get)]

class MctsAgent(Agent):
    def __init__(self, name="MCTS", exploration_weight=1.414, thinking_time=0.5, max_sim_depth=100, 
                 visit_score_ratio=0.6, rollout_type="random", max_branching_factor=4):

        super().__init__(name)
        self.exploration_weight = exploration_weight
        self.thinking_time = thinking_time
        self.max_sim_depth = max_sim_depth
        self.visit_score_ratio = visit_score_ratio
        self.rollout_type = rollout_type
        self.max_branching_factor = max_branching_factor
        
        # Create composite heuristic for evaluation
        self.heuristic = CompositeHeuristic()
        self.heuristic.add_heuristic(EmptyTilesHeuristic(), 3.0)
        self.heuristic.add_heuristic(MonotonicityHeuristic(), 2.5) 
        self.heuristic.add_heuristic(CornerMaxHeuristic(), 3.0)
        self.heuristic.add_heuristic(SmoothnessHeuristic(), 1.0)
        self.heuristic.add_heuristic(MergePotentialHeuristic(), 1.2)

        # Heuristic evaluation cache
        self.eval_cache = {}
        
        self.stats = {
            "iterations": 0,          # Number of MCTS iterations
            "max_depth": 0,           # Maximum depth reached in the tree
            "thinking_time": 0.0,     # Time spent on this move decision
            "avg_reward": 0.0,        # Average reward from simulations
            "avg_sim_depth": 0.0,     # Average depth of simulations
            "rollout_type": self.rollout_type, # Rollout method
            "cache_hits": 0 # Amount of times evaluation cache is used
        }
        
    def get_move(self, game):
        """
        Returns the best move for the given game state using MCTS
        """
        # Reset stats
        self.stats = {
            "iterations": 0,
            "max_depth": 0,
            "thinking_time": 0.0,
            "avg_reward": 0.0,
            "avg_sim_depth": 0.0,
            "rollout_type": self.rollout_type,
            "cache_hits": 0
        }

        # Clear evaluation cache periodically to prevent memory buildup
        # Only keep cache between moves if thinking time is very short
        if self.thinking_time > 0.2:
            self.eval_cache = {}
        
        valid_moves = game.board.get_available_moves()
        if not valid_moves:
            return -1
        if len(valid_moves) == 1:
            return valid_moves[0]
 
        root_game = None
        try:
            root_game = game.copy()
        except (AttributeError, NotImplementedError):
            root_game = deepcopy(game)
        root = MCTSNode(
            root_game, 
            depth=0,
            heuristic=self.heuristic,
            eval_cache=self.eval_cache,
            max_branching_factor=self.max_branching_factor
        )
        
        start_time = time.time()
        iterations = 0
        total_sim_depth = 0
        while time.time() - start_time < self.thinking_time:
            node, depth = self._select(root)
            if not node.is_terminal and node.visits > 0:
                node = self._expand(node)
            reward, sim_depth = self._simulate(node)
            self._backpropagate(node, reward)
            
            iterations += 1
            self.stats["max_depth"] = max(self.stats["max_depth"], depth)
            total_sim_depth += sim_depth
        
        self.stats["thinking_time"] = time.time() - start_time
        self.stats["iterations"] = iterations
        if root.visits > 0:
            self.stats["avg_reward"] = root.score / root.visits
        if iterations > 0:
            self.stats["avg_sim_depth"] = total_sim_depth / iterations
        
        best_move = self._best_move(root)
        
        return best_move
    
    def _select(self, node):
        """
        Select a leaf node by traversing down the tree using UCB1
        """
        current_depth = node.depth
        max_depth = current_depth
        
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return node, max_depth
                
            child = node.best_child(self.exploration_weight)
            if child is None:
                break
                
            node = child
            current_depth = node.depth
            max_depth = max(max_depth, current_depth)
        
        return node, max_depth
    
    def _expand(self, node):
        """
        Expand the tree by creating a child node for an unexplored move
        """
        #valid_moves = node.game_state.board.get_available_moves()
        valid_moves = node.get_pruned_moves() 
        unexplored_moves = [move for move in valid_moves if move not in node.children]
        if not unexplored_moves:
            return node
        move = random.choice(unexplored_moves)

        new_game_state = None
        try:
            new_game_state = node.game_state.copy()
        except (AttributeError, NotImplementedError):
            new_game_state = deepcopy(node.game_state)
        new_game_state.step(move)
        
        child = MCTSNode(
            new_game_state, 
            parent=node, 
            move=move, 
            depth=node.depth + 1,
            heuristic=self.heuristic,
            eval_cache=self.eval_cache,
            max_branching_factor=self.max_branching_factor
        )
        node.add_child(move, child)
        
        self.stats["max_depth"] = max(self.stats["max_depth"], child.depth)
        
        return child
    
    def _simulate(self, node):
        """
        Simulate a playout from the current node, using the specified rollout type
        """
        sim_state = None
        try:
            sim_state = node.game_state.copy()
        except (AttributeError, NotImplementedError):
            sim_state = deepcopy(node.game_state)
        
        depth = 0
        effective_max_depth = min(self.max_sim_depth, 
                               max(3, int(self.thinking_time * 100)))
        while not sim_state.is_game_over() and depth < effective_max_depth:
            valid_moves = sim_state.board.get_available_moves()
            
            if not valid_moves:
                break
 
            if self.rollout_type == "greedy":
                move = self._choose_heuristic_move(sim_state, valid_moves)
            elif self.rollout_type == "random":
                move = random.choice(valid_moves)
            elif self.rollout_type == "expectimax":
                # TODO: add here for hybrid model
                move = random.choice(valid_moves)
            else:
                move = random.choice(valid_moves)
            sim_state.step(move)
            depth += 1
        
        board_hash = hash(sim_state.board)
        if board_hash in self.eval_cache:
            heuristic_value = self.eval_cache[board_hash]
            self.stats["cache_hits"] += 1
        else:
            heuristic_value = self.heuristic.evaluate(sim_state.board.grid)
            self.eval_cache[board_hash] = heuristic_value

        score_component = sim_state.score / 10000.0  # Reduced weight
        reward = heuristic_value + score_component
        
        return reward, depth
 
    def _choose_heuristic_move(self, game_state, valid_moves):
        """
        Choose the best move based on heuristic evaluation
        """
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            next_state = None
            try:
                next_state = game_state.copy()
            except (AttributeError, NotImplementedError):
                next_state = deepcopy(game_state)
            next_state.step(move)

            board_hash = hash(next_state.board)
            if board_hash in self.eval_cache:
                heuristic_value = self.eval_cache[board_hash]
                self.stats["cache_hits"] += 1
            else:
                heuristic_value = self.heuristic.evaluate(next_state.board.grid)
                self.eval_cache[board_hash] = heuristic_value
            
            score_component = next_state.score / 10000.0
            score = heuristic_value + score_component
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move is not None else random.choice(valid_moves)
    
    def _backpropagate(self, node, reward):
        """
        Backpropagate the reward and visits up the tree
        """
        while node is not None:
            node.visits += 1
            node.score += reward
            node = node.parent
    
    def _best_move(self, root):
        """
        Choose the best move based on a weighted combination of visit count and average score
        """
        # If no children, return a random valid move
        if not root.children:
            return random.choice(root.game_state.board.get_available_moves())
        
        
        total_visits = sum(child.visits for child in root.children.values())
        max_avg_score = max((child.score / child.visits) if child.visits > 0 else 0 
                            for child in root.children.values())
        
        # Prevent division by zero
        if total_visits == 0:
            return random.choice(list(root.children.keys()))
        if max_avg_score == 0:
            return max(root.children.items(), key=lambda x: x[1].visits)[0]
        
        best_move = None
        best_combined_score = float('-inf')

        for move, child in root.children.items():
            if child.visits == 0:
                continue
                
            visit_ratio = child.visits / total_visits
            avg_score = child.score / child.visits
            normalized_score = avg_score / max_avg_score if max_avg_score > 0 else 0
            
            visits_weight = self.visit_score_ratio
            score_weight = 1.0 - self.visit_score_ratio
            combined_score = (
                visits_weight * visit_ratio + 
                score_weight * normalized_score
            )
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_move = move
        
        if best_move is None:
            best_move = random.choice(list(root.children.keys()))
        return best_move
    
    def get_stats(self):
        return self.stats
    
    def get_config(self):
        return {
            'thinking_time': self.thinking_time,
            'exploration_weight': self.exploration_weight,
            'max_sim_depth': self.max_sim_depth,
            'visit_score_ratio': self.visit_score_ratio,
            'rollout_type': self.rollout_type,
            'max_branching_factor': self.max_branching_factor
        }
