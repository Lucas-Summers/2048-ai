from .base import Agent

class HeuristicAgent(Agent):
    """
    Agent that uses heuristic evaluation of potential moves to make decisions.
    Can be configured for simple evaluation or deeper search (A*/GBFS).
    """
    
    def __init__(self, name="Heuristic", search_depth=1, use_astar=False):
        super().__init__(name)
        self.search_depth = search_depth
        self.use_astar = use_astar
    
    def get_move(self, game):
        pass
    
    def _evaluate_board(self, board):
        pass
    
    def get_config(self):
        return {
            'search_depth': self.search_depth,
            'use_astar': self.use_astar
        }
