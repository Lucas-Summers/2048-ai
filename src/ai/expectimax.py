from .base import Agent

class ExpectimaxAgent(Agent):
    """
    Agent that uses Expectimax search to find optimal moves in 2048.
    Accounts for randomness in tile placement.
    """
    
    def __init__(self, name="Expectimax", max_depth=3):
        super().__init__(name)
        self.max_depth = max_depth
    
    def get_move(self, game):
        pass
    
    def _expectimax(self, game, depth, is_max_player):
        pass
    
    def _evaluate_board(self, board):
        pass
    
    def get_config(self):
        return {
            'max_depth': self.max_depth
        }
