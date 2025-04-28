# src/ai/random_agent.py
import random
from src.game.game import Game2048

class RandomAgent:
    """
    A simple agent that makes random moves in the 2048 game.
    This serves as a baseline for comparing more sophisticated AI approaches.
    """
    
    def __init__(self):
        """Initialize the random agent."""
        pass
    
    def get_move(self, game):
        """
        Choose a random valid move from the available moves.
        
        Args:
            game: A Game2048 instance representing the current game state
            
        Returns:
            int: Direction to move (0: up, 1: right, 2: down, 3: left)
            Returns -1 if no valid moves are available
        """
        available_moves = game.board.get_available_moves()
        
        if not available_moves:
            return -1  # No valid moves
        
        # Choose a random move from available options
        return random.choice(available_moves)
