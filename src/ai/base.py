class Agent:
    """Base class for all 2048 AI agents."""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
    
    def get_move(self, game):
        """
        Returns the best move for the given game state.
        
        Args:
            game: Game2048 instance
            
        Returns:
            move: Integer representing the move (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
                 or -1 if no valid moves are available
        """
        raise NotImplementedError("Subclasses must implement get_move()")
    
    def get_stats(self):
        """
        Returns the agent's runtime statistics.
        This method should be implemented by all agent subclasses
        to provide meaningful statistics for display in the UI.
        
        Returns:
            dict: A dictionary of statistics specific to this agent
        """
        return {'name': self.name}
    
    def get_config(self):
        """
        Returns the agent's configuration parameters.
        Useful for analysis and reproducibility.
        """
        return {}
