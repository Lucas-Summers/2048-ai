class Agent:
    """Base class for all 2048 AI agents."""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
    
    def get_move(self, game):
        """Returns the best move for the given game state."""
        raise NotImplementedError("Subclasses must implement get_move()")
    
    def get_stats(self):
        """Returns the agent's runtime statistics."""
        return {'name': self.name}
    
    def get_config(self):
        """Returns the agent's configuration parameters."""
        return {}
