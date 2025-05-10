# src/ai/heuristics/heuristic_base.py
class Heuristic:
    """Base class for all heuristic functions."""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.weight = 1.0
        
    def evaluate(self, board):
        """
        Evaluate board state and return a score.
        
        Args:
            board: Current board state
            
        Returns:
            score: Numerical score representing quality of the board
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
        
    def __call__(self, board):
        """Allow heuristics to be used as functions."""
        return self.evaluate(board)


class CompositeHeuristic(Heuristic):
    """Combines multiple heuristics with weights."""
    
    def __init__(self, name="Composite"):
        super().__init__(name)
        self.heuristics = []  # List of (heuristic, weight) tuples
    
    def add_heuristic(self, heuristic, weight=1.0):
        """Add a heuristic with an optional weight."""
        heuristic.weight = weight
        self.heuristics.append(heuristic)
        return self  # Allow chaining
        
    def evaluate(self, board):
        """Evaluate board using weighted sum of all heuristics."""
        if not self.heuristics:
            return 0.0
            
        return sum(h.evaluate(board) * h.weight for h in self.heuristics)
