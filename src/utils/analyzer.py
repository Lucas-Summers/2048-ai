import time
import numpy as np
import json


class AgentAnalyzer:
    """
    Analyzes and compares performance of 2048 AI agents.
    Focuses on collecting, aggregating and storing performance metrics.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_agent(self, agent, num_games=100, max_moves=2000, show_progress=True):
        """
        Evaluate an agent by playing multiple games.
        
        Args:
            agent: Agent instance to evaluate
            num_games: Number of games to play
            max_moves: Maximum moves per game
            show_progress: Whether to display progress updates
        """
        from game.game import Game2048  # Import within function to avoid dependency issues
        
        stats = {
            'agent_name': agent.name,
            'agent_config': agent.get_config(),
            'scores': [],
            'max_tiles': [],
            'moves_per_game': [],
            'game_durations': [],
            'time_per_move': [],
            'valid_moves_per_turn': [],
            'move_distribution': {0: 0, 1: 0, 2: 0, 3: 0},
            'win_rate': 0,  # Games where 2048 tile was reached
        }
        
        if show_progress:
            print(f"Evaluating agent: {agent.name}")
        
        for i in range(num_games):
            if show_progress and (i % 10 == 0 or i == num_games - 1):
                print(f"Playing game {i+1}/{num_games}")
            
            game = Game2048()
            moves_count = 0
            game_start_time = time.time()
            valid_moves_history = []
            
            while not game.is_game_over() and moves_count < max_moves:
                # Track valid moves
                valid_moves = game.board.get_available_moves()
                valid_moves_history.append(len(valid_moves))
                
                # Time the agent's decision
                move_start_time = time.time()
                move = agent.get_move(game)
                move_duration = time.time() - move_start_time
                
                # Update timing stats
                stats['time_per_move'].append(move_duration)
                
                # Update move distribution
                if move >= 0 and move < 4:
                    stats['move_distribution'][move] += 1
                
                # Make the move
                game.step(move)
                moves_count += 1
            
            # Game is over - collect stats
            game_duration = time.time() - game_start_time
            
            stats['scores'].append(game.score)
            stats['max_tiles'].append(game.board.get_max_tile())
            stats['moves_per_game'].append(moves_count)
            stats['game_durations'].append(game_duration)
            stats['valid_moves_per_turn'].extend(valid_moves_history)
            
            # Check if 2048 tile was reached
            if game.board.get_max_tile() >= 2048:
                stats['win_rate'] += 1
        
        # Process aggregated stats
        stats['win_rate'] = (stats['win_rate'] / num_games) * 100
        stats['avg_score'] = np.mean(stats['scores'])
        stats['avg_moves'] = np.mean(stats['moves_per_game'])
        stats['avg_game_duration'] = np.mean(stats['game_durations'])
        stats['avg_move_time'] = np.mean(stats['time_per_move'])
        stats['avg_valid_moves'] = np.mean(stats['valid_moves_per_turn'])
        stats['median_max_tile'] = np.median(stats['max_tiles'])
        
        # Store results
        self.results[agent.name] = stats
        
        if show_progress:
            print(f"Evaluation complete for {agent.name}:")
            print(f"  Average Score: {stats['avg_score']:.2f}")
            print(f"  Win Rate: {stats['win_rate']:.2f}%")
            print(f"  Median Max Tile: {stats['median_max_tile']}")
        
        return stats
    
    def compare_agents(self, agents, num_games=100, max_moves=2000, show_progress=True):
        """
        Compare multiple agents by evaluating each one.
        
        Args:
            agents: List of agent instances to compare
            num_games: Number of games each agent should play
            max_moves: Maximum moves per game
            show_progress: Whether to display progress updates
        """
        for agent in agents:
            if agent.name not in self.results:
                self.evaluate_agent(agent, num_games, max_moves, show_progress)
        
        return self.get_comparison_data()
    
    def get_comparison_data(self):
        """
        Create a structured dictionary of comparison data from the results.
        
        Returns:
            Dictionary with key metrics for all evaluated agents
        """
        comparison_data = {}
        
        for agent_name, stats in self.results.items():
            comparison_data[agent_name] = {
                'avg_score': stats['avg_score'],
                'win_rate': stats['win_rate'],
                'avg_moves': stats['avg_moves'],
                'avg_move_time': stats['avg_move_time'] * 1000,  # Convert to ms
                'avg_game_duration': stats['avg_game_duration'],
                'median_max_tile': stats['median_max_tile'],
                'efficiency': stats['avg_score'] / stats['avg_moves'] if stats['avg_moves'] > 0 else 0
            }
        
        return comparison_data
    
    def save_results(self, filename):
        """Save results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def load_results(self, filename):
        """Load results from a JSON file."""
        with open(filename, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded results for {len(self.results)} agents from {filename}")
