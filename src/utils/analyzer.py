import time
import numpy as np
import json

class AgentAnalyzer:
    def __init__(self):
        self.results = {}
    
    def evaluate_agent(self, agent, num_games=100, max_moves=2000, show_progress=True):
        stats = {
            'agent_name': agent.name,
            'agent_config': agent.get_config(),
            'scores': [],
            'max_tiles': [],
            'moves_per_game': [],
            'game_durations': [],
            'move_distribution': {0: 0, 1: 0, 2: 0, 3: 0},
            'win_rate': 0,
        }
        
        if show_progress:
            print(f"Evaluating agent: {agent.name}")
        
        for i in range(num_games):
            from game.game import Game2048
            if show_progress and (i % 10 == 0 or i == num_games - 1):
                print(f"Playing game {i+1}/{num_games}")
            
            game = Game2048()
            moves_count = 0
            game_start_time = time.time()
            
            while not game.is_game_over() and moves_count < max_moves:
                move = agent.get_move(game)
                if move >= 0 and move < 4:
                    stats['move_distribution'][move] += 1
                
                game.step(move)
                moves_count += 1
            
            game_duration = time.time() - game_start_time
            stats['scores'].append(game.score)
            stats['max_tiles'].append(game.board.get_max_tile())
            stats['moves_per_game'].append(moves_count)
            stats['game_durations'].append(game_duration)
            if game.board.get_max_tile() >= 2048:
                stats['win_rate'] += 1
        
        stats['win_rate'] = float((stats['win_rate'] / num_games) * 100)
        stats['avg_score'] = float(np.mean(stats['scores']))
        stats['avg_moves'] = float(np.mean(stats['moves_per_game']))
        stats['avg_game_duration'] = float(np.mean(stats['game_durations']))
        stats['median_max_tile'] = float(np.median(stats['max_tiles']))
        stats['efficiency'] = float(stats['avg_score'] / stats['avg_moves']) if stats['avg_moves'] > 0 else 0.0
        self.results[agent.name] = stats
        
        if show_progress:
            print(f"Evaluation complete for {agent.name}:")
            print(f"  Average Score: {stats['avg_score']:.2f}")
            print(f"  Win Rate: {stats['win_rate']:.2f}%")
            print(f"  Median Max Tile: {stats['median_max_tile']}")
            print(f"  Efficiency: {stats['efficiency']:.3f}")
        
        return stats
    
    def compare_agents(self, agents, num_games=100, max_moves=2000, show_progress=True):
        for agent in agents:
            if agent.name not in self.results:
                self.evaluate_agent(agent, num_games, max_moves, show_progress)
        
        return self.get_comparison_data()
    
    def get_comparison_data(self):
        comparison_data = {}
        
        for agent_name, stats in self.results.items():
            comparison_data[agent_name] = {
                'avg_score': stats['avg_score'],
                'win_rate': stats['win_rate'],
                'avg_moves': stats['avg_moves'],
                'avg_game_duration': stats['avg_game_duration'],
                'median_max_tile': stats['median_max_tile'],
                'efficiency': stats['efficiency']
            }
        
        return comparison_data
    
    def save_results(self, filename):
        serializable_results = self._convert_numpy_types(self.results)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def _convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def load_results(self, filename):
        with open(filename, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded results for {len(self.results)} agents from {filename}")
