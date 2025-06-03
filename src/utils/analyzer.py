import time
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

def play_single_game(args):
    """Play a single game with the given agent and parameters."""
    agent_config, game_id, agent_class_info = args
    
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from game.game import Game2048
        
        module_name, class_name = agent_class_info
        module = __import__(module_name, fromlist=[class_name])
        agent_class = getattr(module, class_name)
        agent = agent_class(**agent_config)
        
        game = Game2048()
        moves_count = 0
        game_start_time = time.time()
        move_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        
        while not game.is_game_over():
            move = agent.get_move(game)
            if move >= 0 and move < 4:
                move_distribution[move] += 1
            
            game.step(move)
            moves_count += 1
        
        game_duration = time.time() - game_start_time
        
        return {
            'game_id': game_id,
            'score': game.score,
            'max_tile': game.board.get_max_tile(),
            'moves': moves_count,
            'duration': game_duration,
            'move_distribution': move_distribution,
            'win': game.board.get_max_tile() >= 2048
        }
        
    except Exception as e:
        return {
            'game_id': game_id,
            'error': str(e),
            'score': 0,
            'max_tile': 0,
            'moves': 0,
            'duration': 0,
            'move_distribution': {0: 0, 1: 0, 2: 0, 3: 0},
            'win': False
        }


def play_game_batch_parallel(args):
    """Play a batch of games with the same agent IN PARALLEL using threads."""
    agent_config, batch_start_id, batch_size, agent_class_info, games_per_thread = args
    
    # Just run sequentially for small batches to avoid thread overhead
    if batch_size <= 2:
        results = []
        for i in range(batch_size):
            game_id = batch_start_id + i
            single_game_args = (agent_config, game_id, agent_class_info)
            result = play_single_game(single_game_args)
            results.append(result)
        return results
    
    game_args = []
    for i in range(batch_size):
        game_id = batch_start_id + i
        game_args.append((agent_config, game_id, agent_class_info))
    
    max_threads = min(games_per_thread, batch_size)
    all_results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_game = {
            executor.submit(play_single_game, game_arg): i 
            for i, game_arg in enumerate(game_args)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_game):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                game_idx = future_to_game[future]
                error_result = {
                    'game_id': batch_start_id + game_idx,
                    'error': str(e),
                    'score': 0,
                    'max_tile': 0,
                    'moves': 0,
                    'duration': 0,
                    'move_distribution': {0: 0, 1: 0, 2: 0, 3: 0},
                    'win': False
                }
                all_results.append(error_result)
    
    # Sort results by game_id to maintain order
    all_results.sort(key=lambda x: x['game_id'])
    return all_results


class ParallelAgentAnalyzer:
    def __init__(self, num_processes=None, batch_size=10, games_per_thread=2):
        self.results = {}
        self.num_processes = num_processes or max(1, cpu_count() - 1)  # Leave one core free
        self.batch_size = batch_size
        self.games_per_thread = games_per_thread
        
    def _get_agent_class_info(self, agent):
        """Get information needed to reconstruct the agent in a subprocess."""
        module_name = agent.__class__.__module__
        class_name = agent.__class__.__name__
        return (module_name, class_name)
    
    def _get_agent_config(self, agent):
        """Extract agent configuration for reconstruction."""
        config = agent.get_config().copy()
        config['name'] = agent.name
        return config
    
    def evaluate_agent(self, agent, num_games=100, show_progress=True):
        """Evaluate an agent by running games in parallel with 2-level parallelism."""
        stats = {
            'agent_name': agent.name,
            'agent_config': self._get_agent_config(agent),
            'scores': [],
            'max_tiles': [],
            'moves_per_game': [],
            'game_durations': [],
            'move_distribution': {0: 0, 1: 0, 2: 0, 3: 0},
            'win_rate': 0,
            'absolute_max_tile': 0,
            'min_score': 0,
            'max_score': 0,
        }
        
        if show_progress:
            print(f"Evaluating agent: {agent.name} (w/ {self.num_processes} processes + {self.games_per_thread} threads)")
        
        agent_config = self._get_agent_config(agent)
        agent_class_info = self._get_agent_class_info(agent)
        num_batches = (num_games + self.batch_size - 1) // self.batch_size
        
        batch_args = []
        for batch_idx in range(num_batches):
            batch_start_id = batch_idx * self.batch_size
            current_batch_size = min(self.batch_size, num_games - batch_start_id)
            
            batch_args.append((
                agent_config,
                batch_start_id,
                current_batch_size,
                agent_class_info,
                self.games_per_thread
            ))
        
        all_results = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            future_to_batch = {
                executor.submit(play_game_batch_parallel, args): batch_idx 
                for batch_idx, args in enumerate(batch_args)
            }
            
            if show_progress:
                progress_bar = tqdm(total=num_games, desc="Playing games")
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    if show_progress:
                        progress_bar.update(len(batch_results))
                        
                        if all_results:
                            valid_results = [r for r in all_results if 'error' not in r]
                            if valid_results:
                                avg_score = np.mean([r['score'] for r in valid_results])
                                wins = sum(1 for r in valid_results if r['win'])
                                max_tile_so_far = max([r['max_tile'] for r in valid_results])
                                progress_bar.set_postfix(
                                    avg_score=f"{avg_score:.1f}",
                                    wins=f"{wins}/{len(valid_results)}",
                                    max_tile=f"{max_tile_so_far}"
                                )
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
            
            if show_progress:
                progress_bar.close()
        
        valid_results = [r for r in all_results if 'error' not in r]
        error_count = len(all_results) - len(valid_results)
        if error_count > 0:
            print(f"Warning: {error_count} games failed and were excluded from results")
        if not valid_results:
            print(f"Error: No valid games completed for {agent.name}")
            return stats
        
        for result in valid_results:
            stats['scores'].append(result['score'])
            stats['max_tiles'].append(result['max_tile'])
            stats['moves_per_game'].append(result['moves'])
            stats['game_durations'].append(result['duration'])
            
            
            if result['max_tile'] > stats['absolute_max_tile']:
                stats['absolute_max_tile'] = result['max_tile']
            
            for move, count in result['move_distribution'].items():
                stats['move_distribution'][move] += count
            
            if result['win']:
                stats['win_rate'] += 1
        
        if stats['scores']:
            stats['min_score'] = int(min(stats['scores']))
            stats['max_score'] = int(max(stats['scores']))
        
        stats['win_rate'] = float((stats['win_rate'] / len(valid_results)) * 100)
        stats['avg_score'] = float(np.mean(stats['scores']))
        stats['avg_moves'] = float(np.mean(stats['moves_per_game']))
        stats['avg_game_duration'] = float(np.mean(stats['game_durations']))
        stats['median_max_tile'] = float(np.median(stats['max_tiles']))
        stats['efficiency'] = float(stats['avg_score'] / stats['avg_moves']) if stats['avg_moves'] > 0 else 0.0
        stats['absolute_max_tile'] = int(stats['absolute_max_tile'])
        
        self.results[agent.name] = stats
        
        if show_progress:
            print(f"Evaluation complete for {agent.name}:")
            print(f"  Games completed: {len(valid_results)}/{num_games}")
            print(f"  Average Score: {stats['avg_score']:.2f}")
            print(f"  Min Score: {stats['min_score']:,.0f}")
            print(f"  Max Score: {stats['max_score']:,.0f}")
            print(f"  Win Rate: {stats['win_rate']:.2f}%")
            print(f"  Median Max Tile: {stats['median_max_tile']}")
            print(f"  Absolute Max Tile: {stats['absolute_max_tile']}")
            print(f"  Efficiency: {stats['efficiency']:.3f}")
        
        return stats
    
    def compare_agents(self, agents, num_games=100, show_progress=True):
        """Compare multiple agents by evaluating each one using parallelism."""
        agents_to_evaluate = [agent for agent in agents if agent.name not in self.results]
        
        if show_progress and agents_to_evaluate:
            print(f"Comparing {len(agents)} agents on {num_games} games ({len(agents_to_evaluate)} need evaluation)")
        
        for agent in agents_to_evaluate:
            self.evaluate_agent(agent, num_games, show_progress)
        
        return self.get_comparison_data()
    
    def get_comparison_data(self):
        """Get comparison data for all evaluated agents."""
        comparison_data = {}
        
        for agent_name, stats in self.results.items():
            comparison_data[agent_name] = {
                'avg_score': stats['avg_score'],
                'win_rate': stats['win_rate'],
                'avg_moves': stats['avg_moves'],
                'avg_game_duration': stats['avg_game_duration'],
                'median_max_tile': stats['median_max_tile'],
                'absolute_max_tile': stats.get('absolute_max_tile', 0),
                'efficiency': stats['efficiency'],
                'min_score': stats['min_score'],
                'max_score': stats['max_score'],
            }
        
        return comparison_data
    
    def save_results(self, filename):
        """Save results to a JSON file."""
        serializable_results = self._convert_numpy_types(self.results)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
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
        """Load results from a JSON file (same format as sequential analyzer)."""
        with open(filename, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded results for {len(self.results)} agents from {filename}")