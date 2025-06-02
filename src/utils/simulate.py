from analyzer import ParallelAgentAnalyzer
from visualizer import AgentVisualizer
import matplotlib.pyplot as plt
import sys
import os
import time
import argparse
from pathlib import Path
from multiprocessing import cpu_count

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from ai.mcts import MctsAgent
from ai.expectimax import ExpectimaxAgent
from ai.rl import RLAgent
from ai.greedy import GreedyAgent
from ai.random import RandomAgent

def create_test_agents(thinking_time=0.5):
    """Create agent configurations we want to test."""
    agents = []

    RL_MODEL_PATH = "src/utils/runs/2025-05-31_15-39-49/best_2842.pt"
    
    random_agent = RandomAgent(name="Random")
    agents.append(random_agent)
    
    greedy_agent = GreedyAgent(
        name="Greedy",
        tile_weight=1.0,
        score_weight=0.1
    )
    agents.append(greedy_agent)

    mcts = MctsAgent(
        name="MCTS", 
        thinking_time=thinking_time,
        exploration_weight=1.414,
        rollout_type="random"
    )
    agents.append(mcts)
    
    hybrid_expectimax = MctsAgent(
        name="Hybrid (Expectimax)",
        thinking_time=thinking_time,
        exploration_weight=1.414,
        rollout_type="expectimax"
    )
    agents.append(hybrid_expectimax)
    
    expectimax_agent = ExpectimaxAgent(
        name="Expectimax",
        thinking_time=thinking_time
    )
    agents.append(expectimax_agent)
    
    rl_agent = RLAgent.load_model(RL_MODEL_PATH, training=False, name="RL")
    agents.append(rl_agent)

    rl_rollout_agent = RLAgent.load_model(RL_MODEL_PATH, training=False, name="RL_Rollout")
    mcts_rl_agent = MctsAgent(
        name="MCTS_RLHybrid",
        thinking_time=thinking_time,
        exploration_weight=1.414,
        rollout_type="rl",
        rl_model_path=RL_MODEL_PATH
    )
    agents.append(mcts_rl_agent)
    
    return agents

def get_metric(metrics, *keys, default='N/A'):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return default

def safe_fmt(val, fmt):
    try:
        return format(val, fmt)
    except (ValueError, TypeError):
        return str(val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple 2048 AI agents")
    parser.add_argument("-n", "--num_games", type=int, default=100,
                       help="Number of games to simulate per agent (default: 100)")
    parser.add_argument("-t", "--thinking_time", type=float, default=0.5,
                       help="Thinking time per move in seconds (default: 0.5)")
    parser.add_argument("-o", "--output_dir", default="../../results",
                       help="Output directory for results (default: ../../results)")
    parser.add_argument("--processes", type=int, default=None,
                       help="Number of processes (default: auto-detect)")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size (default: 10)")
    parser.add_argument("--threads_per_batch", type=int, default=2,
                       help="Threads per batch (default: 2)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip simulation if results file already exists")
    
    args = parser.parse_args()
    
    thinking_time = args.thinking_time
    num_games = args.num_games
    output_dir = args.output_dir
    num_processes = args.processes
    batch_size = args.batch_size
    games_per_thread = args.threads_per_batch
    
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/agent_results.json"
    
    if os.path.exists(results_file) and args.skip_existing:
        print(f"Results file already exists: {results_file}")
        print("Use --skip_existing=false to overwrite or delete the file manually.")
        sys.exit(0)
    
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}...")
        
        visualizer = AgentVisualizer(results_file)
        comparison_data = visualizer.prepare_comparison_dataframe().to_dict('index')
        
        print("\nAgent Performance Results:")
        for agent_name, metrics in comparison_data.items():
            print(f"\n{agent_name}:")
            print(f"  Average Score: {safe_fmt(get_metric(metrics, 'Avg Score', 'avg_score'), '.2f')}")
            print(f"  Min Score: {safe_fmt(get_metric(metrics, 'Min Score', 'min_score'), ',.0f')}")
            print(f"  Max Score: {safe_fmt(get_metric(metrics, 'Max Score', 'max_score'), ',.0f')}")
            print(f"  Win Rate: {safe_fmt(get_metric(metrics, 'Win Rate (%)', 'win_rate'), '.2f')}%")
            print(f"  Average Moves: {safe_fmt(get_metric(metrics, 'Avg Moves', 'avg_moves'), '.2f')}")
            print(f"  Efficiency: {safe_fmt(get_metric(metrics, 'Efficiency (score/move)', 'efficiency'), '.3f')}")
            print(f"  Median Max Tile: {safe_fmt(get_metric(metrics, 'Median Max Tile', 'median_max_tile'), '.0f')}")
            if 'Absolute Max Tile' in metrics or 'absolute_max_tile' in metrics:
                print(f"  Absolute Max Tile: {safe_fmt(get_metric(metrics, 'Absolute Max Tile', 'absolute_max_tile'), '.0f')}")
    
    else:
        print("No existing results found. Running simulations...")
        print(f"Configuration:")
        print(f"  Games per agent: {num_games}")
        print(f"  Thinking time: {thinking_time}s")
        print(f"  Output directory: {output_dir}")
        
        if num_processes is None:
            num_processes = max(1, cpu_count() - 1)
        
        print(f"Parallel processing settings:")
        print(f"  Processes: {num_processes}")
        print(f"  Batch size: {batch_size}")
        print(f"  Games per thread: {games_per_thread}")
        max_concurrent = num_processes * games_per_thread
        print(f"  Max concurrent games: {max_concurrent}")
        
        analyzer = ParallelAgentAnalyzer(
            num_processes=num_processes, 
            batch_size=batch_size,
            games_per_thread=games_per_thread
        )
        
        print("\nStarting simulations...")
        agents = create_test_agents(thinking_time=thinking_time)
        start_time = time.time()
        analyzer.compare_agents(
            agents, 
            num_games=num_games, 
            show_progress=True
        )
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nSimulations completed in {total_time:.2f} seconds")
        print(f"Average time per game: {total_time / (num_games * len(agents)):.3f} seconds")
        
        analyzer.save_results(results_file)
        comparison_data = analyzer.get_comparison_data()
        
        print("\nAgent Performance Results:")
        for agent_name, metrics in comparison_data.items():
            print(f"\n{agent_name}:")
            print(f"  Average Score: {safe_fmt(get_metric(metrics, 'Avg Score', 'avg_score'), '.2f')}")
            print(f"  Min Score: {safe_fmt(get_metric(metrics, 'Min Score', 'min_score'), ',.0f')}")
            print(f"  Max Score: {safe_fmt(get_metric(metrics, 'Max Score', 'max_score'), ',.0f')}")
            print(f"  Win Rate: {safe_fmt(get_metric(metrics, 'Win Rate (%)', 'win_rate'), '.2f')}%")
            print(f"  Average Moves: {safe_fmt(get_metric(metrics, 'Avg Moves', 'avg_moves'), '.2f')}")
            print(f"  Efficiency: {safe_fmt(get_metric(metrics, 'Efficiency (score/move)', 'efficiency'), '.3f')}")
            print(f"  Median Max Tile: {safe_fmt(get_metric(metrics, 'Median Max Tile', 'median_max_tile'), '.0f')}")
            if 'Absolute Max Tile' in metrics or 'absolute_max_tile' in metrics:
                print(f"  Absolute Max Tile: {safe_fmt(get_metric(metrics, 'Absolute Max Tile', 'absolute_max_tile'), '.0f')}")
        
        visualizer = AgentVisualizer(analyzer.results)

    print("\nCreating visualizations...")
    fig_win = visualizer.plot_win_rates(output_dir=output_dir)
    if fig_win:
        plt.close(fig_win)
    fig_score = visualizer.plot_score_distributions(output_dir=output_dir)
    if fig_score:
        plt.close(fig_score)
    fig_tiles = visualizer.plot_max_tile_distributions(output_dir=output_dir)
    if fig_tiles:
        plt.close(fig_tiles)
    fig_moves = visualizer.plot_move_distributions(output_dir=output_dir)
    if fig_moves:
        plt.close(fig_moves)
    fig_scatter = visualizer.plot_score_vs_game_length(output_dir=output_dir)
    if fig_scatter:
        plt.close(fig_scatter)

    print("\nAnalysis complete!")
    print(f"All files saved in directory: {output_dir}/")