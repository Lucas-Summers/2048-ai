from analyzer import ParallelAgentAnalyzer
from visualizer import AgentVisualizer
import matplotlib.pyplot as plt
import sys
import os
import time
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
    
    random_agent = RandomAgent(name="Random")
    agents.append(random_agent)
    
    greedy_agent = GreedyAgent(
        name="Greedy",
        tile_weight=1.0,
        score_weight=0.1
    )
    agents.append(greedy_agent)

    mcts_random = MctsAgent(
        name="MCTS_Random", 
        thinking_time=thinking_time,
        exploration_weight=1.414,
        rollout_type="random"
    )
    agents.append(mcts_random)
    
    mcts_expectimax = MctsAgent(
        name="MCTS_Expectimax",
        thinking_time=thinking_time,
        exploration_weight=1.414,
        rollout_type="expectimax"
    )
    agents.append(mcts_expectimax)
    
    expectimax_agent = ExpectimaxAgent(
        name="Expectimax",
        thinking_time=thinking_time
    )
    agents.append(expectimax_agent)
    
    rl_agent = RLAgent(
        name="RL",
        training=False
    )
    agents.append(rl_agent)
    
    return agents

if __name__ == "__main__":
    thinking_time = 0.5
    num_games = 100
    output_dir = "../../results"
    num_processes = None  # Auto-detect
    batch_size = 10
    games_per_thread = 10
    
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/agent_results.json"
    
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}...")
        
        visualizer = AgentVisualizer(results_file)
        comparison_data = visualizer.prepare_comparison_dataframe().to_dict('index')
        
        print("\nAgent Performance Results:")
        for agent_name, metrics in comparison_data.items():
            print(f"\n{agent_name}:")
            print(f"  Average Score: {metrics['Avg Score']:.2f}")
            print(f"  Win Rate: {metrics['Win Rate (%)']:.2f}%")
            print(f"  Average Moves: {metrics['Avg Moves']:.2f}")
            print(f"  Efficiency: {metrics['Efficiency (score/move)']:.3f}")
            print(f"  Median Max Tile: {metrics['Median Max Tile']:.0f}")
    
    else:
        print("No existing results found. Running simulations...")
        
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
        
        print("Starting simulations...")
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
            print(f"  Average Score: {metrics['avg_score']:.2f}")
            print(f"  Win Rate: {metrics['win_rate']:.2f}%")
            print(f"  Average Moves: {metrics['avg_moves']:.2f}")
            print(f"  Efficiency: {metrics['efficiency']:.3f}")
            print(f"  Median Max Tile: {metrics['median_max_tile']:.0f}")
            print(f"  Absolute Max Tile: {metrics['absolute_max_tile']:.0f}")
        
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