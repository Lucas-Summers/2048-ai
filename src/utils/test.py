from analyzer import ParallelAgentAnalyzer
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

def create_agent(agent_type, thinking_time=0.5):
    """Create a single agent of the specified type."""
    agent_configs = {
        'random': lambda: RandomAgent(name="Random"),
        'greedy': lambda: GreedyAgent(
            name="Greedy",
            tile_weight=1.0,
            score_weight=0.1
        ),
        'mcts': lambda: MctsAgent(
            name="MCTS", 
            thinking_time=thinking_time,
            exploration_weight=1.414,
            rollout_type="random"
        ),
        'hybrid': lambda: MctsAgent(
            name="Hybrid (Expectimax)",
            thinking_time=thinking_time,
            exploration_weight=1.414,
            rollout_type="expectimax"
        ),
        'expectimax': lambda: ExpectimaxAgent(
            name="Expectimax",
            thinking_time=thinking_time
        ),
        'rl': lambda: RLAgent(
            name="RL",
            training=False
        )
    }
    
    if agent_type not in agent_configs:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agent_configs.keys())}")
    
    return agent_configs[agent_type]()

def print_results(stats, total_time, num_games):
    """Print comprehensive results for the agent."""
    print(f"\nRESULTS FOR {stats['agent_name']}")
    print("="*50)
    print(f"Games Completed: {len(stats['scores'])}/{num_games}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Time per Game: {total_time / len(stats['scores']):.3f} seconds")
    
    print(f"\nPerformance Metrics:")
    print(f"  Average Score: {stats['avg_score']:.2f}")
    print(f"  Min Score: {stats['min_score']:,.0f}")
    print(f"  Max Score: {stats['max_score']:,.0f}")
    print(f"  Win Rate: {stats['win_rate']:.2f}% ({int(stats['win_rate'] * len(stats['scores']) / 100)} wins)")
    print(f"  Average Moves: {stats['avg_moves']:.2f}")
    print(f"  Median Max Tile: {int(stats['median_max_tile'])}")
    print(f"  Absolute Max Tile: {stats.get('absolute_max_tile', 0)}")
    print(f"  Efficiency: {stats['efficiency']:.3f} (score/move)")
     
    max_tiles = stats['max_tiles']
    tile_counts = {}
    for tile in max_tiles:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    print(f"\nMax Tile Distribution:")
    for tile in sorted(tile_counts.keys(), reverse=True)[:6]:
        percentage = tile_counts[tile] / len(max_tiles) * 100
        print(f"  {tile:>4}: {tile_counts[tile]:>3} games ({percentage:>5.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a single 2048 AI agent")
    parser.add_argument("agent_type", 
                       choices=['random', 'greedy', 'mcts', 'hybrid', 'expectimax', 'rl'],
                       help="Type of agent to test")
    parser.add_argument("-n", "--num_games", type=int, default=100,
                       help="Number of games to simulate (default: 100)")
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
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.processes is None:
        num_processes = max(1, cpu_count() - 1)
    else:
        num_processes = args.processes
    
    print(f"Testing {args.agent_type} agent")
    print(f"Games: {args.num_games}")
    print(f"Thinking time: {args.thinking_time}s")
    print(f"Parallel settings: {num_processes} processes, {args.batch_size} batch size, {args.threads_per_batch} threads/batch")
    
    # Create agent and analyzer
    agent = create_agent(args.agent_type, thinking_time=args.thinking_time)
    
    analyzer = ParallelAgentAnalyzer(
        num_processes=num_processes, 
        batch_size=args.batch_size,
        games_per_thread=args.threads_per_batch
    )
    
    print(f"\nStarting simulation for {agent.name}...")
    start_time = time.time()
    
    stats = analyzer.evaluate_agent(agent, num_games=args.num_games, show_progress=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print_results(stats, total_time, args.num_games)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"{args.output_dir}/{args.agent_type}_test_{timestamp}.json"
    analyzer.save_results(results_file)
    
    print(f"\nResults saved to: {results_file}")
    print("Test completed!")