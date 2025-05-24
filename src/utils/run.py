from analyzer import AgentAnalyzer
from visualizer import AgentVisualizer
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from ai.mcts import MctsAgent

def create_test_agents():
    agents = []
    
    mcts_agent = MctsAgent(
        name="MCTS_Optimized",
        thinking_time=0.1,
        exploration_weight=1.414,
        rollout_type="greedy",
        max_branching_factor=4
    )
    agents.append(mcts_agent)
    
    mcts_fast = MctsAgent(
        name="MCTS_Fast",
        thinking_time=0.05,
        exploration_weight=1.414,
        rollout_type="random",
        max_branching_factor=2
    )
    agents.append(mcts_fast)
    
    # TODO: Add other agent types here as you implement them
    # expectimax_agent = ExpectimaxAgent(name="Expectimax", depth=3)
    # agents.append(expectimax_agent)
    
    # rl_agent = RLAgent(name="DeepQ", model_path="models/dqn.pth")
    # agents.append(rl_agent)
    
    return agents

def run_analysis():
    print("Creating agents...")
    agents = create_test_agents()
    
    print("\nStarting analysis...")
    analyzer = AgentAnalyzer()
    analyzer.compare_agents(agents, num_games=5, max_moves=100, show_progress=True)
    
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")

    analyzer.save_results(f"{output_dir}/agent_results.json")
    comparison_data = analyzer.get_comparison_data()
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    for agent_name, metrics in comparison_data.items():
        print(f"\n{agent_name}:")
        print(f"  Average Score: {metrics['avg_score']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Average Moves: {metrics['avg_moves']:.2f}")
        print(f"  Efficiency: {metrics['efficiency']:.3f}")
        print(f"  Median Max Tile: {metrics['median_max_tile']:.0f}")
    
    print("\nCreating visualizations...")
    visualizer = AgentVisualizer(analyzer.results)
    
    print("Generating score distribution plots...")
    score_figs = visualizer.plot_score_distributions_separate(output_dir=output_dir)
    for fig in score_figs:
        plt.close(fig)
    fig2 = visualizer.plot_score_distributions_combined(output_dir=output_dir)
    if fig2:
        plt.close(fig2)
    
    print("Generating max tile distribution plots...")
    max_tile_figs = visualizer.plot_max_tile_distributions_separate(output_dir=output_dir)
    for fig in max_tile_figs:
        plt.close(fig)
    fig4 = visualizer.plot_max_tile_distributions_combined(output_dir=output_dir)
    if fig4:
        plt.close(fig4)
    
    print("Generating game length distribution plots...")
    game_length_figs = visualizer.plot_game_length_distributions_separate(output_dir=output_dir)
    for fig in game_length_figs:
        plt.close(fig)
    fig6 = visualizer.plot_game_length_distributions_combined(output_dir=output_dir)
    if fig6:
        plt.close(fig6)
    
    print("Generating performance metrics plots...")
    fig7 = visualizer.plot_win_rates(output_dir=output_dir)
    if fig7:
        plt.close(fig7)
    fig8 = visualizer.plot_efficiency(output_dir=output_dir)
    if fig8:
        plt.close(fig8)

    visualizer.save_comparison_csv("agent_comparison.csv", output_dir=output_dir)
    
    print("\nAnalysis complete!")
    print(f"All files saved in directory: {output_dir}/")

if __name__ == "__main__":
    run_analysis()
