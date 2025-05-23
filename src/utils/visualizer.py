import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json


class AgentVisualizer:
    def __init__(self, analyzer_results=None):
        """
        Initialize the visualizer with optional analyzer results.
        
        Args:
            analyzer_results: Results dictionary from AgentAnalyzer or path to JSON file
        """
        # Set up styling
        self.setup_style()
        
        # Load results if provided
        self.results = {}
        if analyzer_results:
            if isinstance(analyzer_results, dict):
                self.results = analyzer_results
            elif isinstance(analyzer_results, str):
                self.load_results(analyzer_results)
    
    def setup_style(self):
        """Configure plot styling for consistency and improved aesthetics."""
        sns.set_style("whitegrid")
        # Use multiple color palettes for better distinction between agents
        colors = (sns.color_palette("Set1", 9) + 
                 sns.color_palette("Set2", 8) + 
                 sns.color_palette("Dark2", 8))
        self.agent_colors = colors
        self.move_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    
    def load_results(self, filename):
        """Load results from a JSON file."""
        with open(filename, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded visualization data for {len(self.results)} agents from {filename}")
        return self.results
    
    def save_comparison_csv(self, filename="agent_comparison.csv", output_dir="."):
        """Save comparison data to a CSV file."""
        if not self.results:
            print("No results to export")
            return
        
        df = self.prepare_comparison_dataframe()
        full_path = f"{output_dir}/{filename}"
        df.to_csv(full_path, index=False)
        print(f"Comparison data saved to {full_path}")
    
    def prepare_comparison_dataframe(self):
        """Prepare a DataFrame with comparison metrics for all agents."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for agent_name, stats in self.results.items():
            row = {
                "Agent": agent_name,
                "Avg Score": stats["avg_score"],
                "Win Rate (%)": stats["win_rate"],
                "Avg Moves": stats["avg_moves"],
                "Avg Game Duration (s)": stats["avg_game_duration"],
                "Median Max Tile": stats["median_max_tile"],
                "Efficiency (score/move)": stats["efficiency"]
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_score_distributions_separate(self, output_dir=".", figsize=(8, 6)):
        """Plot separate score distribution for each agent and save individual files."""
        if not self.results:
            print("No results to visualize")
            return []
        
        figures = []
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            fig, ax = plt.subplots(figsize=figsize)
            
            color = self.agent_colors[i % len(self.agent_colors)]
            sns.histplot(stats["scores"], bins=20, alpha=0.7, color=color, ax=ax)
            ax.set_title(f"{agent_name} - Score Distribution\nAvg: {stats['avg_score']:.0f}", 
                        fontsize=14, fontweight="bold")
            ax.set_xlabel("Score", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Save individual file
            safe_name = agent_name.replace(" ", "_").replace("/", "_")
            filename = f"{output_dir}/score_distribution_{safe_name}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            figures.append(fig)
        
        return figures
    
    def plot_score_distributions_combined(self, output_dir=".", figsize=(10, 6)):
        """Plot combined score distributions for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        plt.figure(figsize=figsize)
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            color = self.agent_colors[i % len(self.agent_colors)]
            sns.kdeplot(stats["scores"], label=f"{agent_name} (avg: {stats['avg_score']:.0f})", 
                       color=color, linewidth=2)
        
        plt.title("Combined Score Distributions", fontsize=14, fontweight="bold")
        plt.xlabel("Score", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save file
        filename = f"{output_dir}/score_distributions_combined.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        return plt.gcf()
    
    def plot_max_tile_distributions_separate(self, output_dir=".", figsize=(8, 6)):
        """Plot separate max tile distribution for each agent and save individual files."""
        if not self.results:
            print("No results to visualize")
            return []
        
        figures = []
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            fig, ax = plt.subplots(figsize=figsize)
            
            color = self.agent_colors[i % len(self.agent_colors)]
            max_tiles = stats["max_tiles"]
            unique_tiles = sorted(set(max_tiles))
            counts = [max_tiles.count(tile) for tile in unique_tiles]
            percentages = [count/len(max_tiles)*100 for count in counts]
            
            bars = ax.bar(range(len(unique_tiles)), percentages, color=color, alpha=0.7)
            ax.set_title(f"{agent_name} - Max Tile Distribution\nMedian: {stats['median_max_tile']:.0f}", 
                        fontsize=14, fontweight="bold")
            ax.set_xlabel("Max Tile", fontsize=12)
            ax.set_ylabel("Percentage (%)", fontsize=12)
            ax.set_xticks(range(len(unique_tiles)))
            ax.set_xticklabels(unique_tiles)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                if height > 5:  # Only show label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2., height/2,
                           f'{pct:.0f}%', ha='center', va='center', fontsize=9)
            
            # Save individual file
            safe_name = agent_name.replace(" ", "_").replace("/", "_")
            filename = f"{output_dir}/max_tile_distribution_{safe_name}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            figures.append(fig)
        
        return figures
    
    def plot_max_tile_distributions_combined(self, output_dir=".", figsize=(12, 6)):
        """Plot combined max tile distributions for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        # Get all unique max tiles across all agents
        all_max_tiles = []
        for stats in self.results.values():
            all_max_tiles.extend(stats["max_tiles"])
        unique_tiles = sorted(set(all_max_tiles))
        
        # Create a dictionary of percentages for each agent
        tile_percentages = {}
        for i, (agent_name, stats) in enumerate(self.results.items()):
            percentages = []
            for tile in unique_tiles:
                count = stats["max_tiles"].count(tile)
                percentage = count / len(stats["max_tiles"]) * 100
                percentages.append(percentage)
            tile_percentages[agent_name] = percentages
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(tile_percentages, index=unique_tiles)
        
        # Plot with automatic color cycling
        plt.figure(figsize=figsize)
        ax = df.plot(kind='bar', width=0.8, colormap='Set1')
        
        plt.title('Combined Max Tile Distributions', fontsize=14, fontweight="bold")
        plt.xlabel('Max Tile Value', fontsize=12)
        plt.ylabel('Percentage of Games (%)', fontsize=12)
        plt.legend(title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        # Save file
        filename = f"{output_dir}/max_tile_distributions_combined.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        return plt.gcf()
    
    def plot_game_length_distributions_separate(self, output_dir=".", figsize=(8, 6)):
        """Plot separate game length (moves) distribution for each agent and save individual files."""
        if not self.results:
            print("No results to visualize")
            return []
        
        figures = []
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            fig, ax = plt.subplots(figsize=figsize)
            
            color = self.agent_colors[i % len(self.agent_colors)]
            moves = stats["moves_per_game"]
            sns.histplot(moves, bins=20, alpha=0.7, color=color, ax=ax)
            ax.set_title(f"{agent_name} - Game Length Distribution\nAvg: {stats['avg_moves']:.0f} moves", 
                        fontsize=14, fontweight="bold")
            ax.set_xlabel("Game Length (moves)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Save individual file
            safe_name = agent_name.replace(" ", "_").replace("/", "_")
            filename = f"{output_dir}/game_length_distribution_{safe_name}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            figures.append(fig)
        
        return figures
    
    def plot_game_length_distributions_combined(self, output_dir=".", figsize=(10, 6)):
        """Plot combined game length distributions for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        plt.figure(figsize=figsize)
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            color = self.agent_colors[i % len(self.agent_colors)]
            sns.kdeplot(stats["moves_per_game"], 
                       label=f"{agent_name} (avg: {stats['avg_moves']:.0f})", 
                       color=color, linewidth=2)
        
        plt.title("Combined Game Length Distributions", fontsize=14, fontweight="bold")
        plt.xlabel("Game Length (moves)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save file
        filename = f"{output_dir}/game_length_distributions_combined.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        return plt.gcf()
    
    def plot_win_rates(self, output_dir=".", figsize=(10, 6)):
        """Plot win rates for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        # Prepare data
        agents = []
        win_rates = []
        colors = []
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            agents.append(agent_name)
            win_rates.append(stats["win_rate"])
            colors.append(self.agent_colors[i % len(self.agent_colors)])
        
        # Create figure
        plt.figure(figsize=figsize)
        bars = plt.bar(agents, win_rates, color=colors, alpha=0.8)
        
        plt.title("Win Rate (% games with 2048+ tile)", fontsize=14, fontweight="bold")
        plt.ylabel("Win Rate (%)", fontsize=12)
        plt.xlabel("Agent", fontsize=12)
        plt.ylim(0, max(win_rates) * 1.2 if win_rates else 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add labels on top of bars
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels for better readability if needed
        if len(agents) > 3:
            plt.xticks(rotation=45, ha='right')
        
        # Save file
        filename = f"{output_dir}/win_rates.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        return plt.gcf()
    
    def plot_efficiency(self, output_dir=".", figsize=(10, 6)):
        """Plot efficiency (score per move) for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        # Prepare data
        agents = []
        efficiencies = []
        colors = []
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            agents.append(agent_name)
            efficiencies.append(stats["efficiency"])
            colors.append(self.agent_colors[i % len(self.agent_colors)])
        
        # Create figure
        plt.figure(figsize=figsize)
        bars = plt.bar(agents, efficiencies, color=colors, alpha=0.8)
        
        plt.title("Efficiency (Score per Move)", fontsize=14, fontweight="bold")
        plt.ylabel("Efficiency (score/move)", fontsize=12)
        plt.xlabel("Agent", fontsize=12)
        plt.ylim(0, max(efficiencies) * 1.2 if efficiencies else 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add labels on top of bars
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{eff:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels for better readability if needed
        if len(agents) > 3:
            plt.xticks(rotation=45, ha='right')
        
        # Save file
        filename = f"{output_dir}/efficiency.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        return plt.gcf()
