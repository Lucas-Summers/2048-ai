import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

class AgentVisualizer:
    def __init__(self, analyzer_results=None):
        self.setup_style()
        self.results = {}
        if analyzer_results:
            if isinstance(analyzer_results, dict):
                self.results = analyzer_results
            elif isinstance(analyzer_results, str):
                self.load_results(analyzer_results)
    
    def setup_style(self):
        """Configure plot styling for consistency and improved aesthetics."""
        sns.set_style("whitegrid")
        
        tile_colors = [
            '#f65e3b',  # RED (tile-64)
            '#edc22e',  # YELLOW (tile-2048)
            '#f59563',  # ORANGE (tile-16)
            '#7fb069',  # GREEN
            '#5b9bd5',  # BLUE
            '#8e6a8b',  # PURPLE
            '#776e65',  # BLACK (text)
            '#ede0c8',  # WHITE (tile-4)
        ]
        self.agent_colors = tile_colors
        self.move_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        self.bg_color = '#faf8ef'
        self.text_color = '#776e65'
        self.highlight_color = '#bbada0'
        
        self.standard_figsize = (12, 6)
    
    def load_results(self, filename):
        """Load results from a JSON file."""
        with open(filename, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded visualization data for {len(self.results)} agents from {filename}")
        return self.results
    
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
                "Absolute Max Tile": stats.get("absolute_max_tile", 0),
                "Efficiency (score/move)": stats["efficiency"]
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_score_distributions(self, output_dir=".", figsize=None):
        """Plot score distributions for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        if figsize is None:
            figsize = self.standard_figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        for i, (agent_name, stats) in enumerate(self.results.items()):
            color = self.agent_colors[i % len(self.agent_colors)]
            sns.kdeplot(stats["scores"], label=f"{agent_name} (avg: {stats['avg_score']:.0f})", 
                       color=color, linewidth=2, ax=ax)
        
        ax.set_title("Score Distributions (Log Scale)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Score (Log Scale)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xscale('log')  # Use logarithmic scale for scores
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.apply_game_theme(fig, ax)
        plt.tight_layout()
        
        filename = f"{output_dir}/score_distributions.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor=self.bg_color)
        print(f"Saved: {filename}")
        
        return fig
    
    def plot_max_tile_distributions(self, output_dir=".", figsize=None, max_tiles_to_show=10):
        """Plot max tile distributions for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        if figsize is None:
            figsize = self.standard_figsize
        
        all_max_tiles = []
        for stats in self.results.values():
            all_max_tiles.extend(stats["max_tiles"])
        
        unique_tiles = sorted(set(all_max_tiles), reverse=True)
        highest_tiles = unique_tiles[:max_tiles_to_show]
        selected_tiles = sorted(highest_tiles)
        
        tile_percentages = {}
        for i, (agent_name, stats) in enumerate(self.results.items()):
            percentages = []
            for tile in selected_tiles:
                count = stats["max_tiles"].count(tile)
                percentage = count / len(stats["max_tiles"]) * 100
                percentages.append(percentage)
            tile_percentages[agent_name] = percentages
        
        df = pd.DataFrame(tile_percentages, index=selected_tiles)
        fig, ax = plt.subplots(figsize=figsize)
        
        agent_names = list(self.results.keys())
        colors_map = {}
        for i, agent_name in enumerate(agent_names):
            colors_map[agent_name] = self.agent_colors[i % len(self.agent_colors)]
        df.plot(kind='bar', width=0.8, ax=ax, alpha=0.9, color=colors_map, edgecolor='none')
        
        ax.set_title(f'Max Tile Distributions (Highest {len(selected_tiles)} Tiles)', fontsize=14, fontweight="bold")
        ax.set_xlabel('Max Tile Value', fontsize=12)
        ax.set_ylabel('Percentage of Games (%)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        self.apply_game_theme(fig, ax)
        plt.tight_layout()
        
        filename = f"{output_dir}/max_tile_distributions.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor=self.bg_color)
        print(f"Saved: {filename}")
        
        return fig
    
    def plot_move_distributions(self, output_dir=".", figsize=None):
        """Plot move distributions for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        if figsize is None:
            figsize = self.standard_figsize
        
        move_data = {}
        move_names = ["Up", "Right", "Down", "Left"]
        for agent_name, stats in self.results.items():
            if "move_distribution" in stats:
                move_counts = [
                    stats["move_distribution"].get("0", 0),
                    stats["move_distribution"].get("1", 0),
                    stats["move_distribution"].get("2", 0),
                    stats["move_distribution"].get("3", 0),
                ]
                total_moves = sum(move_counts)
                if total_moves > 0:
                    move_percentages = [count/total_moves * 100 for count in move_counts]
                else:
                    move_percentages = [0, 0, 0, 0]
                
                move_data[agent_name] = move_percentages
        
        if not move_data:
            print("No move distribution data available for visualization")
            return None
        
        df = pd.DataFrame(move_data, index=move_names)
        fig, ax = plt.subplots(figsize=figsize)
        
        agent_names = list(move_data.keys())
        colors_map = {}
        for i, agent_name in enumerate(agent_names):
            colors_map[agent_name] = self.agent_colors[i % len(self.agent_colors)]
        
        df.plot(kind='bar', width=0.8, ax=ax, alpha=0.9, color=colors_map, edgecolor='none')
        ax.set_title('Move Distributions', fontsize=14, fontweight="bold")
        ax.set_xlabel('Move Direction', fontsize=12)
        ax.set_ylabel('Percentage of Moves (%)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=0)  # Keep move names horizontal
        
        self.apply_game_theme(fig, ax)
        plt.tight_layout()
        
        filename = f"{output_dir}/move_distributions.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor=self.bg_color)
        print(f"Saved: {filename}")
        
        return fig
    
    def plot_game_length_distributions(self, output_dir=".", figsize=None):
        """Plot game length distributions for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        if figsize is None:
            figsize = self.standard_figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            color = self.agent_colors[i % len(self.agent_colors)]
            sns.kdeplot(stats["moves_per_game"], 
                       label=f"{agent_name} (avg: {stats['avg_moves']:.0f})", 
                       color=color, linewidth=2, ax=ax)
        
        ax.set_title("Game Length Distributions (Log Scale)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Game Length (moves, Log Scale)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xscale('log')  # Use logarithmic scale for game lengths
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.apply_game_theme(fig, ax)
        plt.tight_layout()
        
        filename = f"{output_dir}/game_length_distributions.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor=self.bg_color)
        print(f"Saved: {filename}")
        
        return fig
    
    def plot_win_rates(self, output_dir=".", figsize=None):
        """Plot win rates for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        if figsize is None:
            figsize = self.standard_figsize
        
        agents = []
        win_rates = []
        colors = []
        for i, (agent_name, stats) in enumerate(self.results.items()):
            agents.append(agent_name)
            win_rates.append(stats["win_rate"])
            colors.append(self.agent_colors[i % len(self.agent_colors)])
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(agents, win_rates, color=colors, alpha=0.9, edgecolor='none')
        
        ax.set_title("Win Rate (% games with 2048+ tile)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Win Rate (%)", fontsize=12)
        ax.set_xlabel("Agent", fontsize=12)
        ax.set_ylim(0, max(win_rates) * 1.2 if win_rates else 100)
        
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, color=self.text_color)
        
        if len(agents) > 3 or any(len(agent) > 15 for agent in agents):
            ax.set_xticklabels(agents, rotation=45, ha='right')
        
        self.apply_game_theme(fig, ax)
        plt.tight_layout()
        
        filename = f"{output_dir}/win_rates.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor=self.bg_color)
        print(f"Saved: {filename}")
        
        return fig
    
    def plot_efficiency(self, output_dir=".", figsize=None):
        """Plot efficiency (score per move) for all agents."""
        if not self.results:
            print("No results to visualize")
            return None
        
        if figsize is None:
            figsize = self.standard_figsize
        
        agents = []
        efficiencies = []
        colors = []
        for i, (agent_name, stats) in enumerate(self.results.items()):
            agents.append(agent_name)
            efficiencies.append(stats["efficiency"])
            colors.append(self.agent_colors[i % len(self.agent_colors)])
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(agents, efficiencies, color=colors, alpha=0.9, edgecolor='none')
        ax.set_title("Efficiency (Score per Move)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Efficiency (score/move)", fontsize=12)
        ax.set_xlabel("Agent", fontsize=12)
        ax.set_ylim(0, max(efficiencies) * 1.2 if efficiencies else 1)
        
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   f'{eff:.2f}', ha='center', va='bottom', fontsize=10, color=self.text_color)
        
        if len(agents) > 3 or any(len(agent) > 15 for agent in agents):
            ax.set_xticklabels(agents, rotation=45, ha='right')
        
        self.apply_game_theme(fig, ax)
        plt.tight_layout()
        
        filename = f"{output_dir}/efficiency.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor=self.bg_color)
        print(f"Saved: {filename}")
        
        return fig
    
    def apply_game_theme(self, fig, ax):
        """Apply 2048 game theme to a matplotlib figure and axis."""
        # Set figure and axis background colors
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        # Set text colors
        ax.title.set_color(self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.tick_params(colors=self.text_color)
        
        # Set grid color
        ax.grid(True, alpha=0.3, color=self.highlight_color)
        
        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(self.highlight_color)
            spine.set_alpha(0.7)
        
        # Set legend text color if legend exists
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(self.bg_color)
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_edgecolor(self.highlight_color)
            for text in legend.get_texts():
                text.set_color(self.text_color)
