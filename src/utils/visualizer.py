import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json


class AgentVisualizer:
    """
    Specialized module for creating visualizations of 2048 agent performance.
    Handles all visualization-related tasks.
    """
    
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
        # Set the seaborn style
        sns.set_style("whitegrid")
        
        # Define a custom color palette for 2048 game tiles
        self.tile_colors = {
            2: "#eee4da",
            4: "#ede0c8",
            8: "#f2b179",
            16: "#f59563",
            32: "#f67c5f",
            64: "#f65e3b",
            128: "#edcf72",
            256: "#edcc61",
            512: "#edc850",
            1024: "#edc53f",
            2048: "#edc22e",
            4096: "#3c3a32"
        }
        
        # Create a color palette for agents
        self.agent_colors = sns.color_palette("viridis", 20)
        
        # Define move names for plotting
        self.move_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    
    def load_results(self, filename):
        """Load results from a JSON file."""
        with open(filename, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded visualization data for {len(self.results)} agents from {filename}")
        return self.results
    
    def save_comparison_csv(self, filename="agent_comparison.csv"):
        """Save comparison data to a CSV file."""
        if not self.results:
            print("No results to export")
            return
        
        # Convert to DataFrame
        df = self.prepare_comparison_dataframe()
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Comparison data saved to {filename}")
    
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
                "Avg Move Time (ms)": stats["avg_move_time"] * 1000 if "avg_move_time" in stats else 0,
                "Avg Game Duration (s)": stats["avg_game_duration"],
                "Median Max Tile": stats["median_max_tile"] if "median_max_tile" in stats else np.median(stats["max_tiles"]),
                "Efficiency (score/move)": stats["avg_score"] / stats["avg_moves"] if stats["avg_moves"] > 0 else 0
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_score_distribution(self, figsize=(10, 6)):
        """
        Plot the distribution of scores for each agent.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        if not self.results:
            print("No results to visualize")
            return None
        
        plt.figure(figsize=figsize)
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            color = self.agent_colors[i % len(self.agent_colors)]
            sns.kdeplot(stats["scores"], label=agent_name, color=color)
        
        plt.title("Score Distribution by Agent", fontsize=14, fontweight="bold")
        plt.xlabel("Score", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(title="Agents")
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_win_rates(self, figsize=(10, 6)):
        """
        Plot the win rates for each agent.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        if not self.results:
            print("No results to visualize")
            return None
        
        agents = []
        win_rates = []
        colors = []
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            agents.append(agent_name)
            win_rates.append(stats["win_rate"])
            colors.append(self.agent_colors[i % len(self.agent_colors)])
        
        plt.figure(figsize=figsize)
        bars = plt.bar(agents, win_rates, color=colors)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title("Win Rate by Agent (% games with 2048+ tile)", fontsize=14, fontweight="bold")
        plt.xlabel("Agent", fontsize=12)
        plt.ylabel("Win Rate (%)", fontsize=12)
        plt.ylim(0, max(win_rates) * 1.2 if win_rates else 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_move_distribution(self, figsize=(12, 8)):
        """
        Plot the distribution of moves for each agent.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        if not self.results:
            print("No results to visualize")
            return None
        
        num_agents = len(self.results)
        fig, axes = plt.subplots(1, num_agents, figsize=figsize, sharey=True)
        
        # Handle case of only one agent
        if num_agents == 1:
            axes = [axes]
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            moves = list(stats["move_distribution"].keys())
            counts = list(stats["move_distribution"].values())
            
            # Convert move numbers to direction names
            move_labels = [self.move_names[int(m)] for m in moves]
            
            # Calculate percentages
            total_moves = sum(counts)
            percentages = [count/total_moves*100 for count in counts] if total_moves > 0 else [0] * len(counts)
            
            # Plot
            bars = axes[i].bar(move_labels, percentages, color=self.agent_colors[i % len(self.agent_colors)])
            axes[i].set_title(agent_name)
            axes[i].set_ylim(0, 100)
            axes[i].set_ylabel('Percentage (%)' if i == 0 else '')
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        fig.suptitle('Move Direction Distribution by Agent', fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
    
    def plot_max_tile_distribution(self, figsize=(10, 6)):
        """
        Plot the distribution of maximum tiles achieved.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        if not self.results:
            print("No results to visualize")
            return None
        
        # Get all unique max tiles across all agents
        all_max_tiles = []
        for stats in self.results.values():
            all_max_tiles.extend(stats["max_tiles"])
        
        unique_tiles = sorted(set(all_max_tiles))
        
        # Create a dictionary of counts for each agent
        tile_counts = {}
        for agent_name, stats in self.results.items():
            counts = {}
            for tile in unique_tiles:
                counts[tile] = stats["max_tiles"].count(tile)
            tile_counts[agent_name] = counts
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(tile_counts)
        
        # Calculate percentages
        for col in df.columns:
            total = df[col].sum()
            df[col] = df[col] / total * 100 if total > 0 else 0
        
        # Plot
        plt.figure(figsize=figsize)
        ax = df.plot(kind='bar')
        
        plt.title('Maximum Tile Distribution by Agent', fontsize=14, fontweight="bold")
        plt.xlabel('Max Tile Value', fontsize=12)
        plt.ylabel('Percentage of Games (%)', fontsize=12)
        plt.legend(title='Agent')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis labels to show actual tile values
        plt.xticks(range(len(unique_tiles)), unique_tiles)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_performance_radar(self, metrics=None, figsize=(8, 8)):
        """
        Plot a radar chart of performance metrics for each agent.
        
        Args:
            metrics: List of metrics to include (defaults to a standard set)
            figsize: Figure size as (width, height) tuple
        """
        if not self.results:
            print("No results to visualize")
            return None
        
        if metrics is None:
            metrics = [
                'Avg Score', 
                'Win Rate (%)', 
                'Avg Moves', 
                'Efficiency (score/move)',
                'Median Max Tile'
            ]
        
        # Get comparison data
        df = self.prepare_comparison_dataframe()
        
        # Filter to only the requested metrics
        df_metrics = df[['Agent'] + metrics].set_index('Agent')
        
        # Normalize each metric to 0-1 scale
        df_norm = df_metrics.copy()
        for col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:  # Avoid division by zero
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 1.0
        
        # Number of metrics
        N = len(metrics)
        
        # Create angles for the radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Add each agent
        for i, agent in enumerate(df_norm.index):
            values = df_norm.loc[agent].values.tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            color = self.agent_colors[i % len(self.agent_colors)]
            ax.plot(angles, values, linewidth=2, label=agent, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Agent Performance Comparison', fontsize=14, fontweight="bold")
        return fig
    
    def plot_time_performance(self, figsize=(10, 6)):
        """
        Plot time-related performance metrics.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        if not self.results:
            print("No results to visualize")
            return None
        
        # Prepare data
        agents = []
        move_times = []
        game_times = []
        colors = []
        
        for i, (agent_name, stats) in enumerate(self.results.items()):
            agents.append(agent_name)
            move_times.append(stats["avg_move_time"] * 1000)  # Convert to ms
            game_times.append(stats["avg_game_duration"])
            colors.append(self.agent_colors[i % len(self.agent_colors)])
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot move times
        x = np.arange(len(agents))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, move_times, width, label="Avg Move Time (ms)", 
                       color=[self.lighten_color(c) for c in colors])
        ax1.set_ylabel("Avg Move Time (ms)", fontsize=12)
        ax1.tick_params(axis="y")
        
        # Create second y-axis for game duration
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, game_times, width, label="Avg Game Duration (s)",
                       color=colors)
        ax2.set_ylabel("Avg Game Duration (s)", fontsize=12)
        ax2.tick_params(axis="y")
        
        # Add labels and legend
        ax1.set_title("Time Performance Metrics", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, rotation=45, ha="right")
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        # Add value labels
        for i, v in enumerate(move_times):
            ax1.text(i - width/2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=9, rotation=45)
        
        for i, v in enumerate(game_times):
            ax2.text(i + width/2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=9, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_dashboard(self, figsize=(15, 12), dpi=100, save_path=None, show=True):
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            figsize: Figure size as (width, height) tuple
            dpi: Resolution of the figure
            save_path: Optional path to save the figure
            show: Whether to display the figure
            
        Returns:
            The matplotlib figure object
        """
        if not self.results:
            print("No results to visualize")
            return None
        
        # Create figure and grid
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # Plot score distribution
        ax1 = fig.add_subplot(gs[0, 0:2])
        for i, (agent_name, stats) in enumerate(self.results.items()):
            color = self.agent_colors[i % len(self.agent_colors)]
            sns.kdeplot(stats["scores"], label=agent_name, color=color, ax=ax1)
        ax1.set_title("Score Distribution", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Score")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot win rates
        ax2 = fig.add_subplot(gs[0, 2])
        agents = []
        win_rates = []
        colors = []
        for i, (agent_name, stats) in enumerate(self.results.items()):
            agents.append(agent_name)
            win_rates.append(stats["win_rate"])
            colors.append(self.agent_colors[i % len(self.agent_colors)])
        bars = ax2.bar(agents, win_rates, color=colors)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        ax2.set_title("Win Rate (%)", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, max(win_rates) * 1.2 if win_rates else 100)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot max tile distribution
        ax3 = fig.add_subplot(gs[1, :])
        all_max_tiles = []
        for stats in self.results.values():
            all_max_tiles.extend(stats["max_tiles"])
        unique_tiles = sorted(set(all_max_tiles))
        tile_counts = {}
        for agent_name, stats in self.results.items():
            counts = {}
            for tile in unique_tiles:
                counts[tile] = stats["max_tiles"].count(tile)
            tile_counts[agent_name] = counts
        df = pd.DataFrame(tile_counts)
        for col in df.columns:
            total = df[col].sum()
            df[col] = df[col] / total * 100 if total > 0 else 0
        df.plot(kind='bar', ax=ax3)
        ax3.set_title("Maximum Tile Distribution", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Max Tile Value")
        ax3.set_ylabel("Percentage of Games (%)")
        ax3.set_xticklabels(unique_tiles)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot move distribution
        ax4 = fig.add_subplot(gs[2, 0])
        data = []
        for agent_name, stats in self.results.items():
            total_moves = sum(stats["move_distribution"].values())
            if total_moves == 0:
                continue
            for move, count in stats["move_distribution"].items():
                percentage = (count / total_moves) * 100
                data.append({
                    'Agent': agent_name,
                    'Direction': self.move_names[int(move)],
                    'Percentage': percentage
                })
        df_moves = pd.DataFrame(data)
        if not df_moves.empty:
            sns.barplot(x='Direction', y='Percentage', hue='Agent', data=df_moves, ax=ax4)
        ax4.set_title("Move Direction Preference", fontsize=12, fontweight="bold")
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot performance metrics table
        ax5 = fig.add_subplot(gs[2, 1:])
        df_metrics = self.prepare_comparison_dataframe()
        ax5.axis('tight')
        ax5.axis('off')
        table = ax5.table(
            cellText=df_metrics.round(2).values,
            colLabels=df_metrics.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax5.set_title("Performance Metrics", fontsize=12, fontweight="bold")
        
        # Add overall title
        plt.suptitle("2048 AI Agent Performance Comparison", fontsize=16, fontweight="bold")
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig
    
    def lighten_color(self, color, amount=0.5):
        """
        Lighten a color by the given amount.
        
        Args:
            color: Color to lighten
            amount: Amount to lighten (0-1)
        """
        import matplotlib.colors as mc
        import colorsys
        
        try:
            c = mc.to_rgb(color)
            c = colorsys.rgb_to_hls(*c)
            return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        except:
            return color  # Return original if conversion fails
