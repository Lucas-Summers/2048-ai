# 2048 AI

A comprehensive implementation of various AI algorithms to play the 2048 game, featuring multiple agents, a web interface, and advanced performance analysis tools with parallel processing capabilities.

## Overview

This project implements and compares different AI approaches for playing the 2048 game:

- **Random Agent**: Makes random valid moves (baseline for comparison)
- **Greedy Agent**: Single-step lookahead with configurable tile value and score optimization weights
- **Monte Carlo Tree Search (MCTS)**: Advanced tree search with UCB1 selection and multiple rollout strategies:
  - Random rollouts for fast exploration
  - Expectimax rollouts for strategic depth
  - Reinforcement Learning rollouts using trained models
- **Expectimax**: Progressive deepening expectimax with quality loss minimization and enhanced heuristics
- **Reinforcement Learning (RL)**: Deep Q-Network (DQN) agent with epsilon-greedy exploration, experience replay, and adaptive learning rate scheduling
- **Hybrid Agents**: MCTS agents with specialized rollout strategies combining different AI approaches

## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 2048-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Web Interface

1. **Start the Flask server**:
   ```bash
   python src/app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8080`

3. **Interactive features**:
   - Play manually using arrow keys
   - Select AI agents from the dropdown menu to watch them play
   - Adjust AI speed settings (full/fast/medium/slow) with smooth animations
   - View real-time agent statistics including efficiency, game duration, and algorithm-specific metrics
   - Monitor highest tiles achieved with animated tracking
   - Game over modal with detailed statistics

#### Training the RL Agent

Train a new reinforcement learning model with advanced features:

```bash
python src/utils/trainRL.py
```

Features include:
- Adaptive learning rate scheduling based on performance plateaus
- Enhanced training with 50,000 episodes by default
- Model checkpointing and best model selection
- Training progress visualization and statistics
- Automatic model saving in timestamped directories

#### Performance Analysis and Simulation

Run comprehensive agent comparison with parallel processing:

```bash
python src/utils/simulate.py
```

Advanced simulation features:
- **Parallel processing**: Multi-core agent testing for faster results
- **Configurable parameters**: Customizable thinking time, number of games, and batch sizes
- **Comprehensive metrics**: Win rates, score distributions, efficiency analysis
- **Data persistence**: Save and load results to avoid re-running simulations
- **Statistical visualization**: Automated generation of comparison charts
- **Command-line interface**: Full argument parsing for flexible configuration

Example usage:
```bash
# Run 500 games per agent with 1.0s thinking time
python src/utils/simulate.py -n 500 -t 1.0

# Use 8 processes with custom batch settings
python src/utils/simulate.py --processes 8 --batch_size 20

# Skip simulation if results exist
python src/utils/simulate.py --skip_existing
```

#### Additional Utilities

- **`src/utils/test.py`**: Individual agent testing and debugging
- **`src/utils/analyzer.py`**: Advanced statistical analysis with parallel processing
- **`src/utils/visualizer.py`**: Comprehensive data visualization and chart generation


## License

This project is licensed under the terms specified in the LICENSE file.