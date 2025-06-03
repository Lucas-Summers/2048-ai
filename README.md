# 2048 AI

A comprehensive implementation of various AI algorithms to play the 2048 game, featuring multiple agents, a web interface, and performance analysis tools.

## Overview

This project implements and compares different AI approaches for playing the 2048 game:

- **Random Agent**: Makes random valid moves with directional distribution tracking (baseline for comparison)
- **Greedy Agent**: Single-step lookahead with tile value and score optimization using heuristic evaluation
- **Monte Carlo Tree Search (MCTS)**: Uses tree search with UCB1 selection and configurable rollout strategies (random, expectimax, or RL)
- **Expectimax**: Progressive deepening expectimax with quality loss minimization and strategic tile placement
- **Reinforcement Learning (RL)**: Deep Q-Network (DQN) agent with epsilon-greedy exploration and experience replay

The project features:
- **Interactive web interface** for playing and watching AI agents
- **Standardized performance metrics** across all agents (efficiency, game duration, and agent-specific insights)
- **Real-time statistics tracking** including decision quality, search depth, and behavioral patterns
- **Performance analysis tools** with statistical visualization and comparison charts
- **Deep learning framework** with PyTorch for RL agent training and inference
- **Comprehensive heuristics system** for position evaluation and move quality assessment

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

3. **Play the game**:
   - Use arrow keys or on-screen buttons to make moves
   - Select different AI agents from the dropdown to watch them play
   - View real-time standardized statistics (efficiency, game duration) and agent-specific metrics
   - Monitor decision quality, search performance, and behavioral patterns
   - Compare agent performance across multiple games with consistent metrics

#### Training the RL Agent

Train a new reinforcement learning model:

```bash
python src/utils/trainRL.py
```

The training script will:
- Train for 20,000 episodes by default
- Save the best model based on evaluation performance
- Store models in the `runs/` directory with timestamps
- Display training progress and statistics

#### Performance Analysis

Run comprehensive agent analysis:

```bash
python src/utils/run.py
```

This will:
- Test all agents across multiple games
- Generate performance visualizations
- Save results and charts to the `results/` directory
- Create statistical comparison reports

## Project Structure

```
2048-ai/
├── src/
│   ├── ai/                    # AI agent implementations
│   │   ├── expectimax.py      # Progressive deepening expectimax with quality loss analysis
│   │   ├── mcts.py           # MCTS with UCB1 selection and multiple rollout strategies
│   │   ├── greedy.py         # Single-step heuristic evaluation agent
│   │   ├── random.py         # Random baseline with directional distribution tracking
│   │   ├── rl.py             # Deep Q-Network with epsilon-greedy exploration
│   │   └── heuristics/       # Heuristic evaluation functions
│   ├── game/                 # Game logic implementation
│   ├── utils/                # Utility scripts
│   │   ├── run.py           # Agent comparison and testing
│   │   ├── trainRL.py       # RL agent training
│   │   ├── analyzer.py      # Performance analysis
│   │   └── visualizer.py    # Data visualization
│   └── app.py               # Web interface
├── templates/               # HTML templates
├── static/                 # CSS/JS assets
└── requirements.txt        # Dependencies
```

## License

This project is licensed under the terms specified in the LICENSE file.