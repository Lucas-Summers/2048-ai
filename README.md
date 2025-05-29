# 2048 AI Project

A comprehensive implementation of various AI algorithms to play the 2048 game, featuring multiple agents, a web interface, and performance analysis tools.

## Overview

This project implements and compares different AI approaches for playing the 2048 game:

- **Random Agent**: Makes random valid moves (baseline for comparison)
- **Greedy Agent**: Makes moves based on immediate score maximization with heuristic evaluation
- **Monte Carlo Tree Search (MCTS)**: Uses tree search with random or expectimax rollouts
- **Expectimax**: Uses expectimax algorithm with progressive deepening and complete search
- **Reinforcement Learning (RL)**: Deep Q-Network (DQN) agent trained with experience replay

The project features:
- **Interactive web interface** for playing and watching AI agents
- **Simplified, clean implementations** focused on core algorithms
- **Standardized statistics** across all agents for fair comparison
- **Performance analysis tools** with statistical visualization
- **Deep learning framework** with PyTorch for RL agent
- **Comprehensive heuristics system** for game state evaluation

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
   - View real-time statistics and performance metrics
   - Compare agent performance across multiple games

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

## AI Agents Details

### Random Agent
- **Strategy**: Selects random valid moves
- **Features**:
  - Provides baseline performance for comparison
  - No computational overhead
  - Uniform move distribution
- **Parameters**: 
  - `seed` (int, optional): Random seed for reproducibility
- **Use Case**: Performance baseline and testing

### Greedy Agent
- **Strategy**: Maximizes immediate score gain combined with tile value improvement
- **Features**:
  - Single-step lookahead with composite evaluation
  - Fast decision making 
  - Combines tile progression and score optimization
- **Parameters**:
  - `thinking_time` (float): Time limit per move in seconds (default: 0.5)
  - `tile_weight` (float): Weight for tile value scoring (default: 1.0)
  - `score_weight` (float): Weight for game score (default: 0.1)
- **Use Case**: Fast baseline with decent performance

### Monte Carlo Tree Search (MCTS)
- **Strategy**: Explores move tree using UCB1 selection with configurable rollout strategies
- **Features**:
  - Tree search with UCB1 selection
  - Two rollout types: random simulation or expectimax evaluation
  - Complete move exploration (no pruning)
  - Time-based search with iteration limits
- **Parameters**:
  - `thinking_time` (float): Time limit per move in seconds (default: 0.5)
  - `exploration_weight` (float): UCB1 exploration parameter (default: 1.414)
  - `rollout_type` (str): "random" or "expectimax" rollouts (default: "expectimax")
- **Use Case**: Balanced exploration and exploitation with strong performance

### Expectimax Agent
- **Strategy**: Uses expectimax algorithm with progressive deepening and complete search
- **Features**:
  - Progressive deepening within time budget
  - Complete search space exploration (no pruning)
  - Considers tile placement probabilities (90% chance of 2, 10% chance of 4)
  - Composite heuristic evaluation with multiple criteria
- **Parameters**:
  - `thinking_time` (float): Time limit per move in seconds (default: 0.5)
- **Use Case**: Strong deterministic performance with complete game-tree analysis

### Reinforcement Learning (DQN) Agent
- **Strategy**: Learns optimal policy through experience using Deep Q-Networks
- **Features**:
  - Neural network with 2 hidden layers (128 neurons each)
  - Experience replay buffer for stable learning
  - Target network for training stability
  - Epsilon-greedy exploration strategy
  - GPU acceleration (CUDA/MPS support)
  - Model saving and loading capabilities
- **Parameters**:
  - `lr` (float): Learning rate (default: 1e-3)
  - `gamma` (float): Discount factor (default: 0.99)
  - `epsilon_start` (float): Initial exploration rate (default: 1.0)
  - `epsilon_end` (float): Final exploration rate (default: 0.05)
  - `epsilon_decay_steps` (int): Steps for exploration decay (default: 50,000)
  - `batch_size` (int): Training batch size (default: 128)
  - `buffer_capacity` (int): Experience replay buffer size (default: 50,000)
  - `target_update_interval` (int): Target network update frequency (default: 1,000)
  - `training` (bool): Enable/disable training mode (default: False)
- **Use Case**: Learns optimal strategy through self-play, strongest performance when properly trained

## Agent Statistics

All agents provide standardized statistics for consistent comparison:

- **`search_iterations`**: Number of evaluations/iterations performed per move
- **`max_depth_reached`**: Maximum search depth explored during the move
- **`avg_reward`**: Average evaluation score/reward for the move

### Statistics by Agent Type:

- **Random**: search_iterations=1, max_depth_reached=0, avg_reward=0.0
- **Greedy**: search_iterations=2-4 (moves evaluated), max_depth_reached=1, avg_reward=evaluation score
- **MCTS**: search_iterations=iterations performed, max_depth_reached=tree depth, avg_reward=simulation reward
- **Expectimax**: search_iterations=move evaluations, max_depth_reached=search depth, avg_reward=heuristic value
- **RL**: search_iterations=0-1 (predictions), max_depth_reached=1, avg_reward=Q-value

## Heuristics System

The project implements a comprehensive heuristics system used by MCTS and Expectimax agents for board evaluation:

### Composite Heuristic Weights
All agents using heuristics employ the same standardized weights for fair comparison:

```python
# Standard heuristic configuration
heuristics = [
    (EmptyTilesHeuristic(), 3.0),      # Available space
    (MonotonicityHeuristic(), 1.5),    # Board ordering  
    (CornerMaxHeuristic(), 5.0),       # Corner placement
    (SmoothnessHeuristic(), 2.0),      # Adjacent similarity
    (MergePotentialHeuristic(), 2.5),  # Merge opportunities
    (MaxValueHeuristic(), 3.0)         # Tile progression
]
```

### Heuristic Descriptions:

- **Empty Tiles**: Counts available spaces (0-16, higher is better)
- **Monotonicity**: Measures board ordering (0-16, higher is better) 
- **Corner Max**: Rewards corner placement of max tile (variable, corners preferred)
- **Smoothness**: Evaluates adjacent tile similarity (negative values, closer to 0 is better)
- **Merge Potential**: Identifies immediate merge opportunities (0+, higher is better)
- **Max Value**: Rewards tile progression (log2 values, higher is better)

## Project Structure

```
2048-ai/
├── src/
│   ├── ai/                    # AI agent implementations
│   │   ├── expectimax.py      # Simplified expectimax agent
│   │   ├── mcts.py           # Simplified MCTS agent
│   │   ├── greedy.py         # Greedy agent
│   │   ├── random.py         # Random agent
│   │   ├── rl.py             # Deep Q-Network agent
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