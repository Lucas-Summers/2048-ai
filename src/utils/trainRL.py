from __future__ import annotations
import statistics
import time
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

# Add parent directory to path so we can import from ai and game modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from ai.rl import RLAgent
from game.game import Game2048

# hyperparameters for RL training
TOTAL_EPISODES: int = 50_000
MAX_MOVES: int = 5_000
EVAL_INTERVAL: int = 500
EVAL_GAMES: int = 100

AGENT_KWARGS = dict(
    training=True,
    lr=1e-4,
    gamma=0.99,
    batch_size=256,
    buffer_capacity=100_000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=100_000,
    target_update_interval=1_000,
)

# store best models in "runs" directory
RUN_DIR = Path("runs") / time.strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)


class LRManager:
    def __init__(self, initial_lr=1e-4, min_lr=5e-6, decay_factor=0.5, patience=2000):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.best_score = -float('inf')
        self.wait = 0
        self.current_lr = initial_lr

    def update(self, score, agent):
        if score > self.best_score:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                agent.update_learning_rate(self.current_lr)
                print(f"Reduced learning rate to {self.current_lr}")
                self.wait = 0
        return self.current_lr

def snake_indices():
    # Returns the indices for a 4x4 snake pattern
    return [
        (0,0), (0,1), (0,2), (0,3),
        (1,3), (1,2), (1,1), (1,0),
        (2,0), (2,1), (2,2), (2,3),
        (3,3), (3,2), (3,1), (3,0)
    ]

def snake_order_reward(grid):
    # Reward based on snake pattern (optimal for 2048)
    indices = snake_indices()
    values = [grid[i] for i in indices]
    reward = 0
    for i in range(len(values)-1):
        if values[i] >= values[i+1]:
            reward += 1
    return reward / (len(values)-1)  # Normalize to [0,1]

def monotonicity_reward(grid):
    reward = 0
    for row in grid:
        if all(row[i] >= row[i+1] for i in range(3)) or all(row[i] <= row[i+1] for i in range(3)):
            reward += 1
        else:
            reward -= 1  # Penalize non-monotonic rows and columns
    for col in grid.T:
        if all(col[i] >= col[i+1] for i in range(3)) or all(col[i] <= col[i+1] for i in range(3)):
            reward += 1
        else:
            reward -= 1
    return reward / 8  # Normalize

def corner_max_reward(grid):
    # Reward if the max tile is in a corner
    max_tile = np.max(grid)
    corners = [grid[0,0], grid[0,3], grid[3,0], grid[3,3]]
    return 1.0 if max_tile in corners else 0.0

def merge_potential_reward(grid):
    # Reward for setting up potential merges
    reward = 0
    for i in range(4):
        for j in range(3):
            if grid[i, j] == grid[i, j+1] and grid[i, j] != 0:
                reward += 1
            if grid[j, i] == grid[j+1, i] and grid[j, i] != 0:
                reward += 1
    return reward / 12  # Normalize (max 12 possible merges in a 4x4 grid)

def smoothness_reward(grid):
    # Reward for having similar values in adjacent cells
    reward = 0
    for i in range(4):
        for j in range(3):
            reward -= abs(grid[i, j] - grid[i, j+1])
            reward -= abs(grid[j, i] - grid[j+1, i])
    return -reward / 128  # Normalize and make positive

def play_episode(agent: RLAgent, max_moves: int = MAX_MOVES) -> tuple[float, int, float, int]:
    """Play a single training episode and return (reward, moves, avg_loss, max_tile)."""
    game = Game2048()
    total_reward = 0.0
    moves = 0
    prev_score = 0
    prev_empty_cells = game.board.get_empty_cells_count()
    prev_max_tile = game.board.get_max_tile()
    losses = []
    max_tile = prev_max_tile
    for _ in range(max_moves):
        if game.is_game_over():
            break
        action = agent.get_move(game)
        info = game.step(action)

        score_delta = info["score"] - prev_score
        empty_delta = game.board.get_empty_cells_count() - prev_empty_cells
        max_tile_increase = int(game.board.get_max_tile() > prev_max_tile)
        snake_reward = snake_order_reward(game.board.grid)
        mono_reward = monotonicity_reward(game.board.grid)
        corner_reward = corner_max_reward(game.board.grid)
        merge_reward = merge_potential_reward(game.board.grid)
        smooth_reward = smoothness_reward(game.board.grid)
        reward = (
            score_delta
            + 0.5 * empty_delta
            + 10.0 * max_tile_increase
            + 5.0 * snake_reward
            + 2.0 * mono_reward
            + 3.0 * corner_reward
            + 1.0 * merge_reward
            + 1.0 * smooth_reward
        )
        prev_score = info["score"]
        prev_empty_cells = game.board.get_empty_cells_count()
        prev_max_tile = game.board.get_max_tile()
        max_tile = max(max_tile, prev_max_tile)
        done = info["game_over"]
        if done:
            reward -= 100.0  # Penalty for losing
        loss = agent.remember_last(reward, game.board.grid, done)
        if loss is not None:
            losses.append(loss)
        total_reward += reward
        moves += 1
        if done:
            break
    agent.end_episode()
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return total_reward, moves, avg_loss, max_tile


def evaluate(agent: RLAgent, games: int = EVAL_GAMES) -> float:
    """Run greedy games and return average score."""
    prev_mode = agent.training
    agent.training = False
    scores = []
    for _ in range(games):
        g = Game2048()
        while not g.is_game_over():
            a = agent.get_move(g)
            g.step(a)
        scores.append(g.score)
    agent.training = prev_mode
    return statistics.mean(scores)

if __name__ == "__main__":
    agent = RLAgent(**AGENT_KWARGS)
    best_eval_score = 0.0
    recent_rewards: Deque[float] = deque(maxlen=100)
    recent_losses: Deque[float] = deque(maxlen=100)
    recent_max_tiles: Deque[int] = deque(maxlen=100)
    learning_started = False

    lr_scheduler = LRManager(
        initial_lr=AGENT_KWARGS['lr'],
        min_lr=5e-6,
        decay_factor=0.5,
        patience=3  # num evaluations w/o improvement
    )

    for episode in range(1, TOTAL_EPISODES + 1):
        # Play one episode
        total_reward, moves, avg_loss, max_tile = play_episode(agent, MAX_MOVES)
        recent_rewards.append(total_reward)
        recent_max_tiles.append(max_tile)
        if avg_loss > 0:
            recent_losses.append(avg_loss)

        # Print when learning starts
        if not learning_started and len(agent.replay_buffer) >= agent.batch_size:
            print(f"Learning started at episode {episode} (replay buffer filled)")
            learning_started = True

        # Log of training progress
        if episode % 100 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_loss_val = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            avg_max_tile = sum(recent_max_tiles) / len(recent_max_tiles) if recent_max_tiles else 0.0
            epsilon = agent._current_epsilon()
            print(f"Episode {episode}: reward = {total_reward:.1f}, avg = {avg_reward:.2f}, "
                  f"moves = {moves}, epsilon = {epsilon:.3f}, avg_loss = {avg_loss_val:.5f}, avg_max_tile = {avg_max_tile:.2f}")

        # Evaluate periodically
        if episode % EVAL_INTERVAL == 0:
            avg_score = evaluate(agent, EVAL_GAMES)
            print(f"[Eval] Episode {episode}: average greedy score = {avg_score:.0f}")

            # Update learning rate based on evaluation performance
            current_lr = lr_scheduler.update(avg_score, agent)

            if avg_score > best_eval_score:
                best_eval_score = avg_score
                ckpt = RUN_DIR / f"best_{int(best_eval_score)}.pt"
                agent.save_model(ckpt.as_posix())
                print(f"  New best score! Model saved to {ckpt}")
