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
EVAL_GAMES: int = 20

AGENT_KWARGS = dict(
    training=True,
    lr=1e-4,
    gamma=0.99,
    batch_size=1024,
    buffer_capacity=500_000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=500_000,
    target_update_interval=5_000,
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


def play_episode(agent: RLAgent, max_moves: int = MAX_MOVES) -> Tuple[float, int]:
    """Play a single training episode and return (reward, moves)."""
    game = Game2048()
    total_reward = 0.0
    moves = 0
    for _ in range(max_moves):
        if game.is_game_over():
            break

        prev_empty_cells = game.board.get_empty_cells_count()
        prev_max_tile = game.board.get_max_tile()

        action = agent.get_move(game)
        info = game.step(action)

        score_reward = np.log2(info["score"] + 1) / 16.0
        empty_cells_reward = (game.board.get_empty_cells_count() - prev_empty_cells) * 0.1
        merge_reward = 0.5 if game.board.get_max_tile() > prev_max_tile else 0
        game_over_penalty = -2.0 if info["game_over"] else 0

        reward = score_reward + empty_cells_reward + merge_reward + game_over_penalty

        done = info["game_over"]
        agent.remember_last(reward, game.board.grid, done)
        total_reward += reward
        moves += 1
        if done:
            break
    agent.end_episode()
    return total_reward, moves


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

    lr_scheduler = LRManager(
        initial_lr=AGENT_KWARGS['lr'],
        min_lr=5e-6,
        decay_factor=0.5,
        patience=3  # num evaluations w/o improvement
    )

    for episode in range(1, TOTAL_EPISODES + 1):
        # Play one episode
        total_reward, moves = play_episode(agent, MAX_MOVES)
        recent_rewards.append(total_reward)

        # Log of training progress
        if episode % 100 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            epsilon = agent._current_epsilon()
            print(f"Episode {episode}: reward = {total_reward:.1f}, avg = {avg_reward:.2f}, "
                  f"moves = {moves}, epsilon = {epsilon:.3f}")

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
