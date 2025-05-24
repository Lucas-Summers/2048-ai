from __future__ import annotations
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Deque, Tuple
import numpy as np
from ai.rl import RLAgent
from game.game import Game2048

# hyperparameters for RL training
TOTAL_EPISODES: int = 20_000
MAX_MOVES: int = 5_000
EVAL_INTERVAL: int = 500
EVAL_GAMES: int = 20

AGENT_KWARGS = dict(
    training=True,
    lr=5e-4,
    gamma=0.995,
    batch_size=512,
    buffer_capacity=200_000,
    epsilon_start=1.0,
    epsilon_end=0.02,
    epsilon_decay_steps=300_000,
    target_update_interval=10_000,
)

# store best models in "runs" directory
RUN_DIR = Path("runs") / time.strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)


def play_episode(agent: RLAgent, max_moves: int = MAX_MOVES) -> Tuple[float, int]:
    """Play a single training episode and return (reward, moves)."""
    game = Game2048()
    total_reward = 0.0
    moves = 0
    for _ in range(max_moves):
        if game.is_game_over():
            break
        action = agent.get_move(game)
        info = game.step(action)
        reward = np.log1p(info["score"]) / 5.0  # mild shaping
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

    for episode in range(1, TOTAL_EPISODES + 1):
        reward, moves = play_episode(agent)
        recent_rewards.append(reward)

        if episode % 10 == 0:
            avg100 = statistics.mean(recent_rewards) if recent_rewards else 0.0
            print(
                f"Episode {episode:>6}/{TOTAL_EPISODES}  "
                f"ε={agent._current_epsilon():.3f}  "
                f"avgR100={avg100:.3f}  "
                f"moves={moves}  "
                f"bestEval={best_eval_score:.0f}")

        # Periodic evaluation after ever EVAL_INTERVAL episodes
        if episode % EVAL_INTERVAL == 0:
            avg_score = evaluate(agent, EVAL_GAMES)
            print(f"[Eval] Episode {episode}: average greedy score = {avg_score:.0f}")
            if avg_score > best_eval_score:
                best_eval_score = avg_score
                ckpt = RUN_DIR / f"best_{int(best_eval_score)}.pt"
                agent.save_model(ckpt.as_posix())
                print(f"  New best score! Model saved to {ckpt}")

    # Save the final model
    agent.save_model((RUN_DIR / "final.pt").as_posix())
    print(
        f"Training finished. Best evaluation score: {best_eval_score:.0f}\n"
        f"Models stored in {RUN_DIR}")
