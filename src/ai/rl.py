import random
from collections import deque, namedtuple
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Transition function for RL
Transition = namedtuple("Transition", [
    "state",      # np.ndarray shape (16,)
    "action",     # int
    "reward",     # float
    "next_state", # np.ndarray | None
    "done"        # bool
])

class ReplayBuffer:
    """Replay buffer for storing transitions."""
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        indices = random.sample(range(len(self.memory)), batch_size)
        return [self.memory[i] for i in indices]

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 16, hidden: int = 128, output_dim: int = 4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):  # x: (B, 16)
        return self.layers(x)


def preprocess_grid(grid: np.ndarray) -> np.ndarray:
    """Flatten 4×4 grid and apply log2 scaling (zero stays zero)."""
    flat = grid.flatten()
    with np.errstate(divide="ignore"):
        vec = np.where(flat > 0, np.log2(flat), 0).astype(np.float32)
    return vec


def default_device(explicit: Optional[str] = None) -> torch.device:
    """Select the best available device: CUDA (NVIDIA GPU) -> MPS (Apple GPU) -> CPU."""
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class RLAgent:
    """Deep Q‑Learning agent compatible with `AgentAnalyzer`."""

    def __init__(
        self,
        name: str = "DQNAgent",
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        batch_size: int = 128,
        buffer_capacity: int = 50_000,
        target_update_interval: int = 1_000,
        device: Optional[str] = None,
        training: bool = False,
    ):
        self.name = name
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.training = training
        self._steps_done = 0

        self.device = default_device(device)

        # Neural network defs
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Save the constructor args
        self._init_kwargs = dict(
            name=name, lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            batch_size=batch_size, buffer_capacity=buffer_capacity,
            target_update_interval=target_update_interval,
            device=str(self.device), training=training,
        )

    def get_move(self, game):
        """Return an action (0–3) for the given *game* state."""
        state_vec = preprocess_grid(game.board.grid)
        valid_moves = game.get_available_moves()

        # ε‑greedy selection
        eps = self._current_epsilon()
        if random.random() < eps:
            action = random.choice(valid_moves)
        else:
            q_values = self._predict_q(state_vec)
            mask = np.ones(4, dtype=bool)
            mask[valid_moves] = False
            q_values[mask] = -np.inf
            action = int(np.nanargmax(q_values))

        if not self.training:
            self._steps_done += 1
            return action

        # save state and action for learning
        self._pending_state = state_vec
        self._pending_action = action
        return action


    def remember_last(self, reward: float, next_board: np.ndarray, done: bool):
        """Store the latest transition and trigger a learning step."""
        next_state = None if done else preprocess_grid(next_board)
        self.replay_buffer.push(
            self._pending_state, self._pending_action, reward, next_state, done
        )
        self._learn()

    def end_episode(self):
        if self._steps_done % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _current_epsilon(self) -> float:
        if not self.training:
            return 0.0
        fraction = min(1.0, self._steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _predict_q(self, state_vec: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state_vec, device=self.device).unsqueeze(0)
            return self.policy_net(state).cpu().numpy()[0]

    def _learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        self._steps_done += 1

        batch = Transition(*zip(*self.replay_buffer.sample(self.batch_size)))

        state_batch = torch.as_tensor(np.stack(batch.state), device=self.device)
        action_batch = torch.as_tensor(batch.action, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.as_tensor(batch.reward, device=self.device, dtype=torch.float32)

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.as_tensor(
            np.stack([s for s in batch.next_state if s is not None]), device=self.device
        ) if non_final_mask.any() else torch.empty((0, 16), device=self.device)

        # Q(s,a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # V(s')
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.any():
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected = reward_batch + self.gamma * next_state_values

        loss = self.criterion(state_action_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def save_model(self, path: str):
        """Save the model to the specified path."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "steps_done": self._steps_done,
            "init_kwargs": self._init_kwargs,
        }, path)

    @classmethod
    def load_model(cls, path: str, **overrides):
        checkpoint = torch.load(path, map_location="cpu")
        kwargs = checkpoint["init_kwargs"]
        kwargs.update(overrides)
        agent = cls(**kwargs)
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent._steps_done = checkpoint["steps_done"]
        return agent

    def get_config(self):
        return self._init_kwargs.copy()

    def __repr__(self):
        return f"<RLAgent {self.name} device={self.device} steps={self._steps_done} training={self.training}>"
