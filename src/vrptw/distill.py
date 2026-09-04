from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .core import load_solomon_instance
from .operators import N_ACTIONS
from .rl import DEVICE, QNet
from .solvers import HybridRuleSolver


class TrajectoryCollector:
    """Collects (state_vector, action_index, reward) trajectories from baseline runs."""

    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []

    def add(self, state: np.ndarray, action: int, reward: float):
        self.states.append(state.copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "states": np.array(self.states, dtype=np.float32),
                    "actions": np.array(self.actions, dtype=np.int64),
                    "rewards": np.array(self.rewards, dtype=np.float32),
                },
                f,
            )
        print(f"Saved {len(self.states)} trajectory steps to {filepath}")

    @staticmethod
    def load(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data["states"], data["actions"], data["rewards"]


def bc_pretrain_operator_controller(
    states: np.ndarray,
    actions: np.ndarray,
    q_net: QNet,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> float:
    """Pre-train Dueling DDQN Q-Net via Behavior Cloning (Cross-Entropy Loss)."""
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    q_net.to(DEVICE)
    q_net.train()

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for b_states, b_actions in loader:
            b_states = b_states.to(DEVICE)
            b_actions = b_actions.to(DEVICE)

            logits = q_net(b_states)
            loss = criterion(logits, b_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(b_states)

        final_loss = epoch_loss / len(states)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch + 1}/{epochs}] - BC Loss: {final_loss:.4f}")

    return final_loss


def main():
    parser = argparse.ArgumentParser(description="VRPTW Trajectory Collection & Behavior Cloning Distillation")
    parser.add_argument("--collect", action="store_true", help="Collect trajectory data from solver runs")
    parser.add_argument("--train", action="store_true", help="Pre-train OperatorController QNet from collected data")
    parser.add_argument(
        "--instances",
        type=str,
        default="data/Solomon/rc101.txt,data/Solomon/r101.txt",
        help="Comma-separated instance file paths",
    )
    parser.add_argument("--input", type=str, default="scratch/trajectories.pkl", help="Input file path for training")
    parser.add_argument("--output", type=str, default="scratch/operator_bc_weights.pt", help="Output weights file path")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")

    args = parser.parse_args()

    if args.collect:
        collector = TrajectoryCollector()
        instance_paths = [p.strip() for p in args.instances.split(",")]
        cfg = Config(hybrid_iterations=1500)

        for path in instance_paths:
            if not os.path.exists(path):
                print(f"Warning: {path} not found. Skipping.")
                continue
            inst = load_solomon_instance(path)
            solver = HybridRuleSolver(inst, cfg)
            print(f"Collecting trajectories from {inst.name} using HybridRuleSolver...")

            # Run solver and record states & actions
            plan, _ = solver.solve(seed=42)
            # Fetch recorded trajectories if stored on solver
            if hasattr(solver, "trajectory_buffer"):
                for s, a, r in solver.trajectory_buffer:
                    collector.add(s, a, r)

        collector.save(args.input)

    if args.train:
        if not os.path.exists(args.input):
            print(f"Error: Trajectory dataset {args.input} not found! Run --collect first.")
            sys.exit(1)

        print(f"Loading trajectories from {args.input}...")
        states, actions, rewards = TrajectoryCollector.load(args.input)
        print(f"Dataset shape: States {states.shape}, Actions {actions.shape}")

        cfg = Config()
        q_net = QNet(cfg.op_state_dim, N_ACTIONS, cfg.op_hidden)

        print("Pre-training OperatorController via Behavior Cloning...")
        loss = bc_pretrain_operator_controller(states, actions, q_net, epochs=args.epochs)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.save(q_net.state_dict(), args.output)
        print(f"Successfully saved distilled weights to {args.output} (Final Loss: {loss:.4f})")


if __name__ == "__main__":
    main()
