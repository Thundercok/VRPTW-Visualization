from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config, default_data_path
from .core import Inst, Plan
from .generators import load_datasets
from .split_controller import DEVICE, SplitController, extract_giant_tour


def load_elite_plans(folders: list[str], insts_dict: dict[str, Inst]) -> list[Plan]:
    seen = {}
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path) as f:
                        data = json.load(f)
                    inst_name = data["instance"]
                    if inst_name in insts_dict:
                        plan = Plan(data["routes"], insts_dict[inst_name], data.get("algo", ""))
                        if plan.feasible:
                            existing = seen.get(inst_name)
                            if (
                                existing is None
                                or plan.nv < existing.nv
                                or (plan.nv == existing.nv and plan.cost < existing.cost)
                            ):
                                seen[inst_name] = plan
                except Exception:
                    pass
    return list(seen.values())


def pretrain_split_controller(
    plans: list[Plan], epochs: int = 120, lr: float = 3e-4, save_path: str | None = None
) -> dict:
    """
    Behavior cloning pre-training for SplitController using elite partitions.
    """
    if not plans:
        print("No plans available for pre-training.")
        return {}

    print(f"Pre-training SplitController on {len(plans)} plans for {epochs} epochs...")

    # We use a dummy instance/config to initialize the controller first
    dummy_plan = plans[0]
    cfg = Config(split_lr=lr)

    # We will train a single shared policy on all instances
    # Since the state features are normalized (coords in [0,1], loads, slacks, remaining frac),
    # the network can generalize across different instances!
    controller = SplitController(cfg, dummy_plan.inst)
    optimizer = optim.Adam(controller.q.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        samples_count = 0
        random.shuffle(plans)

        for plan in plans:
            inst = plan.inst
            # Dynamically update the controller's instance context for feature extraction
            controller.inst = inst
            # No GNN heatmap is used during simple pre-training, or we can use zero
            controller.heatmap = np.zeros((inst.n + 1, inst.n + 1), dtype=np.float32)

            tour = extract_giant_tour(plan)

            # Data Augmentation: randomly shift the circular tour
            if random.random() < 0.5:
                shift = random.randint(0, len(tour) - 1)
                tour = tour[shift:] + tour[:shift]

            # Map each customer to its route index in the elite plan
            cust_to_route = {}
            for r_idx, route in enumerate(plan.routes):
                for c in route:
                    cust_to_route[c] = r_idx

            current_route = []
            states = []
            targets = []

            for i, customer in enumerate(tour):
                if i == 0:
                    current_route.append(customer)
                    continue

                # Build state for deciding split/continue at customer i
                state, can_continue = controller._build_state(current_route, customer, tour, i, len(current_route))

                # If we cannot continue, the action is forced split (1),
                # which doesn't provide a useful gradients for voluntary choice.
                if not can_continue:
                    # Forced split: start new route
                    current_route = [customer]
                    continue

                # Determine target action from elite partition:
                # If they are in the same route in the elite plan → CONTINUE (0), else SPLIT (1)
                prev_cust = current_route[-1]
                target_action = 0 if cust_to_route[prev_cust] == cust_to_route[customer] else 1

                states.append(state)
                targets.append(target_action)

                if target_action == 1:
                    current_route = [customer]
                else:
                    current_route.append(customer)

            if states:
                s_t = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
                y_t = torch.tensor(targets, dtype=torch.long, device=DEVICE)

                logits = controller.q(s_t)
                loss = loss_fn(logits, y_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(states)
                samples_count += len(states)

        if samples_count > 0:
            avg_loss = epoch_loss / samples_count
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.5f}")

    # Sync target network weights
    controller.q_target.load_state_dict(controller.q.state_dict())

    # Clone weights in the standard format
    weights = {}
    for k, v in controller.q.state_dict().items():
        weights[f"split.{k}"] = v.clone().cpu()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Check if safetensors is available
        try:
            from safetensors.torch import save_file

            save_file(weights, save_path)
            print(f"Saved pre-trained weights to safetensors → {save_path}")
        except ImportError:
            torch.save(weights, save_path.replace(".safetensors", ".pt"))
            print(f"Saved pre-trained weights to torch file → {save_path.replace('.safetensors', '.pt')}")

    return weights


def main():
    parser = argparse.ArgumentParser(description="Pre-train the SplitController on Elite Archive plans.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to Solomon datasets")
    parser.add_argument(
        "--elite-path",
        type=str,
        nargs="+",
        default=["./elite_plans", "./results/ultimate-publication-suite"],
        help="Path(s) to elite JSON solutions",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Pre-training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./output_split_bench/split_pretrained.safetensors",
        help="Path to save weights",
    )
    args = parser.parse_args()

    data_path = args.data_path or default_data_path()
    print(f"Loading datasets from: {data_path}")
    datasets = load_datasets(data_path)

    insts_dict = {}
    for _group, insts in datasets.items():
        for inst in insts:
            insts_dict[inst.name] = inst

    plans = load_elite_plans(args.elite_path, insts_dict)
    print(f"Loaded {len(plans)} feasible elite plans.")

    if not plans:
        print("Error: No plans found to pretrain on. Make sure --elite-path is correct.")
        return

    pretrain_split_controller(plans, epochs=args.epochs, lr=args.lr, save_path=args.save_path)


if __name__ == "__main__":
    main()
