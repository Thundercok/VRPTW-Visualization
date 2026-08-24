from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import Config
from .core import Inst, Plan
from .rl import DEVICE, PrioritizedReplayBuffer


class SplitQNet(nn.Module):
    """
    Dueling DDQN for binary split decision (CONTINUE vs SPLIT).
    """

    def __init__(self, state_dim: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 actions: 0=CONTINUE, 1=SPLIT
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + a - a.mean(dim=1, keepdim=True)


def extract_giant_tour(plan: Plan) -> list[int]:
    """
    Convert a Plan into a single ordered customer sequence.
    Sorts all customers by their polar angle relative to the depot.
    Ties are broken by distance from the depot.
    """
    inst = plan.inst
    depot_coord = inst.coords[0]

    customers = []
    for r in plan.routes:
        customers.extend(r)

    # Remove duplicates just in case, though a plan should have exactly
    # each customer visited once.
    customers = list(dict.fromkeys(customers))

    def polar_angle_key(c: int) -> tuple[float, float]:
        dx = inst.coords[c][0] - depot_coord[0]
        dy = inst.coords[c][1] - depot_coord[1]
        angle = math.atan2(dy, dx)
        dist = inst.dist[0, c]
        return angle, dist

    customers.sort(key=polar_angle_key)
    return customers


class SplitController:
    """
    RL-guided split controller utilizing Dueling DDQN with PER.
    """

    def __init__(self, cfg: Config, inst: Inst, heatmap: np.ndarray | None = None):
        self.cfg = cfg
        self.inst = inst
        self.heatmap = heatmap if heatmap is not None else np.zeros((inst.n + 1, inst.n + 1), dtype=np.float32)

        self.q = SplitQNet(cfg.split_state_dim, cfg.split_hidden).to(DEVICE)
        self.q_target = SplitQNet(cfg.split_state_dim, cfg.split_hidden).to(DEVICE)
        self.q_target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.split_lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=5000, eta_min=1e-5)
        self.buf = PrioritizedReplayBuffer(cfg.split_buffer, expected_steps=cfg.per_beta_steps)

        self.eps = cfg.split_eps_start
        # Decay eps over approx 80% of hybrid iterations
        self.eps_decay = 0.02 ** (1.0 / max(cfg.hybrid_iterations * 0.8, 1))
        self.step = 0

    def reset(self) -> None:
        self.eps = self.cfg.split_eps_start

    def act(self, state: np.ndarray, can_continue: bool) -> int:
        """
        Choose action {0: CONTINUE, 1: SPLIT}.
        If can_continue is False, we are forced to SPLIT (action 1).
        """
        if not can_continue:
            return 1

        if random.random() < self.eps:
            return random.choice([0, 1])

        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = self.q(s_t)[0]
            return int(q_values.argmax().item())

    def _build_state(
        self,
        current_route: list[int],
        next_cust: int,
        tour: list[int],
        tour_idx: int,
        routes_so_far: int,
    ) -> tuple[np.ndarray, bool]:
        """
        Extracts the 18-dimensional state representation for the current decision point.
        Returns state array and can_continue boolean.
        """
        inst = self.inst
        capacity = max(inst.capacity, 1.0)
        horizon = max(inst.horizon, 1.0)

        # Current customer in route
        if current_route:
            cur = current_route[-1]
        else:
            cur = 0

        # Get coordinates normalized in [0, 1]
        min_coords = inst.coords.min(axis=0)
        max_coords = inst.coords.max(axis=0)
        coords_range = np.where(max_coords - min_coords == 0, 1.0, max_coords - min_coords)

        norm_cur_coord = (inst.coords[cur] - min_coords) / coords_range
        norm_next_coord = (inst.coords[next_cust] - min_coords) / coords_range

        # Route load calculations
        route_load = sum(inst.demands[c] for c in current_route)
        route_load_frac = route_load / capacity
        load_if_continue = (route_load + inst.demands[next_cust]) / capacity

        # Route time calculations
        current_t = 0.0
        prev = 0
        for node in current_route:
            current_t += inst.dist[prev, node]
            current_t = max(current_t, inst.ready_times[node]) + inst.service_times[node]
            prev = node

        route_time_frac = current_t / horizon

        # Slack for current customer
        if cur != 0:
            # Reconstruct arrival time at cur
            arrival_cur = current_t - inst.service_times[cur]
            slack_cur = (inst.due_times[cur] - arrival_cur) / horizon
        else:
            slack_cur = 1.0

        # Slack and feasibility if we CONTINUE
        t_arrival_continue = current_t + inst.dist[cur, next_cust]
        slack_if_continue = (inst.due_times[next_cust] - t_arrival_continue) / horizon

        t_arrival_continue_start = max(t_arrival_continue, inst.ready_times[next_cust])
        t_finish_continue = t_arrival_continue_start + inst.service_times[next_cust]
        t_return_continue = t_finish_continue + inst.dist[next_cust, 0]

        feasible_continue = bool(
            (route_load + inst.demands[next_cust] <= capacity)
            and (t_arrival_continue <= inst.due_times[next_cust])
            and (t_return_continue <= inst.due_times[0])
        )

        # Slack if we SPLIT (start fresh from depot)
        t_arrival_split = inst.dist[0, next_cust]
        slack_if_split = (inst.due_times[next_cust] - t_arrival_split) / horizon

        # GNN heatmaps
        gnn_cur_next = self.heatmap[cur, next_cust]
        gnn_depot_next = self.heatmap[0, next_cust]

        # Remaining sequence info
        n_customers = max(inst.n, 1)
        remaining_frac = (len(tour) - tour_idx) / n_customers

        # Remaining demands
        remaining_demand = sum(inst.demands[c] for c in tour[tour_idx:])
        # Estimate minimum remaining routes needed
        min_rem_routes = max(math.ceil(remaining_demand / capacity), 1)
        demand_remaining_scaled = remaining_demand / (capacity * min_rem_routes)

        state = np.array(
            [
                norm_cur_coord[0],
                norm_cur_coord[1],
                norm_next_coord[0],
                norm_next_coord[1],
                inst.dist[cur, next_cust] / max(inst.max_dist, 1.0),
                inst.dist[0, next_cust] / max(inst.max_dist, 1.0),
                route_load_frac,
                load_if_continue,
                route_time_frac,
                slack_cur,
                slack_if_continue,
                slack_if_split,
                gnn_cur_next,
                gnn_depot_next,
                routes_so_far / n_customers,
                remaining_frac,
                demand_remaining_scaled,
                1.0 if feasible_continue else 0.0,
            ],
            dtype=np.float32,
        )

        return state, feasible_continue

    def _run_episode(self, tour: list[int], incumbent_nv: int, incumbent_cost: float) -> tuple[list[list[int]], list]:
        """
        Simulates the sequential splitting decision process on a giant tour.
        Returns the partitioned routes and a list of collected transitions.
        """
        routes = []
        current_route = []
        transitions = []

        for i, customer in enumerate(tour):
            if i == 0:
                current_route.append(customer)
                continue

            state, can_continue = self._build_state(
                current_route, customer, tour, i, len(routes)
            )

            # Action: 0 = CONTINUE, 1 = SPLIT
            if not can_continue:
                action = 1
                is_forced = True
                step_reward = 0.0
            else:
                action = self.act(state, can_continue=True)
                is_forced = False
                # voluntary SPLIT has small penalty, CONTINUE has small reward
                step_reward = -0.1 if action == 1 else 0.05

            if action == 1:
                # Store route and start new
                routes.append(current_route)
                current_route = [customer]
            else:
                current_route.append(customer)

            # Build next state representation
            next_state, _ = self._build_state(
                current_route, customer, tour, i, len(routes)
            )

            done = 1.0 if (i == len(tour) - 1) else 0.0
            transitions.append({
                "state": state,
                "action": action,
                "reward": step_reward,
                "next_state": next_state,
                "done": done,
                "is_forced": is_forced
            })

        routes.append(current_route)
        return routes, transitions

    def train_step(self) -> None:
        """
        Perform one DQN gradient descent step.
        """
        self.step += 1
        if len(self.buf) < self.cfg.split_batch:
            return

        (s, a, r, ns, d), idxs, is_w = self.buf.sample(self.cfg.split_batch)
        s = torch.tensor(s).to(DEVICE)
        a = torch.tensor(a, dtype=torch.long).to(DEVICE)
        r = torch.tensor(r).to(DEVICE)
        ns = torch.tensor(ns).to(DEVICE)
        d = torch.tensor(d).to(DEVICE)

        qp = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.q(ns).argmax(1).unsqueeze(1)
            qn = self.q_target(ns).gather(1, best_a).squeeze(1)
            target = r + self.cfg.split_gamma * qn * (1.0 - d)

        td_errors = (qp - target).detach().cpu().numpy()
        self.buf.update_priorities(idxs, td_errors)

        loss = (is_w * F.smooth_l1_loss(qp, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()

        # Soft update target network
        tau = self.cfg.split_tau
        for target_param, local_param in zip(self.q_target.parameters(), self.q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        self.eps = max(self.cfg.split_eps_end, self.eps * self.eps_decay)

    def try_split(self, plan: Plan, incumbent_nv: int) -> Plan | None:
        """
        Standalone method that takes a Plan, extracts a giant tour, runs SplitController,
        evaluates feasibility, computes rewards, trains SplitController, and returns
        the new Plan if it strictly dominates the original.
        """
        if not plan.routes:
            return None

        tour = extract_giant_tour(plan)
        routes, transitions = self._run_episode(tour, incumbent_nv, plan.cost)

        # Construct proposed candidate
        cand = Plan(routes, plan.inst, "RL-SPLIT")

        # Episode reward
        episode_reward = 0.0
        if cand.feasible:
            # Highly reward reducing NV below current best
            if cand.nv < incumbent_nv:
                episode_reward += (incumbent_nv - cand.nv) * self.cfg.split_nv_penalty
            # Cost improvement bonus if same or better NV
            if cand.nv <= incumbent_nv:
                cost_impr = max(0.0, (plan.cost - cand.cost) / max(plan.cost, 1.0))
                episode_reward += cost_impr * 2.0
        else:
            episode_reward -= self.cfg.split_infeasible_penalty

        # Distribute episode reward to the final step transition
        if transitions:
            transitions[-1]["reward"] += episode_reward

        # Push experiences to buffer (skipping forced split decisions)
        for trans in transitions:
            if not trans["is_forced"]:
                self.buf.push(
                    trans["state"],
                    trans["action"],
                    trans["reward"],
                    trans["next_state"],
                    trans["done"]
                )

        # Trigger gradient updates
        self.train_step()

        if cand.feasible and cand.dominates(plan):
            return cand

        return None
