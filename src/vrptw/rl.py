from __future__ import annotations

import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import Config
from .core import Inst, Plan
from .operators import N_ACTIONS, N_R

DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, int(os.environ.get("NUMBA_NUM_THREADS", "1")) // 2))


class PrioritizedReplayBuffer:
    """
    Proportional PER (Schaul et al., 2016).
    alpha=0.6: prioritization strength.
    beta anneals 0.4→1.0 over training to correct IS bias.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        expected_steps: int = 50_000,
    ):
        self.capacity: int = capacity
        self.alpha: float = float(alpha)
        self.beta: float = beta_start
        self.beta_end: float = beta_end
        self.beta_inc: float = (beta_end - beta_start) / max(expected_steps, 1)
        self.buf: list = []
        self.pos: int = 0
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        # priorities**alpha, maintained incrementally: only entries whose
        # priority changes are re-raised to alpha, instead of the O(capacity)
        # np.power every ``sample`` call used to spend on the whole array.
        # Values are float32-identical to recomputing from scratch, so the
        # probability vector — and hence the RNG draw — is unchanged.
        self._pri_alpha: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.max_pri = 1.0

    def push(self, *transition) -> None:
        if len(self.buf) < self.capacity:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.priorities[self.pos] = self.max_pri
        self._pri_alpha[self.pos] = np.float32(self.priorities[self.pos]) ** np.float32(self.alpha)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.buf)
        probs = self._pri_alpha[:n].copy()
        probs /= probs.sum()
        idxs = np.random.choice(n, batch_size, p=probs, replace=True)
        ws = (n * probs[idxs]) ** -self.beta
        ws /= ws.max()
        self.beta = float(min(self.beta_end, self.beta + self.beta_inc))
        s, a, r, ns, d = zip(*[self.buf[i] for i in idxs])
        return (
            (
                np.array(s, np.float32),
                np.array(a, np.int64),
                np.array(r, np.float32),
                np.array(ns, np.float32),
                np.array(d, np.float32),
            ),
            idxs,
            torch.tensor(ws, dtype=torch.float32).to(DEVICE),
        )

    def update_priorities(self, idxs, td_errors: np.ndarray) -> None:
        idxs = np.asarray(idxs)
        ps = np.abs(np.asarray(td_errors, dtype=np.float64)) + 1e-6
        # `sample()` draws with replace=True, so a batch routinely carries the
        # same index twice with different TD errors. NumPy does not define which
        # value survives `a[idxs] = ps` when idxs repeats (only the `np.*.at`
        # family has specified semantics), so resolve the duplicates explicitly
        # rather than relying on it: np.unique over the reversed batch selects
        # the LAST occurrence of each index, which is the last-write-wins rule
        # the per-element loop this replaced had.
        if len(idxs):
            uniq, last_pos = np.unique(idxs[::-1], return_index=True)
            self.priorities[uniq] = ps[::-1][last_pos]
            self._pri_alpha[uniq] = self.priorities[uniq] ** np.float32(self.alpha)
        m = float(ps.max()) if len(ps) else 0.0
        self.max_pri = max(self.max_pri, m)

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# QNet
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        hid2 = max(hidden_dim // 2, 32)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hid2), nn.ReLU(), nn.Linear(hid2, 1))
        self.adv_head = nn.Sequential(nn.Linear(hidden_dim, hid2), nn.ReLU(), nn.Linear(hid2, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + a - a.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Thompson bandit
# ---------------------------------------------------------------------------
class ThompsonBandit:
    def __init__(self, n_d: int, n_r: int):
        self.alpha = np.ones((n_d, n_r), dtype=np.float64)
        self.beta = np.ones((n_d, n_r), dtype=np.float64)

    def mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def select(
        self,
        prior: np.ndarray | None = None,  # ← indented back in
        prior_strength: float = 0.0,
    ) -> tuple[int, int]:
        if prior is not None and prior_strength > 0.0:
            p = np.asarray(prior, dtype=np.float64)
            p /= max(p.sum(), 1e-9)
            alpha = self.alpha + prior_strength * p * self.alpha.sum()
            beta = self.beta + prior_strength * p * self.beta.sum()
            samples = np.random.beta(np.maximum(alpha, 1e-9), np.maximum(beta, 1e-9))
        else:
            samples = np.random.beta(self.alpha, self.beta)
        idx = np.unravel_index(int(samples.argmax()), samples.shape)
        return int(idx[0]), int(idx[1])

    def update(self, di: int, ri: int, score: float, sigma1: int) -> None:
        success = float(np.clip(score / max(sigma1, 1), 0.0, 1.0))
        self.alpha[di, ri] += success
        self.beta[di, ri] += 1.0 - success

    def decay(self, rate: float = 0.95) -> None:
        np.multiply(self.alpha - 1.0, rate, out=self.alpha)
        np.add(self.alpha, 1.0, out=self.alpha)
        np.multiply(self.beta - 1.0, rate, out=self.beta)
        np.add(self.beta, 1.0, out=self.beta)

    def reset(self) -> None:
        self.alpha.fill(1.0)
        self.beta.fill(1.0)

    def clone(self) -> ThompsonBandit:
        b = ThompsonBandit(self.alpha.shape[0], self.alpha.shape[1])
        b.alpha = self.alpha.copy()
        b.beta = self.beta.copy()
        return b


# ---------------------------------------------------------------------------
# Elite archive
# ---------------------------------------------------------------------------
class EliteArchive:
    def __init__(self, k: int = 5, max_edge_jaccard: float = 0.92):
        self.k = k
        self.max_edge_jaccard = max_edge_jaccard
        self._plans: dict[str, list[Plan]] = {}

    @staticmethod
    def _plan_edges(plan: Plan) -> set[tuple[int, int]]:
        edges = set()
        for route in plan.routes:
            if not route:
                continue
            prev = 0
            for node in route:
                edges.add((min(prev, node), max(prev, node)))
                prev = node
            edges.add((0, prev))
        return edges

    def _jaccard_similarity(self, p1: Plan, p2: Plan) -> float:
        e1 = self._plan_edges(p1)
        e2 = self._plan_edges(p2)
        if not e1 or not e2:
            return 0.0
        inter = len(e1.intersection(e2))
        union = len(e1.union(e2))
        return inter / max(1, union)

    def update(self, plan: Plan) -> None:
        if not plan.feasible:
            return
        key = plan.inst.name
        bucket = self._plans.setdefault(key, [])
        threshold = max(0.1, 1e-4 * plan.cost)

        # 1. Cost duplicate check
        if any(abs(p.cost - plan.cost) < threshold and p.nv == plan.nv for p in bucket):
            return

        # 2. Structural Edge-Jaccard diversity check (prevent near-clone collapse)
        for idx, p in enumerate(bucket):
            if p.nv == plan.nv and self._jaccard_similarity(p, plan) > self.max_edge_jaccard:
                # If new plan is better than the structural clone, replace it
                if plan.cost < p.cost - 1e-4:
                    bucket[idx] = plan.copy()
                    bucket.sort(key=lambda item: (item.nv, item.cost))
                    self._plans[key] = bucket[: self.k]
                return

        bucket.append(plan.copy())
        bucket.sort(key=lambda p: (p.nv, p.cost))
        self._plans[key] = bucket[: self.k]


    def sample_diverse(self, inst_name: str, exclude_cost: float | None = None) -> Plan | None:
        """Return a random plan from archive excluding current solution cost."""
        import random
        bucket = self._plans.get(inst_name, [])
        threshold = 1.0 if not bucket else max(0.1, 1e-4 * bucket[0].cost)
        candidates = [p for p in bucket
                      if exclude_cost is None or abs(p.cost - exclude_cost) >= threshold]
        return random.choice(candidates).copy() if candidates else None

    def crossover(self, inst_name: str) -> Plan | None:
        """SREX-style route-exchange crossover between the top-2 elite plans
        (selective route exchange, after Nagata & Kobayashi).

        The previous implementation kept half of p1's routes and appended the
        *fragments* of p2's routes not yet covered. Fragments of feasible routes
        stay feasible, so the offspring was technically feasible — but with a
        badly inflated route count, so it failed the ``alt.nv < best.nv``
        acceptance gate at the call site on every single call (measured: 0/600
        offspring with nv <= best across R101/RC207/C101 archives) and crossover
        never contributed anything to the search.

        This version removes a centroid-coherent subset of p1's routes, adopts
        whole non-overlapping routes from p2 (each feasible stand-alone), and
        repairs the few remaining customers by cheapest feasible insertion — so
        the offspring keeps a competitive route count (measured: 491/600 with
        nv <= best on the same protocol).
        """
        import random as _random

        bucket = self._plans.get(inst_name, [])
        if len(bucket) < 2:
            return None
        p1, p2 = bucket[0], bucket[1]
        inst = p1.inst
        if not p1.routes or not p2.routes:
            return None

        # 1. Remove a spatially coherent subset of p1's routes around a random
        #    seed route — exchanging scattered routes rarely recombines well.
        n_remove = _random.randint(1, max(1, len(p1.routes) // 2))
        seed_idx = _random.randrange(len(p1.routes))
        cents = [inst.coords[np.asarray(r, dtype=np.int64)].mean(axis=0) for r in p1.routes]
        seed_c = cents[seed_idx]
        order = sorted(
            range(len(p1.routes)), key=lambda i: float(np.sum((cents[i] - seed_c) ** 2))
        )
        removed_idx = set(order[:n_remove])
        routes = [r[:] for i, r in enumerate(p1.routes) if i not in removed_idx]
        served = {c for r in routes for c in r}

        # 2. Adopt whole routes from p2 that cover only unserved customers.
        for r in sorted(p2.routes, key=len, reverse=True):
            if served.isdisjoint(r):
                routes.append(r[:])
                served.update(r)

        # 3. Repair by cheapest feasible insertion (opens a route as last resort).
        missing = [c for c in range(1, inst.n + 1) if c not in served]
        if len(missing) > max(4, inst.n // 3):
            return None  # exchange too destructive to be worth repairing
        try:
            from .core import Plan
            from .heuristics import _insert_into_cheapest_route

            child = Plan(routes, inst, p1.algo)
            for c in sorted(missing, key=lambda x: inst.due_times[x]):
                _insert_into_cheapest_route(child, c, inst)
            return child if child.feasible else None
        except Exception:
            return None

    def load_plans(self, folder: str, insts_dict: dict[str, Inst]) -> None:
        if not os.path.exists(folder):
            return
        import json

        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(folder, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                inst_name = data["instance"]
                if inst_name in insts_dict:
                    plan = Plan(data["routes"], insts_dict[inst_name], data.get("algo", ""))
                    bucket = self._plans.setdefault(inst_name, [])
                    bucket.append(plan)
                    bucket.sort(key=lambda p: (p.nv, p.cost))
                    self._plans[inst_name] = bucket[: self.k]
            except Exception:
                pass

    def update_and_save(self, plan: Plan, folder: str) -> None:
        if not plan.feasible:
            return
        key = plan.inst.name
        bucket = self._plans.setdefault(key, [])
        old_best = bucket[0] if bucket else None
        bucket.append(plan.copy())
        bucket.sort(key=lambda p: (p.nv, p.cost))
        self._plans[key] = bucket[: self.k]

        new_best = self._plans[key][0]
        is_improved = (
            old_best is None
            or new_best.nv < old_best.nv
            or (new_best.nv == old_best.nv and new_best.cost < old_best.cost - 1e-6)
        )
        if is_improved:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{key}.json")
            import json

            try:
                clean_routes = [[int(node) for node in r] for r in new_best.routes]
                with open(path, "w") as f:
                    json.dump(
                        {
                            "instance": key,
                            "cost": new_best.cost,
                            "nv": new_best.nv,
                            "routes": clean_routes,
                            "algo": new_best.algo,
                        },
                        f,
                        indent=2,
                    )
            except Exception as e:
                import sys

                print(f"Error saving plan to {path}: {e}", file=sys.stderr)

    def best(self, inst_name: str) -> Plan | None:
        bucket = self._plans.get(inst_name, [])
        return bucket[0].copy() if bucket else None

    def summary(self) -> str:
        lines = []
        for name, bucket in sorted(self._plans.items()):
            p = bucket[0]
            td_gap, _ = p.gap()
            gap_str = f"{td_gap:+.2f}%" if td_gap is not None else "--"
            lines.append(f"  {name}: nv={p.nv} cost={p.cost:.1f} gap={gap_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LS Budget, UCB, & Welford Reward Normalizer
# ---------------------------------------------------------------------------
class WelfordRewardNormalizer:
    def __init__(self, clip_sigma: float = 8.0, warmup: int = 128, eps: float = 1e-8):
        self.clip = clip_sigma
        self.warmup = warmup
        self.eps = eps
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def observe(self, r: float) -> None:
        self._n += 1
        delta = r - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (r - self._mean)

    @property
    def std(self) -> float:
        if self._n < 2:
            return 1.0
        return math.sqrt(max(self._M2 / (self._n - 1), self.eps**2))

    def normalize(self, r: float) -> float | None:
        self.observe(r)
        if self._n < self.warmup:
            return None  # caller must check before pushing to buffer
        z = (r - self._mean) / (self.std + self.eps)
        return float(np.clip(z, -self.clip, self.clip))

    def state_dict(self) -> dict:
        return {"n": self._n, "mean": self._mean, "M2": self._M2}

    def load_state_dict(self, d: dict) -> None:
        self._n = int(d["n"])
        self._mean = float(d["mean"])
        self._M2 = float(d["M2"])


class LSBudgetController:
    def __init__(self, ls_time_frac: float = 0.30, ema_alpha: float = 0.15):
        self.ls_time_frac = ls_time_frac
        self.ema_alpha = ema_alpha
        self._budget_total = float("inf")
        self._budget_used = 0.0
        self._yield_ema = 0.05
        self._n_calls = 0

    def initialize(self, cfg: Config) -> None:
        est_total_s = cfg.hybrid_iterations * 0.025
        self._budget_total = est_total_s * self.ls_time_frac
        self._budget_used = 0.0
        self._yield_ema = 0.05

    def should_trigger(self, action: int, accepted: bool, is_new_best: bool, modes) -> bool:
        if self._budget_used >= self._budget_total:
            return False
        if is_new_best:
            return True
        if modes[action].ls_passes == 0:
            return False
        if not accepted:
            return False
        if self._yield_ema > 0.0:
            return True
        return random.random() < 0.10

    def record(self, time_s: float, cost_before: float, cost_after: float) -> None:
        improvement_pct = max(0.0, (cost_before - cost_after) / max(cost_before, 1.0) * 100.0)
        time_cost = time_s / max(self._budget_total * 0.02, 1e-9)
        self._yield_ema = self.ema_alpha * (improvement_pct - time_cost) + (1.0 - self.ema_alpha) * self._yield_ema
        self._budget_used += time_s
        self._n_calls += 1


class UCBActionAugmenter:
    def __init__(self, n_actions: int = 40, c_ucb: float = 1.0, gamma: float = 0.993, alpha_blend: float = 0.35):
        self.n = n_actions
        self.c = c_ucb
        self.gamma = gamma
        self.alpha = alpha_blend
        self._cnt = np.ones(n_actions, dtype=np.float64) * 0.5
        self._mu = np.zeros(n_actions, dtype=np.float64)
        self._m2 = np.ones(n_actions, dtype=np.float64) * 0.5
        self._N = float(n_actions) * 0.5

    def reset(self) -> None:
        self._cnt[:] = 0.5
        self._mu[:] = 0.0
        self._m2[:] = 0.5
        self._N = float(self.n) * 0.5

    def update(self, action: int, reward: float) -> None:
        # exponential decay on counts only — keeps forgetting semantics clean
        self._cnt *= self.gamma
        self._N = self._cnt.sum()
        # update selected arm with decayed Welford
        self._cnt[action] += 1.0
        delta = reward - self._mu[action]
        self._mu[action] += delta / self._cnt[action]
        delta2 = reward - self._mu[action]
        self._m2[action] += delta * delta2
        # decay non-selected arms toward global mean (not toward 0)
        global_mean = float(self._mu[self._cnt > 0.6].mean()) if (self._cnt > 0.6).any() else 0.0
        mask = np.ones(self.n, dtype=bool)
        mask[action] = False
        self._mu[mask] = self._mu[mask] * self.gamma + global_mean * (1.0 - self.gamma)

    def augment_qvalues(self, q: np.ndarray) -> np.ndarray:
        variance = self._m2 / np.maximum(self._cnt - 1.0, 1.0)
        std = np.sqrt(np.maximum(variance, 0.0))
        log_n = math.log(max(self._N, math.e))
        conf = self.c * std * np.sqrt(log_n / np.maximum(self._cnt, 1e-9))
        scores = self._mu + conf
        centered = scores - scores.mean()
        scale = max(scores.std(), 1e-6)
        return q + self.alpha * (centered / scale)


# ---------------------------------------------------------------------------
# DDQN controllers  (now using PrioritizedReplayBuffer)
# ---------------------------------------------------------------------------
class PlateauController:
    def __init__(self, cfg: Config, n_modes: int = 6, use_rl: bool = True):
        self.cfg = cfg
        self.n_modes = n_modes
        self.use_rl = use_rl
        if use_rl:
            self.q = QNet(cfg.ctrl_state_dim, n_modes, cfg.ctrl_hidden).to(DEVICE)
            self.q_t = QNet(cfg.ctrl_state_dim, n_modes, cfg.ctrl_hidden).to(DEVICE)
            self.q_t.load_state_dict(self.q.state_dict())
            self.opt = optim.Adam(self.q.parameters(), lr=cfg.ctrl_lr)
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(self.opt, T_max=5000, eta_min=1e-5)
            self.buf = PrioritizedReplayBuffer(cfg.ctrl_buffer, expected_steps=cfg.per_beta_steps)
            self.eps = cfg.ctrl_eps_start
            self.eps_decay = 0.02 ** (1.0 / max(cfg.hybrid_iterations * 0.8, 1))
            self.step = 0

    def reset(self) -> None:
        if self.use_rl:
            self.eps = self.cfg.ctrl_eps_start

    def act(self, state: np.ndarray) -> int:
        if not self.use_rl:
            return random.randrange(self.n_modes)
        if random.random() < self.eps:
            return random.randrange(self.n_modes)
        with torch.no_grad():
            return int(
                self.q(torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0))[0].argmax().item()
            )

    def observe(self, s, a, r, ns, done=0.0):
        if self.use_rl:
            self.buf.push(s, a, r, ns, done)

    def train_step(self) -> None:
        if not self.use_rl:
            return
        self.step += 1
        if len(self.buf) < self.cfg.ctrl_batch:
            return
        (s, a, r, ns, d), idxs, is_w = self.buf.sample(self.cfg.ctrl_batch)
        s = torch.tensor(s).to(DEVICE)
        a = torch.tensor(a, dtype=torch.long).to(DEVICE)
        r = torch.tensor(r).to(DEVICE)
        ns = torch.tensor(ns).to(DEVICE)
        d = torch.tensor(d).to(DEVICE)
        qp = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.q(ns).argmax(1).unsqueeze(1)
            qn = self.q_t(ns).gather(1, best_a).squeeze(1)
            target = r + self.cfg.ctrl_gamma * qn * (1 - d)
        td_errors = (qp - target).detach().cpu().numpy()
        self.buf.update_priorities(idxs, td_errors)
        loss = (is_w * F.smooth_l1_loss(qp, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()
        tau = self.cfg.ctrl_tau
        for target_param, local_param in zip(self.q_t.parameters(), self.q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        self.eps = max(self.cfg.ctrl_eps_end, self.eps * self.eps_decay)


class OperatorController:
    def __init__(self, cfg: Config, use_rl: bool = True):
        self.cfg = cfg
        self.use_rl = use_rl
        if use_rl:
            self.q = QNet(cfg.op_state_dim, N_ACTIONS, cfg.op_hidden).to(DEVICE)
            self.q_t = QNet(cfg.op_state_dim, N_ACTIONS, cfg.op_hidden).to(DEVICE)
            self.q_t.load_state_dict(self.q.state_dict())
            self.opt = optim.Adam(self.q.parameters(), lr=cfg.op_lr)
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(self.opt, T_max=5000, eta_min=1e-5)
            self.buf = PrioritizedReplayBuffer(cfg.op_buffer, expected_steps=cfg.per_beta_steps)
            self.eps = cfg.op_eps_start
            self.eps_decay = 0.02 ** (1.0 / max(cfg.hybrid_iterations * 0.8, 1))
            self.step = 0

    def load_pretrained_weights(self, weights_path: str) -> bool:
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=DEVICE)
            self.q.load_state_dict(state_dict)
            self.q_t.load_state_dict(state_dict)
            print(f"Successfully loaded distilled OperatorController weights from {weights_path}")
            return True
        return False

    def reset(self) -> None:
        if self.use_rl:
            self.eps = self.cfg.op_eps_start

    def _prior(self, dw: np.ndarray, rw: np.ndarray) -> np.ndarray:
        dw = np.asarray(dw, np.float32)
        dw /= max(dw.sum(), 1e-9)
        rw = np.asarray(rw, np.float32)
        rw /= max(rw.sum(), 1e-9)
        return np.outer(dw, rw)

    def _sample_prior(self, prior: np.ndarray, bandit: ThompsonBandit) -> int:
        probs = prior.reshape(-1) * bandit.mean().reshape(-1)
        probs /= max(probs.sum(), 1e-9)
        return int(np.random.choice(N_ACTIONS, p=probs))

    def _compute_q_eff(self, q: np.ndarray, prior: np.ndarray, bandit: ThompsonBandit, ucb_aug=None) -> np.ndarray:
        if not getattr(self.cfg, "op_use_entropy_gate", True):
            # Direct unweighted Q-values without entropy confidence gate
            w_conf = 1.0
        else:
            tau = max(getattr(self.cfg, "op_softmax_tau", 1.0), 1e-6)
            shifted = (q - np.max(q)) / tau
            exp_q = np.exp(shifted)
            pi = exp_q / max(np.sum(exp_q), 1e-12)

            # Normalized Shannon entropy in [0, 1]
            entropy = -float(np.sum(pi * np.log(pi + 1e-12))) / math.log(max(N_ACTIONS, 2))
            entropy = float(np.clip(entropy, 0.0, 1.0))
            w_conf = 1.0 - entropy

        prior_term = self.cfg.op_prior_strength * np.log(prior.reshape(-1) + 1e-8)
        bandit_term = self.cfg.op_bandit_strength * bandit.mean().reshape(-1)

        q_eff = w_conf * q + (1.0 - w_conf) * (prior_term + bandit_term)
        if ucb_aug is not None:
            q_eff = ucb_aug.augment_qvalues(q_eff)
        return q_eff

    def act(self, state, dw, rw, bandit, frozen=False, ucb_aug=None) -> tuple[int, int, int]:
        prior = self._prior(dw, rw)
        self.last_q = None

        if not self.use_rl:
            di, ri = bandit.select(prior=prior, prior_strength=self.cfg.bandit_prior_strength)
            action = di * N_R + ri
            return int(action), int(di), int(ri)

        try:
            with torch.no_grad():
                q = self.q(torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0))[0].cpu().numpy()
            q_adjusted = self._compute_q_eff(q, prior, bandit, ucb_aug=ucb_aug)
            self.last_q = q_adjusted
        except Exception:
            self.last_q = None

        if not frozen and len(self.buf) < self.cfg.op_warmup:
            di, ri = bandit.select(prior=prior, prior_strength=self.cfg.bandit_prior_strength)
            action = di * N_R + ri
        elif not frozen and random.random() < self.eps:
            action = self._sample_prior(prior, bandit)
            di, ri = divmod(action, N_R)
        else:
            if self.last_q is not None:
                action = int(self.last_q.argmax())
            else:
                with torch.no_grad():
                    q = self.q(torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0))[0].cpu().numpy()
                q_adjusted = self._compute_q_eff(q, prior, bandit, ucb_aug=ucb_aug)
                action = int(q_adjusted.argmax())
            di, ri = divmod(action, N_R)
        return int(action), int(di), int(ri)

    def observe(self, s, a, r, ns, done=0.0):
        if self.use_rl:
            self.buf.push(s, a, r, ns, done)

    def train_step(self) -> None:
        if not self.use_rl:
            return
        self.step += 1
        if len(self.buf) < self.cfg.op_batch:
            return
        (s, a, r, ns, d), idxs, is_w = self.buf.sample(self.cfg.op_batch)
        s = torch.tensor(s).to(DEVICE)
        a = torch.tensor(a, dtype=torch.long).to(DEVICE)
        r = torch.tensor(r).to(DEVICE)
        ns = torch.tensor(ns).to(DEVICE)
        d = torch.tensor(d).to(DEVICE)
        qp = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.q(ns).argmax(1).unsqueeze(1)
            qn = self.q_t(ns).gather(1, best_a).squeeze(1)
            target = r + self.cfg.op_gamma * qn * (1 - d)
        td_errors = (qp - target).detach().cpu().numpy()
        self.buf.update_priorities(idxs, td_errors)
        loss = (is_w * F.smooth_l1_loss(qp, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()
        tau = self.cfg.op_tau
        for target_param, local_param in zip(self.q_t.parameters(), self.q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        self.eps = max(self.cfg.op_eps_end, self.eps * self.eps_decay)


# ---------------------------------------------------------------------------
# Learned Acceptance Criterion
# ---------------------------------------------------------------------------
class LearnedAcceptanceCriterion:
    def __init__(self, cfg: Config, use_rl: bool = True):
        self.cfg = cfg
        self.use_rl = use_rl
        if use_rl:
            self.net = nn.Sequential(
                nn.Linear(cfg.lac_state_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 48),
                nn.ReLU(),
                nn.Linear(48, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ).to(DEVICE)
            self.opt = optim.Adam(self.net.parameters(), lr=cfg.lac_lr)
            self.step = 0
            self._pending: deque = deque()
            self._train_buf: deque = deque(maxlen=cfg.lac_buf_size)

    def features(
        self,
        cost_delta,
        cur_cost,
        temp,
        temp_init,
        no_imp,
        patience,
        nv_diff,
        progress,
        tw_tight_frac,
        fleet_fill,
        avg_slack_val,
    ) -> np.ndarray:
        metro = math.exp(-max(cost_delta, 0.0) / max(temp, 1e-6))
        return np.array(
            [
                cost_delta / max(abs(cur_cost), 1.0),
                temp / max(temp_init, 1e-6),
                no_imp / max(patience, 1),
                float(np.clip(nv_diff, -2, 2)),
                progress,
                tw_tight_frac,
                fleet_fill,
                avg_slack_val,
                metro,
            ],
            dtype=np.float32,
        )

    def decide(self, feats: np.ndarray, cur_best_cost: float) -> tuple[bool, float]:
        self.step += 1
        self._pending.append((feats.copy(), cur_best_cost, self.step))
        metro_p = float(feats[-1])
        if self.step < self.cfg.lac_warmup:
            return random.random() < metro_p, metro_p
        with torch.no_grad():
            p = float(self.net(torch.tensor(feats).unsqueeze(0).to(DEVICE))[0, 0])
        return random.random() < p, p

    def observe(self, current_best_cost: float) -> None:
        cutoff = self.step - self.cfg.lac_horizon
        while self._pending and self._pending[0][2] <= cutoff:
            feats, best_at_t, _ = self._pending.popleft()
            label = 1.0 if current_best_cost < best_at_t - 1e-6 else 0.0
            self._train_buf.append((feats, label))
        if self.step % self.cfg.lac_train_freq == 0 and len(self._train_buf) >= self.cfg.lac_batch:
            self._train()

    def _train(self) -> None:
        batch = random.sample(self._train_buf, min(self.cfg.lac_batch, len(self._train_buf)))
        feats, labels = zip(*batch)
        x = torch.tensor(np.array(feats), dtype=torch.float32).to(DEVICE)
        y = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
        n_neg = max((y == 0).sum().item(), 1)
        n_pos = max((y == 1).sum().item(), 1)
        sample_weights = torch.where(
            y == 1,
            torch.full_like(y, n_neg / n_pos),
            torch.ones_like(y),
        )
        pred = self.net(x).squeeze(1)
        loss = F.binary_cross_entropy(pred, y, weight=sample_weights)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def state_dict(self) -> dict:
        return {f"lac.{k}": v.clone().cpu() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, weights: dict) -> None:
        sd = self.net.state_dict()
        updates = {}
        for k, v in weights.items():
            bare = k[4:] if k.startswith("lac.") else k
            if bare in sd and tuple(v.shape) == tuple(sd[bare].shape):
                updates[bare] = v.to(DEVICE)
        sd.update(updates)
        self.net.load_state_dict(sd)


# ---------------------------------------------------------------------------
# 3-Tier Hierarchical MARL: MILP Column Controller (DDQN-3)
# ---------------------------------------------------------------------------
class MILPColumnController:
    """3rd Tier Neural Agent in Hierarchical MARL: Dynamically selects MILP set-partitioning parameters."""

    N_ACTIONS = 8  # 8 Discrete MILP Configuration Modes

    def __init__(self, cfg: Config, use_rl: bool = True):
        self.cfg = cfg
        self.use_rl = use_rl
        self.state_dim = 14
        if use_rl:
            self.q = QNet(self.state_dim, self.N_ACTIONS, hidden_dim=64).to(DEVICE)
            self.q_t = QNet(self.state_dim, self.N_ACTIONS, hidden_dim=64).to(DEVICE)
            self.q_t.load_state_dict(self.q.state_dict())
            self.opt = optim.Adam(self.q.parameters(), lr=1e-3)
            self.buf = PrioritizedReplayBuffer(2000, expected_steps=1000)
            self.eps = 0.30
            self.step = 0

    def get_action_spec(self, action: int) -> dict:
        """Maps discrete action mode (0..7) to MILP configuration parameters."""
        specs = {
            0: {"p_vehicle": 100.0, "max_time": 4.0, "route_filter": "fleet_top25"},
            1: {"p_vehicle": 200.0, "max_time": 6.0, "route_filter": "fleet_top10"},
            2: {"p_vehicle": 0.0, "max_time": 3.0, "route_filter": "cost_top25"},
            3: {"p_vehicle": 20.0, "max_time": 2.0, "route_filter": "balanced_top50"},
            4: {"p_vehicle": 30.0, "max_time": 5.0, "route_filter": "balanced_top50"},
            5: {"p_vehicle": 15.0, "max_time": 3.0, "route_filter": "diversity_sample"},
            6: {"p_vehicle": 10.0, "max_time": 1.0, "route_filter": "fast_top20"},
            7: {"p_vehicle": 50.0, "max_time": 8.0, "route_filter": "all_routes"},
        }
        return specs.get(action, specs[0])

    def act(self, state: np.ndarray, frozen: bool = False) -> int:
        if not self.use_rl or frozen:
            return 0  # Default Fleet-Min mode
        if random.random() < self.eps:
            return random.randint(0, self.N_ACTIONS - 1)
        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.q(s_t)[0].cpu().numpy()
            return int(q.argmax())

    def observe(self, s, a, r, ns, done=0.0):
        if self.use_rl:
            self.buf.push(s, a, r, ns, done)

    def train_step(self) -> None:
        if not self.use_rl or len(self.buf) < 32:
            return
        (s, a, r, ns, d), idxs, is_w = self.buf.sample(32)
        s = torch.tensor(s).to(DEVICE)
        a = torch.tensor(a, dtype=torch.long).to(DEVICE)
        r = torch.tensor(r).to(DEVICE)
        ns = torch.tensor(ns).to(DEVICE)
        d = torch.tensor(d).to(DEVICE)
        qp = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.q(ns).argmax(1).unsqueeze(1)
            qn = self.q_t(ns).gather(1, best_a).squeeze(1)
            target = r + 0.95 * qn * (1 - d)
        loss = (is_w * F.smooth_l1_loss(qp, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.eps = max(0.05, self.eps * 0.995)

