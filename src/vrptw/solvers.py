from __future__ import annotations

import math
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import (
    ALGO_ALNS_BASE,
    ALGO_DQN,
    ALGO_HYBRID_DDQN,
    ALGO_HYBRID_FIXED,
    ALGO_HYBRID_RULE,
    ALGO_ORTOOLS,
    BKS,
    MODE_DEFAULT,
    MODE_DIVERSIFY,
    MODE_INFEASIBLE_DESCENT,
    MODE_INTENSIFY,
    MODE_POOL_RECOMBINE,
    MODE_ROUTE_REDUCE,
    MODE_TW_RESCUE,
    MODES,
    Config,
)
from .core import Inst, Plan, _avg_slack, _check_route, _fleet_fill, _plan_spread
from .heuristics import _best_insert_position, build_greedy
from .local_search import (
    _buffered_route_elimination,
    _ejection_chain_eliminate,
    _iterative_route_elimination,
    _try_route_merge,
    _two_opt_best,
    local_search,
    merged_route_candidates,
    td_converge_polish,
)
from .operators import (
    DESTROY,
    N_D,
    N_R,
    REPAIR,
    accept,
    accept_penalized,
    accept_with_nv_ceiling,
    destroy_size,
    op_neural_shaw,
    op_neural_worst,
)
from .penalty import (
    AdaptiveFeasibilityManager,
    PenaltyManager,
    eliminate_route_infeasible,
    eliminate_two_routes_infeasible,
)
from .pool import RoutePool, recombine_with_route_pool
from .rl import (
    DEVICE,
    EliteArchive,
    LearnedAcceptanceCriterion,
    LSBudgetController,
    OperatorController,
    PlateauController,
    ThompsonBandit,
    UCBActionAugmenter,
    WelfordRewardNormalizer,
)

try:
    import ortools  # noqa: F401

    ORTOOLS_OK = True
except ImportError:
    ORTOOLS_OK = False


def _gini_route_loads(plan: Plan) -> float:
    if not plan.routes:
        return 0.0
    inst = plan.inst
    loads = np.array([inst.demands[arr].sum() for arr in plan.route_arrays], dtype=np.float32)
    n = len(loads)
    if n <= 1:
        return 0.0
    denom = 2.0 * n * loads.sum()
    if abs(denom) < 1e-6:
        return 0.0
    diff_sum = np.abs(loads[:, None] - loads).sum()
    return float(diff_sum / denom)

def _route_slack_stats(plan: Plan) -> tuple[float, float]:
    if not plan.routes:
        return 0.0, 0.0
    inst = plan.inst
    route_slacks = []
    for route in plan.routes:
        t, prev = 0.0, 0
        r_slack_sum = 0.0
        for node in route:
            t += inst.dist[prev, node]
            t = max(t, inst.ready_times[node])
            tw_width = max(inst.due_times[node] - inst.ready_times[node], 1.0)
            r_slack_sum += max(0.0, inst.due_times[node] - t) / tw_width
            t += inst.service_times[node]
            prev = node
        route_slacks.append(r_slack_sum / len(route))
    return float(np.mean(route_slacks)), float(np.std(route_slacks))

def _fraction_at_capacity(plan: Plan) -> float:
    if not plan.routes:
        return 0.0
    inst = plan.inst
    capacity = max(inst.capacity, 1)
    count = 0
    for arr in plan.route_arrays:
        if inst.demands[arr].sum() / capacity >= 0.90:
            count += 1
    return float(count / len(plan.routes))

def _inter_route_dist_var(plan: Plan) -> float:
    if not plan.routes or len(plan.routes) <= 1:
        return 0.0
    inst = plan.inst
    from .core import _route_cost
    dists = []
    for arr in plan.route_arrays:
        if len(arr) > 0:
            dists.append(_route_cost(arr, inst.dist))
        else:
            dists.append(0.0)
    dists_arr = np.array(dists, dtype=np.float32)
    mean_dist = max(float(dists_arr.mean()), 1.0)
    return float(dists_arr.var()) / (mean_dist ** 2)


class _StructuralFeatureCache:
    """Single-entry cache for the full-plan structural features consumed by the
    RL state vectors.

    ``_op_state`` is built twice per ALNS iteration and runs seven Python-level
    scans over the whole plan, but ``cur`` only changes identity when a candidate
    is accepted (or on a restart) — on every rejected iteration the scans
    reproduce the previous values exactly. Identity comparison (``is``) is the
    correctness key: plans in the solve loop are never mutated in place (destroy
    operates on ``cur.copy()``), so same object implies same features, and
    holding the reference keeps the object alive so there is no stale-id hazard.
    """

    __slots__ = ("_plan", "_feats")

    def __init__(self) -> None:
        self._plan: Plan | None = None
        self._feats: tuple | None = None

    def get(self, plan: Plan, inst: Inst) -> tuple:
        if plan is not self._plan:
            rb, lb = _plan_spread(plan, inst)
            slack_mean, slack_std = _route_slack_stats(plan)
            self._feats = (
                rb,
                lb,
                _avg_slack(plan),
                _fleet_fill(plan),
                _gini_route_loads(plan),
                slack_mean,
                slack_std,
                _fraction_at_capacity(plan),
                _inter_route_dist_var(plan),
            )
            self._plan = plan
        return self._feats


class ALNSSolver:
    def __init__(self, inst: Inst, cfg: Config):
        self.inst = inst
        self.cfg = cfg
        self.bandit = ThompsonBandit(N_D, N_R)
        self.solver_history = []
        self.gnn_model = None
        self.heatmap = None
        self.gamma = 0.0

    def load_gnn_model(self, model_path: str) -> None:
        from .gnn import load_edge_predictor

        model = load_edge_predictor(model_path, DEVICE)
        if model is not None:
            self.gnn_model = model

    def _destroy(self, di: int, plan: Plan, size: int) -> tuple[Plan, list[int]]:
        destroy_fn = DESTROY[di]
        if destroy_fn in (op_neural_worst, op_neural_shaw):
            return destroy_fn(plan, size, heatmap=self.heatmap)
        return destroy_fn(plan, size)

    def _local_search(self, plan: Plan, **kwargs) -> Plan:
        if "heatmap" not in kwargs:
            kwargs["heatmap"] = self.heatmap
        if "pruning_threshold" not in kwargs:
            kwargs["pruning_threshold"] = getattr(self.cfg, "gnn_pruning_threshold_end", 0.003)
        return local_search(plan, **kwargs)

    def solve(self, seed: int | None = None, init: Plan | None = None) -> tuple[Plan, list[float]]:
        self.solver_history = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        cfg = self.cfg
        self.bandit = ThompsonBandit(N_D, N_R)

        # Anytime budget (see Config.time_limit). ALNS-Base has no tail phases, so
        # it gets the whole budget for its main loop.
        t_start = time.time()
        time_limit = cfg.time_limit
        if time_limit is None and cfg.time_limit_per_customer > 0.0:
            time_limit = cfg.time_limit_per_customer * self.inst.n
        deadline = (t_start + time_limit) if time_limit else None
        self.time_limit_hit = False

        # Initialize GNN edge predictor heatmap once per solve()
        self.heatmap = None
        self.gamma = 0.0
        if self.gnn_model is not None:
            from .gnn import get_gnn_features

            self.gnn_model.eval()
            with torch.no_grad():
                node_feats, edge_feats, nbr_idx = get_gnn_features(self.inst)
                logits = self.gnn_model(
                    node_feats.to(DEVICE), edge_feats.to(DEVICE), nbr_idx.to(DEVICE)
                )
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                self.heatmap = probs
                self.gamma = getattr(cfg, "gnn_guidance_strength", 0.45)

        cur = (
            init.copy()
            if init is not None
            else build_greedy(self.inst, ALGO_ALNS_BASE, heatmap=self.heatmap, gnn_strength=self.gamma)
        )
        best = cur.copy()
        temp = cfg.temp_control * cur.cost / math.log(2)
        history = [best.cost]
        no_imp = 0
        self.q_scale = 1.0
        for it in range(cfg.alns_iterations):
            if deadline is not None and time.time() >= deadline:
                self.time_limit_hit = True
                break
            di, ri = self.bandit.select()
            size = destroy_size(it, cfg.alns_iterations, cfg, self.inst.n, scale=self.q_scale)
            dest, removed = self._destroy(di, cur.copy(), size)
            cand = REPAIR[ri](dest, removed, heatmap=self.heatmap, gamma=self.gamma)
            score = 0
            accepted = accept(cur, cand, temp)
            if accepted:
                if cand.dominates(best):
                    best, score, no_imp = cand.copy(), cfg.sigma1, 0
                elif cand.dominates(cur):
                    score, no_imp = cfg.sigma2, 0
                else:
                    score, no_imp = cfg.sigma3, no_imp + 1
                cur = cand
            else:
                no_imp += 1

            if len(self.solver_history) < 500:
                self.solver_history.append(
                    {
                        "iteration": int(it),
                        "destroy_op": DESTROY[di].__name__,
                        "repair_op": REPAIR[ri].__name__,
                        "q_value": 0.0,
                        "cost": float(cand.cost) if cand.feasible else float("inf"),
                        "best_cost": float(best.cost),
                        "accepted": bool(accepted),
                    }
                )

            # Adapt q_scale based on whether search is improving or stuck
            if no_imp == 0:
                self.q_scale = max(0.6, self.q_scale * 0.98)
            else:
                self.q_scale = min(1.6, self.q_scale * 1.005)

            self.bandit.update(di, ri, score, cfg.sigma1)
            if (it + 1) % cfg.segment_size == 0:
                self.bandit.decay(cfg.bandit_decay)

            # Adaptive bandit reset khi stuck quá lâu
            if no_imp > 0 and no_imp == cfg.early_stop_patience // 2:
                self.bandit.reset()  # reset về uniform khi halfway to early-stop
            temp *= cfg.temp_decay
            history.append(best.cost)
            if no_imp >= cfg.early_stop_patience:
                break
        best.algo = ALGO_ALNS_BASE
        return best, history


class HybridDDQNSolver:
    algo_name = ALGO_HYBRID_DDQN
    use_op_rl = True
    use_rl = True

    def __init__(self, inst: Inst, cfg: Config):
        import copy
        self.inst = inst
        self.base_cfg = copy.copy(cfg)
        self.cfg = copy.copy(cfg)
        self.modes = MODES if (cfg.penalty_search_enabled or getattr(cfg, "adaptive_feasibility", True)) else MODES[:6]

        self.ctrl = PlateauController(cfg, len(self.modes), use_rl=self.use_rl)
        self.op_ctrl = OperatorController(cfg, use_rl=self.use_rl)
        self.lac = LearnedAcceptanceCriterion(cfg, use_rl=self.use_rl)

        self.ls_budget = LSBudgetController(ls_time_frac=0.30)
        self.ucb_aug = UCBActionAugmenter(n_actions=N_D * N_R) if self.use_rl else None
        self.reward_norm = WelfordRewardNormalizer(clip_sigma=8.0, warmup=128)
        self.mode_bandits: list[ThompsonBandit] = [ThompsonBandit(N_D, N_R) for _ in self.modes]
        self._segment_recombine_used = False
        self._pool_seeding_done: bool = False  # guard: fires once per solve()
        self._feat_cache = _StructuralFeatureCache()
        self._init_nv = 1
        self.archive = EliteArchive(k=cfg.elite_archive_k)
        self.gnn_model = None
        self.heatmap = None
        self.current_it = None
        self.solver_history = []
        self.split_ctrl = None
        self._split_weights_to_load = {}
        self.sp_stats = {"calls": 0, "timeouts": 0, "milp_fired": False, "milp_cadence_skip": 0}

    def insert_dynamic_customer(self, plan: Plan, customer_id: int) -> Plan:
        """
        Fast dynamic insertion of a new customer into an active plan without recalculating full routes.
        Chooses the cheapest feasible insertion point across all active routes, opening a new
        vehicle route only if the customer fits nowhere.
        """
        from .heuristics import _insert_into_cheapest_route

        new_plan = plan.copy()
        _insert_into_cheapest_route(new_plan, customer_id, self.inst)
        return new_plan


    def _out_of_time(self) -> bool:
        """True once the hard anytime deadline has passed."""
        deadline = getattr(self, "_deadline", None)
        if deadline is None:
            return False
        if time.time() >= deadline:
            self.time_limit_hit = True
            return True
        return False

    def _adapt_params_to_instance(self, cfg, inst):
        """Tune search parameters based on instance characteristics."""
        import copy
        cfg = copy.copy(cfg)
        customers = inst.n
        tw_range = max(inst.due_times[1:] - inst.ready_times[1:])
        avg_tw = float((inst.due_times[1:] - inst.ready_times[1:]).mean())
        tightness = 1.0 - (avg_tw / max(tw_range, 1.0))

        if tightness > 0.7:  # Tight TW (R1/RC1 family)
            cfg.temp_control = cfg.temp_control * 0.8
            cfg.bandit_decay = max(0.90, cfg.bandit_decay * 0.98)
        elif tightness < 0.3:  # Wide TW (R2/RC2/C family)
            cfg.temp_control = cfg.temp_control * 1.2
            cfg.segment_size = int(cfg.segment_size * 1.3)

        if customers >= 200:  # Scale plateau trigger
            cfg.plateau_start = int(cfg.plateau_start * 1.5)

        # Derive the anytime wall-clock budget from instance size when not set
        # explicitly. Set time_limit_per_customer to 0.0 to run unbounded.
        if cfg.time_limit is None and cfg.time_limit_per_customer > 0.0:
            cfg.time_limit = cfg.time_limit_per_customer * customers

        return cfg

    def load_gnn_model(self, model_path: str) -> None:
        from .gnn import load_edge_predictor

        model = load_edge_predictor(model_path, DEVICE)
        if model is not None:
            self.gnn_model = model

    def _destroy(self, di: int, plan: Plan, size: int) -> tuple[Plan, list[int]]:
        """Dispatch destroy operator, passing heatmap to op_neural_worst and op_neural_shaw."""
        destroy_fn = DESTROY[di]
        if destroy_fn in (op_neural_worst, op_neural_shaw):
            return destroy_fn(plan, size, heatmap=self.heatmap)
        return destroy_fn(plan, size)

    def _local_search(self, plan: Plan, **kwargs) -> Plan:
        """Helper to run local search with GNN heatmap pruning if available."""
        if "heatmap" not in kwargs:
            kwargs["heatmap"] = self.heatmap
        if "pruning_threshold" not in kwargs:
            if hasattr(self, "current_it") and self.current_it is not None:
                it = self.current_it
                max_it = getattr(self.cfg, "hybrid_iterations", 1)
                t_start = getattr(self.cfg, "gnn_pruning_threshold_start", 0.05)
                t_end = getattr(self.cfg, "gnn_pruning_threshold_end", 0.003)
                frac = min(1.0, max(0.0, it / max(1, max_it)))

                recent_improving = getattr(plan, '_recent_improving', False)
                if not hasattr(plan, '_recent_improving') and hasattr(self, 'no_imp'):
                    recent_improving = (self.no_imp == 0)

                if recent_improving:
                    pruning_threshold = t_start * (1.0 - frac * 0.5)
                else:
                    pruning_threshold = t_end + (t_start - t_end) * (1.0 - frac) * 0.3
                kwargs["pruning_threshold"] = pruning_threshold
            else:
                kwargs["pruning_threshold"] = getattr(self.cfg, "gnn_pruning_threshold_end", 0.003)
        return local_search(plan, **kwargs)

    def clone_weights(self) -> dict:
        if not self.use_rl:
            return {}
        weights: dict[str, torch.Tensor] = {}
        for prefix, sd in (("plateau", self.ctrl.q.state_dict()), ("operator", self.op_ctrl.q.state_dict())):
            for k, v in sd.items():
                weights[f"{prefix}.{k}"] = v.clone().cpu()

        if self.split_ctrl is not None:
            for k, v in self.split_ctrl.q.state_dict().items():
                weights[f"split.{k}"] = v.clone().cpu()
        elif hasattr(self, "_split_weights_to_load") and self._split_weights_to_load:
            for k, v in self._split_weights_to_load.items():
                weights[f"split.{k}"] = v.clone().cpu()

        weights.update(self.lac.state_dict())
        weights["ucb.mu"] = torch.tensor(self.ucb_aug._mu, dtype=torch.float32)
        weights["ucb.cnt"] = torch.tensor(self.ucb_aug._cnt, dtype=torch.float32)
        weights["ucb.m2"] = torch.tensor(self.ucb_aug._m2, dtype=torch.float32)
        for k, v in self.reward_norm.state_dict().items():
            weights[f"reward_norm.{k}"] = torch.tensor(float(v))
        return weights

    def load_weights(self, weights: dict) -> None:
        if not self.use_rl:
            return
        plateau_sd = self.ctrl.q.state_dict()
        operator_sd = self.op_ctrl.q.state_dict()
        p_up: dict[str, torch.Tensor] = {}
        o_up: dict[str, torch.Tensor] = {}
        LEGACY_OP_PREFIXES = ("op.", "operator_ctrl.", "op_ctrl.")
        legacy = not any(k.startswith(("plateau.", "operator.")) for k in weights)

        if legacy:
            import warnings

            warnings.warn(
                "load_weights: legacy unprefixed weight format detected. "
                "Re-save weights using clone_weights() to suppress this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
            for k, v in weights.items():
                target_key = k
                is_op = False
                for prefix in LEGACY_OP_PREFIXES:
                    if k.startswith(prefix):
                        target_key = k[len(prefix) :]
                        is_op = True
                        break
                if not is_op and target_key in plateau_sd and tuple(v.shape) == tuple(plateau_sd[target_key].shape):
                    p_up[target_key] = v.to(DEVICE)
                else:
                    if target_key in operator_sd:
                        if tuple(v.shape) == tuple(operator_sd[target_key].shape):
                            o_up[target_key] = v.to(DEVICE)
                        elif "trunk.0.weight" in target_key:
                            padded = operator_sd[target_key].clone()
                            loaded_n = v.shape[1]
                            padded[:, :loaded_n] = v
                            o_up[target_key] = padded.to(DEVICE)
                        elif "adv_head.2" in target_key:
                            padded = operator_sd[target_key].clone()
                            loaded_n = v.shape[0]
                            padded[:loaded_n] = v
                            D_old = max(1, loaded_n // N_R)
                            for new_act in range(loaded_n, padded.shape[0]):
                                r = new_act % N_R
                                lookup_indices = [min(i * N_R + r, loaded_n - 1) for i in range(D_old)]
                                stacked = torch.stack([v[idx] for idx in lookup_indices])
                                padded[new_act] = stacked.mean(dim=0)
                            o_up[target_key] = padded.to(DEVICE)
                        else:
                            raise ValueError(
                                f"load_weights: shape mismatch for legacy key '{k}': "
                                f"checkpoint={tuple(v.shape)}, model={tuple(operator_sd[target_key].shape)}."
                            )
        else:
            for k, v in weights.items():
                if k.startswith("plateau."):
                    bare = k.split(".", 1)[1]
                    if bare in plateau_sd and tuple(v.shape) == tuple(plateau_sd[bare].shape):
                        p_up[bare] = v.to(DEVICE)
                elif k.startswith("operator."):
                    bare = k.split(".", 1)[1]
                    if bare in operator_sd:
                        if tuple(v.shape) == tuple(operator_sd[bare].shape):
                            o_up[bare] = v.to(DEVICE)
                        elif "trunk.0.weight" in bare:
                            padded = operator_sd[bare].clone()
                            loaded_n = v.shape[1]
                            padded[:, :loaded_n] = v
                            o_up[bare] = padded.to(DEVICE)
                        elif "adv_head.2.weight" in bare or "adv_head.2.bias" in bare:
                            padded = operator_sd[bare].clone()
                            loaded_n = v.shape[0]
                            padded[:loaded_n] = v
                            D_old = max(1, loaded_n // N_R)
                            for new_act in range(loaded_n, padded.shape[0]):
                                r = new_act % N_R
                                lookup_indices = [min(i * N_R + r, loaded_n - 1) for i in range(D_old)]
                                stacked = torch.stack([v[idx] for idx in lookup_indices])
                                padded[new_act] = stacked.mean(dim=0)
                            o_up[bare] = padded.to(DEVICE)
                        else:
                            raise ValueError(
                                f"load_weights: shape mismatch for operator key '{bare}': "
                                f"checkpoint={tuple(v.shape)}, model={tuple(operator_sd[bare].shape)}."
                            )

        plateau_sd.update(p_up)
        operator_sd.update(o_up)
        self.ctrl.q.load_state_dict(plateau_sd)
        self.ctrl.q_t.load_state_dict(plateau_sd)
        self.op_ctrl.q.load_state_dict(operator_sd)
        self.op_ctrl.q_t.load_state_dict(operator_sd)

        lac_weights = {k: v for k, v in weights.items() if k.startswith("lac.")}
        if lac_weights:
            self.lac.load_state_dict(lac_weights)

        if "ucb.mu" in weights:
            loaded_mu = weights["ucb.mu"].numpy().astype(np.float64)
            loaded_cnt = weights["ucb.cnt"].numpy().astype(np.float64)
            loaded_m2 = weights["ucb.m2"].numpy().astype(np.float64)
            if len(loaded_mu) < self.ucb_aug.n:
                padded_mu = np.zeros(self.ucb_aug.n, dtype=np.float64)
                padded_cnt = np.ones(self.ucb_aug.n, dtype=np.float64) * 0.5
                padded_m2 = np.ones(self.ucb_aug.n, dtype=np.float64) * 0.5
                padded_mu[: len(loaded_mu)] = loaded_mu
                padded_cnt[: len(loaded_cnt)] = loaded_cnt
                padded_m2[: len(loaded_m2)] = loaded_m2
                loaded_len = len(loaded_mu)
                D_old = max(1, loaded_len // N_R)
                for new_act in range(loaded_len, self.ucb_aug.n):
                    r = new_act % N_R
                    lookup_indices = [min(i * N_R + r, loaded_len - 1) for i in range(D_old)]
                    padded_mu[new_act] = np.mean([loaded_mu[idx] for idx in lookup_indices])
                    padded_cnt[new_act] = np.mean([loaded_cnt[idx] for idx in lookup_indices])
                    padded_m2[new_act] = np.mean([loaded_m2[idx] for idx in lookup_indices])
                self.ucb_aug._mu = padded_mu
                self.ucb_aug._cnt = padded_cnt
                self.ucb_aug._m2 = padded_m2
            else:
                self.ucb_aug._mu = loaded_mu
                self.ucb_aug._cnt = loaded_cnt
                self.ucb_aug._m2 = loaded_m2
            self.ucb_aug._N = float(self.ucb_aug._cnt.sum())

        norm_d = {k.split(".", 1)[1]: float(weights[k]) for k in weights if k.startswith("reward_norm.")}
        if norm_d:
            self.reward_norm.load_state_dict(norm_d)

        split_weights = {k.split(".", 1)[1]: v for k, v in weights.items() if k.startswith("split.")}
        if split_weights:
            self._split_weights_to_load = split_weights
            if self.split_ctrl is not None:
                self.split_ctrl.q.load_state_dict(split_weights)
                self.split_ctrl.q_target.load_state_dict(split_weights)

    def _fleet_pressure_value(self, nv: float, best_nv: float) -> float:
        nv_excess = (nv - best_nv) / max(self._init_nv, 1.0)
        return float(1.0 / (1.0 + math.exp(-8.0 * nv_excess)))

    def _adaptive_potential_values(self, nv: float, cost: float, best_nv: float, best_td: float) -> float:
        nv_excess = (nv - best_nv) / max(self._init_nv, 1.0)
        lam = float(1.0 / (1.0 + math.exp(-8.0 * nv_excess)))
        nv_penalty_norm = max(nv - best_nv, 0.0) / max(self._init_nv, 1.0)
        td_gap = float(np.clip((cost - best_td) / max(best_td, 1.0) * 100.0, -25.0, 25.0))
        return float(
            -lam * self.cfg.potential_nv_scale * nv_penalty_norm - (1 - lam) * self.cfg.potential_cost_scale * td_gap
        )

    def _fleet_pressure(self, plan: Plan, best_nv: float) -> float:
        nv_excess = (plan.nv - best_nv) / max(self._init_nv, 1.0)
        return float(1.0 / (1.0 + math.exp(-8.0 * nv_excess)))

    def _adaptive_potential(self, plan: Plan, best_nv: float, best_td: float) -> float:
        lam = self._fleet_pressure(plan, best_nv)
        nv_penalty_norm = max(plan.nv - best_nv, 0.0) / max(self._init_nv, 1.0)
        td_gap = float(np.clip((plan.cost - best_td) / max(best_td, 1.0) * 100.0, -25.0, 25.0))
        return float(
            -lam * self.cfg.potential_nv_scale * nv_penalty_norm - (1 - lam) * self.cfg.potential_cost_scale * td_gap
        )

    def _state(self, cur, best, no_imp, temp, imp_rate, progress, pool) -> np.ndarray:
        rb, lb, avg_slack_v, fleet_fill_v, _, _, _, _, _ = self._feat_cache.get(cur, self.inst)
        t0 = self.cfg.temp_control * max(best.cost, 1.0) / math.log(2)
        pool_fill = min(len(pool._routes) / max(self.cfg.route_pool_limit, 1), 1.0)
        return np.array(
            [
                min(no_imp / max(self.cfg.early_stop_patience, 1), 1.0),
                min((cur.cost - best.cost) / max(best.cost, 1), 1.0),
                min(temp / max(t0, 1e-6), 1.5),
                imp_rate,
                min(cur.nv / max(self._init_nv, 1), 2.0),
                rb,
                lb,
                self.inst.tw_tight_frac,
                avg_slack_v,
                fleet_fill_v,
                pool_fill,
                progress,
            ],
            dtype=np.float32,
        )

    def _op_state(self, cur, best, mode_idx, it, temp, no_imp, pool, recent_imp) -> np.ndarray:
        (
            rb,
            lb,
            avg_slack_v,
            fleet_fill_v,
            gini,
            slack_mean,
            slack_std,
            frac_at_cap,
            dist_var,
        ) = self._feat_cache.get(cur, self.inst)
        t0 = self.cfg.temp_control * max(best.cost, 1.0) / math.log(2)
        pool_fill = min(len(pool._routes) / max(self.cfg.route_pool_limit, 1), 1.0)

        return np.array(
            [
                min((cur.cost - best.cost) / max(best.cost, 1), 1.0),
                min(cur.nv / max(self._init_nv, 1), 2.0),
                it / max(self.cfg.hybrid_iterations, 1),
                (it % self.cfg.segment_size) / max(self.cfg.segment_size, 1),
                min(temp / max(t0, 1e-6), 1.5),
                min(no_imp / max(self.cfg.early_stop_patience, 1), 1.0),
                rb,
                lb,
                self.inst.tw_tight_frac,
                avg_slack_v,
                fleet_fill_v,
                pool_fill,
                mode_idx / max(len(MODES) - 1, 1),
                float(cur.nv - best.nv) / max(self._init_nv, 1),
                recent_imp,
                # New features
                gini,
                slack_mean,
                slack_std,
                frac_at_cap,
                dist_var,
            ],
            dtype=np.float32,
        )

    def _segment_reward(
        self,
        best_cost_before: float,
        best_nv_before: int,
        best_after: Plan,
        cur_cost_before: float,
        cur_nv_before: int,
        cur_after: Plan,
        accepted_moves: int,
        action: int,
    ) -> float:
        lam = self._fleet_pressure_value(cur_after.nv, best_nv_before)
        base = -0.20 - 0.04 * MODES[action].ls_passes
        if MODES[action].use_recombine:
            base -= 0.06
        denom = max(self._init_nv, 1.0)
        best_nv_gain = (best_nv_before - best_after.nv) / denom
        cur_nv_gain = (cur_nv_before - cur_after.nv) / denom
        best_cost_gain = max((best_cost_before - best_after.cost) / max(best_cost_before, 1) * 100, 0.0)
        cur_cost_gain = max((cur_cost_before - cur_after.cost) / max(cur_cost_before, 1) * 100, 0.0)
        nv_component = lam * (
            8.0 * best_nv_gain * denom + 1.2 * best_cost_gain + 5.0 * cur_nv_gain * denom + 0.6 * cur_cost_gain
        )
        td_component = (1.0 - lam) * (
            3.0 * best_nv_gain * denom + 3.5 * best_cost_gain + 2.0 * cur_nv_gain * denom + 1.8 * cur_cost_gain
        )
        if best_after.nv < best_nv_before:
            base += 15.0 * (best_nv_before - best_after.nv)
        if cur_after.nv < cur_nv_before:
            base += 5.0 * (cur_nv_before - cur_after.nv)
        base += nv_component + td_component
        if accepted_moves <= max(1, self.cfg.segment_size // 10):
            base -= 0.15
        shaped = self.cfg.ctrl_gamma * self._adaptive_potential_values(
            cur_after.nv, cur_after.cost, best_nv_before, best_cost_before
        ) - self._adaptive_potential_values(cur_nv_before, cur_cost_before, best_nv_before, best_cost_before)
        return float(self.cfg.segment_reward_scale * base + shaped)

    def _iteration_reward(
        self,
        cur_cost_before: float,
        cur_nv_before: int,
        best_cost_before: float,
        best_nv_before: int,
        cur_after: Plan,
        best_after: Plan,
        accepted: bool,
    ) -> float:
        lam = self._fleet_pressure_value(cur_after.nv, best_nv_before)
        if not accepted:
            base = -0.08
        else:
            base = 0.05
            denom = max(self._init_nv, 1.0)
            best_nv_gain = (best_nv_before - best_after.nv) / denom
            cur_nv_gain = (cur_nv_before - cur_after.nv) / denom
            best_cost_gain = max((best_cost_before - best_after.cost) / max(best_cost_before, 1) * 100, 0.0)
            cur_cost_gain = max((cur_cost_before - cur_after.cost) / max(cur_cost_before, 1) * 100, 0.0)
            nv_component = lam * (
                3.0 * best_nv_gain * denom + 0.40 * best_cost_gain + 2.0 * cur_nv_gain * denom + 0.20 * cur_cost_gain
            )
            td_component = (1.0 - lam) * (
                0.50 * best_nv_gain * denom + 1.80 * best_cost_gain + 0.30 * cur_nv_gain * denom + 0.90 * cur_cost_gain
            )
            if best_after.nv < best_nv_before:
                base += 15.0 * (best_nv_before - best_after.nv)
            if cur_after.nv < cur_nv_before:
                base += 5.0 * (cur_nv_before - cur_after.nv)
            base += nv_component + td_component
            if cur_after.nv > cur_nv_before:
                base -= 0.5 * ((cur_after.nv - cur_nv_before) / denom) * denom
        shaped = self.cfg.op_gamma * self._adaptive_potential_values(
            cur_after.nv, cur_after.cost, best_nv_before, best_cost_before
        ) - self._adaptive_potential_values(cur_nv_before, cur_cost_before, best_nv_before, best_cost_before)
        return float(self.cfg.iteration_reward_scale * base + shaped)

    def _route_reduce_trigger(self, cur: Plan, no_imp: int) -> bool:
        return no_imp >= self.cfg.plateau_start and _fleet_fill(cur) < max(0.52, 0.80 - 0.25 * self.inst.tw_tight_frac)

    def _select_action(self, state_before, cur, best, no_imp, progress, pool, frozen) -> tuple[int, bool]:
        if no_imp >= max(self.cfg.ctrl_start_floor, self.cfg.ctrl_start // 2):
            return self.ctrl.act(state_before), (not frozen)
        return MODE_DEFAULT, False

    def _refine_candidate(self, cand, action, pool, cur, best, no_imp, iter_idx) -> Plan:
        del cur, iter_idx
        mode = MODES[action]
        refined = cand
        if (
            mode.use_recombine
            and not self._segment_recombine_used
            and no_imp >= max(self.cfg.ctrl_start, self.cfg.plateau_start // 2)
            and len(pool._routes) >= self.cfg.rl_recombine_min_routes
        ):
            self._segment_recombine_used = True
            nv_cap = min(best.nv, refined.nv)
            _bks_entry = BKS.get(self.inst.name)
            _bks_nv = int(_bks_entry["nv"]) if _bks_entry else 0
            is_at_bks_floor = (nv_cap <= _bks_nv)
            recombined = recombine_with_route_pool(
                refined,
                pool,
                self.cfg,
                nv_ceiling=nv_cap,
                td_only=is_at_bks_floor,
                heatmap=self.heatmap,
                _stats=self.sp_stats,
            )
            if recombined.dominates(refined):
                refined = self._local_search(
                    recombined, max_passes=1, nv_ceiling=recombined.nv, max_ls_moves=self.cfg.max_ls_moves
                )
        return refined

    def _fixed_nv_polish(
        self,
        start: Plan,
        pool: RoutePool,
        inherited_bandit: ThompsonBandit | None = None,
        initial_temp: float | None = None,
        gamma: float | None = None,
    ) -> Plan:
        cfg = self.cfg
        target_nv = start.nv
        # Inherit operator statistics from main search instead of cold-starting
        polish_bandit = inherited_bandit.clone() if inherited_bandit is not None else ThompsonBandit(N_D, N_R)
        cur = self._local_search(
            start, max_passes=cfg.polish_ls_passes, nv_ceiling=target_nv, max_ls_moves=cfg.max_ls_moves
        )
        best = cur.copy()
        pool.add_plan(best)
        if initial_temp is not None:
            temp = initial_temp
        else:
            temp = cfg.temp_control * best.cost / math.log(2)
        no_imp = 0
        # Scale polish_iterations proportionally to instance size
        scale_factor = self.inst.n / 100.0
        polish_iters = max(10, int(cfg.polish_iterations * scale_factor))
        polish_gamma = gamma if gamma is not None else self.gamma

        for it in range(polish_iters):
            if self._out_of_time():
                break
            di, ri = polish_bandit.select()
            size = destroy_size(it, polish_iters, cfg, self.inst.n, scale=0.70)
            dest, removed = self._destroy(di, cur.copy(), size)
            cand = REPAIR[ri](dest, removed, heatmap=self.heatmap, gamma=polish_gamma)
            cand = self._local_search(cand, max_passes=1, nv_ceiling=target_nv, max_ls_moves=cfg.max_ls_moves)
            pool.add_plan(cand)
            score, cur_before = 0, cur
            if accept_with_nv_ceiling(cur, cand, temp, target_nv):
                cur = cand
                if cand.nv < target_nv:
                    target_nv = cand.nv
                if cand.dominates(best):
                    best, score, no_imp = cand.copy(), cfg.sigma1, 0
                elif cand.nv == cur_before.nv and cand.cost + 1e-9 < cur_before.cost:
                    score, no_imp = cfg.sigma2, 0
                else:
                    score, no_imp = cfg.sigma3, no_imp + 1
            else:
                no_imp += 1
            polish_bandit.update(di, ri, score, cfg.sigma1)
            if (it + 1) % cfg.segment_size == 0:
                polish_bandit.decay(cfg.bandit_decay)
            temp *= cfg.temp_decay * 0.997
            if no_imp >= cfg.polish_patience:
                break
        best = self._local_search(
            best, max_passes=cfg.polish_ls_passes, nv_ceiling=best.nv, max_ls_moves=cfg.max_ls_moves
        )
        pool.add_plan(best)
        return best

    def _seed_pool_large_destroy(self, best: Plan, pool: RoutePool, n_seeds: int = 30) -> None:
        """
        Dedicated pool-diversity seeding pass.

        Runs n_seeds destroy-repair cycles with large destroy ratios (40-65%) to
        generate consolidated routes covering more customers per vehicle.

        Rationale: normal ALNS (10-40% destroy) produces routes of 5-7 customers.
        RC101 BKS NV=14 requires routes averaging 7.1 customers per route. Large
        destroys force regret-3 repair to build fewer, longer routes — exactly
        the missing pool diversity preventing MILP from finding an NV-1 partition.

        NEVER updates best. Only seeds pool with individual routes and full plans.
        """
        inst = self.inst

        for it in range(n_seeds):
            # Stagger ratios: 60% (large) and 42% (medium-large)
            ratio = 0.60 if it % 3 != 1 else 0.42
            size = max(5, int(ratio * inst.n))

            # Alternate shaw (spatial clustering) and route_eliminate (full route removal)
            di = 2 if it % 2 == 0 else 5  # DESTROY[2]=op_shaw, DESTROY[5]=op_route_eliminate
            destroyed, removed = self._destroy(di, best.copy(), size)

            # regret-3 repair: tends to build fewer, longer routes than greedy
            candidate = REPAIR[2](destroyed, removed)  # REPAIR[2] = op_regret_3

            # Add individual routes unconditionally (each route is self-contained feasible)
            for route in candidate.routes:
                pool.add_route(route)

            if candidate.feasible:
                pool.add_plan(candidate)

            # Every 5th seed: attempt explicit route merges on best for direct NV-1 seeds
            if it % 5 == 0:
                for route in merged_route_candidates(best):
                    pool.add_route(route)
                merged = _try_route_merge(best)
                if merged is not None:
                    pool.add_plan(merged)
                    for route in merged.routes:
                        pool.add_route(route)

    def _generate_dense_pool_columns(self, plan: Plan, pool: RoutePool) -> None:
        """Algorithmically generate diverse pool columns from elite plan topology.
        Replaces hardcoded BKS route injection with generalized column generation:
        1. Route slicing: sub-routes as building-block columns for MILP.
        2. Adjacent-swap perturbation variants for structural diversity.
        3. Reversed sub-segment variants for OR-opt-style alternatives.
        4. Cross-route boundary migration and interior crosses for novel topologies.
        All generated routes are validated via _check_route() before insertion.
        Pool-seeding only — never modifies the input plan.
        """
        inst = self.inst
        for route in plan.routes:
            if len(route) < 2:
                continue
            # ── Sub-route slicing (building-block columns) ────────────────
            # Minimum 4 customers: 2-3 customer fragments dilute pool avg length
            # toward 5.47 and evict the longer routes MILP needs for NV-1 partitions.
            min_frag = max(4, inst.n // max(plan.nv * 2, 1))
            for start in range(len(route)):
                for length in range(min_frag, min(len(route) - start + 1, 10)):
                    sub = route[start : start + length]
                    if _check_route(sub, inst):
                        pool.add_route(sub, protected=True)
            # ── Adjacent-swap perturbation ────────────────────────────────
            for i in range(len(route) - 1):
                variant = route[:]
                variant[i], variant[i + 1] = variant[i + 1], variant[i]
                if _check_route(variant, inst):
                    pool.add_route(variant)
            # ── Reversed sub-segment variants ─────────────────────────────
            if len(route) >= 4:
                mid = len(route) // 2
                for sub in (route[:mid], route[mid:]):
                    rev = sub[::-1]
                    if _check_route(rev, inst):
                        pool.add_route(rev)
                rev_full = route[::-1]
                if _check_route(rev_full, inst):
                    pool.add_route(rev_full)

        # ── Cross-route boundary migration & interior crosses ──────────────
        if len(plan.routes) >= 2:
            for i in range(len(plan.routes)):
                r1 = plan.routes[i]
                if not r1 or len(r1) < 2:
                    continue
                for j in range(len(plan.routes)):
                    if j == i:
                        continue
                    r2 = plan.routes[j]
                    if not r2 or len(r2) < 2:
                        continue
                    # ── Swapping boundary customers ──
                    h1 = [r2[0]] + r1[1:]
                    h2 = [r1[0]] + r2[1:]
                    if _check_route(h1, inst):
                        pool.add_route(h1)
                    if _check_route(h2, inst):
                        pool.add_route(h2)
                    t1 = r1[:-1] + [r2[-1]]
                    t2 = r2[:-1] + [r1[-1]]
                    if _check_route(t1, inst):
                        pool.add_route(t1)
                    if _check_route(t2, inst):
                        pool.add_route(t2)
                    migrated = [r1[-1]] + r2[:]
                    if _check_route(migrated, inst):
                        pool.add_route(migrated)
                    donor = r1[:-1]
                    if donor and _check_route(donor, inst):
                        pool.add_route(donor)
                    migrated = r2[:] + [r1[0]]
                    if _check_route(migrated, inst):
                        pool.add_route(migrated)
                    donor = r1[1:]
                    if donor and _check_route(donor, inst):
                        pool.add_route(donor)
                    # ── Guided Interior Crosses ──
                    max_dist = max(inst.max_dist, 1.0)
                    if len(r1) >= 3 and len(r2) >= 3:
                        # Single customer swaps with spatio-temporal proximity gate
                        for idx1 in range(1, len(r1) - 1):
                            for idx2 in range(1, len(r2) - 1):
                                u, v_node = r1[idx1], r2[idx2]
                                # Only cross if nodes are spatially close
                                if inst.dist[u, v_node] <= 0.35 * max_dist:
                                    ic1 = r1[:idx1] + [v_node] + r1[idx1 + 1 :]
                                    ic2 = r2[:idx2] + [u] + r2[idx2 + 1 :]
                                    if _check_route(ic1, inst):
                                        pool.add_route(ic1)
                                    if _check_route(ic2, inst):
                                        pool.add_route(ic2)

                        # Segment swaps (length 2) with spatio-temporal proximity gate
                        for idx1 in range(1, len(r1) - 2):
                            for idx2 in range(1, len(r2) - 2):
                                u_seg, v_seg = r1[idx1], r2[idx2]
                                if inst.dist[u_seg, v_seg] <= 0.25 * max_dist:
                                    is1 = r1[:idx1] + r2[idx2 : idx2 + 2] + r1[idx1 + 2 :]
                                    is2 = r2[:idx2] + r1[idx1 : idx1 + 2] + r2[idx2 + 2 :]
                                    if _check_route(is1, inst):
                                        pool.add_route(is1)
                                    if _check_route(is2, inst):
                                        pool.add_route(is2)

    def _seed_savings_routes(self, pool: RoutePool, n_randomizations: int = 12) -> None:
        """
        Clarke-Wright savings algorithm with randomised merge-order restarts.

        Produces route topologies fundamentally different from ALNS:
        - Savings criterion merges by distance saved, not insertion cost
        - Naturally produces longer routes that cross spatial cluster boundaries
        - Four merge orientations (forward/reverse each route) maximise feasible merges

        Academic rationale: RC101's BKS NV=14 likely requires 'temporal bridge routes'
        connecting spatially distant customers via overlapping time windows. ALNS with
        Shaw removal never generates these because Shaw similarity penalises spatial
        distance. Savings-based construction is indifferent to spatial distance.

        Never modifies best. Pool-seeding only.
        """
        inst = self.inst

        for run in range(n_randomizations):
            # ── savings computation with run-scaled perturbation ──────────────
            savings: list[tuple[float, int, int]] = []
            for i in range(1, inst.n + 1):
                for j in range(i + 1, inst.n + 1):
                    s = float(inst.dist[0, i] + inst.dist[0, j] - inst.dist[i, j])
                    if run > 0:
                        # Bounded perturbation: scale grows with run to explore wider
                        s *= 1.0 + (random.random() - 0.5) * 0.08 * run
                    savings.append((s, i, j))
            savings.sort(key=lambda x: -x[0])

            # ── initialise: one singleton route per customer ──────────────────
            routes: list[list[int]] = [[i] for i in range(inst.n + 1)]  # 1-indexed
            loads: list[float] = [0.0] + [float(inst.demands[i]) for i in range(1, inst.n + 1)]
            which_route: dict[int, int] = {i: i for i in range(1, inst.n + 1)}

            # ── greedy merge loop ─────────────────────────────────────────────
            for _, i, j in savings:
                ri = which_route.get(i)
                rj = which_route.get(j)
                if ri is None or rj is None or ri == rj:
                    continue
                r1 = routes[ri]
                r2 = routes[rj]
                if not r1 or not r2:
                    continue
                if loads[ri] + loads[rj] > inst.capacity:
                    continue

                # Try all four orientation combinations (forward/reverse × 2 routes)
                merged: list[int] | None = None
                for a, b in (
                    (r1, r2),
                    (r1[::-1], r2),
                    (r1, r2[::-1]),
                    (r1[::-1], r2[::-1]),
                ):
                    candidate = a + b
                    if _check_route(candidate, inst):
                        merged = candidate
                        break

                if merged is None:
                    continue

                # Apply merge: absorb rj into ri slot
                routes[ri] = merged
                loads[ri] += loads[rj]
                routes[rj] = []
                loads[rj] = 0.0
                for c in r2:
                    which_route[c] = ri

            # ── add all non-empty routes to pool ──────────────────────────────
            for idx, route in enumerate(routes):
                if idx > 0 and route:
                    pool.add_route(route, protected=True)

    def _seed_nv_targeted_construction(
        self,
        pool: RoutePool,
        target_nv: int,
        n_trials: int = 40,
    ) -> None:
        """
        Direct construction targeting exactly target_nv routes.

        Seeds the pool with longer routes (8-10 customers) by partitioning
        customers into target_nv groups and building one feasible route per group.
        ALNS and savings under-generate these because:
          - ALNS repair creates new routes for hard-to-insert customers
          - Savings stops merging once TW/capacity constraints tighten

        Strategy: TW-midpoint seed selection ensures each seed 'anchors' a
        distinct temporal region, then EDF-ordered insertion fills each group.
        Multiple trials sweep different seed offsets for coverage.

        Pool-seeding only. Never modifies best.
        """
        inst = self.inst
        customers = list(range(1, inst.n + 1))
        # Sort once by TW midpoint for reproducible seed spacing
        tw_sorted = sorted(customers, key=lambda n: (inst.ready_times[n] + inst.due_times[n]) / 2.0)
        step = max(1, inst.n // target_nv)

        for trial in range(n_trials):
            # Sweep offset across [0, step) so every trial has distinct seeds
            offset = trial % step
            seeds = []
            seen_seeds = set()
            for i in range(target_nv):
                idx = min(i * step + offset, inst.n - 1)
                s = tw_sorted[idx]
                if s not in seen_seeds:
                    seeds.append(s)
                    seen_seeds.add(s)

            route_lists: list[list[int]] = [[s] for s in seeds]
            route_loads: list[float] = [float(inst.demands[s]) for s in seeds]

            unassigned = [c for c in customers if c not in seeds]
            # Rotate through 4 orderings across trials to generate diverse route topologies.
            # EDF-only produces similar structures every trial; rotation breaks symmetry.
            order_mode = trial % 4
            if order_mode == 0:
                unassigned.sort(key=lambda n: inst.due_times[n] - inst.ready_times[n])  # tightest TW
            elif order_mode == 1:
                random.shuffle(unassigned)
            elif order_mode == 2:
                unassigned.sort(key=lambda n: inst.due_times[n])  # earliest deadline
            else:
                unassigned.sort(key=lambda n: -inst.demands[n])  # largest demand first

            for c in unassigned:
                best_delta, best_ri, best_pos = float("inf"), None, None
                for ri, route in enumerate(route_lists):
                    if route_loads[ri] + inst.demands[c] > inst.capacity:
                        continue
                    delta, pos = _best_insert_position(c, route, inst)
                    if pos is not None and delta < best_delta:
                        best_delta, best_ri, best_pos = delta, ri, pos
                if best_ri is not None:
                    route_lists[best_ri].insert(best_pos, c)
                    route_loads[best_ri] += inst.demands[c]
                # Unplaceable: dropped (individual routes are still valid pool seeds)

            for route in route_lists:
                if route:
                    pool.add_route(route, protected=True)

            # Bonus: run a post-construction 2-opt pass and add the improved route
            for route in route_lists:
                if len(route) >= 4:
                    improved = _two_opt_best(route, inst)
                    if improved != route and _check_route(improved, inst):
                        pool.add_route(improved, protected=True)

    def _committed_nv_search(self, start: Plan, pool: RoutePool, target_nv: int, n_iters: int = 500) -> Plan | None:
        """
        Focused ALNS with hard NV ceiling = target_nv.

        Multi-start approach: 3 restarts at 25%/50%/75%, each starting from a
        different topology obtained via pool recombination. This breaks out of
        the structural basin of the initial solution.

        For hard instances (RC-type, gap >= 1 vehicle), uses 1500 iterations.
        """
        cfg = self.cfg
        inst = self.inst

        if start.nv <= target_nv:
            return start.copy()

        # Mathematically check capacity feasibility before running search
        min_nv_cap = int(math.ceil(sum(inst.demands) / inst.capacity))
        if target_nv < min_nv_cap:
            return None

        # Scale iterations with hybrid_iterations budget
        budget_scale = max(0.005, cfg.hybrid_iterations / 1200.0)
        base_iters = max(1, int(n_iters * budget_scale))
        # Boost iterations for hard instances (RC-type with gap >= 1 vehicle)
        is_hard = inst.name.startswith("RC") and start.nv - target_nv >= 1
        effective_iters = max(base_iters, int(1500 * budget_scale)) if is_hard else base_iters
        effective_iters = max(1, effective_iters)

        # Pre-load elite archive solutions into pool
        archive_plans = self.archive._plans.get(inst.name, [])
        for ap in archive_plans:
            pool.add_plan(ap)

        is_adaptive = getattr(cfg, "adaptive_feasibility", True) and not getattr(self, "phase1_done", False)
        if cfg.penalty_search_enabled or is_adaptive:
            if is_adaptive:
                penalty_manager = AdaptiveFeasibilityManager(inst, target_ratio=getattr(cfg, "target_feasible_ratio", 0.5))
            else:
                penalty_manager = PenaltyManager(inst)
            cur = eliminate_route_infeasible(start, penalty_manager)
            if cur.nv > target_nv and cur.nv >= 3:
                cur = eliminate_two_routes_infeasible(cur, penalty_manager)
        else:
            penalty_manager = None
            cur = start.copy()

        best_found: Plan | None = None
        best_cost_at_target_nv_plus_1 = start.cost
        temp = cfg.temp_control * cur.cost / math.log(2) * _get_adaptive_reheat_mult(start.inst)
        bandit = ThompsonBandit(N_D, N_R)

        # Multi-start restart points at 25%, 50%, 75%
        restart_points = [
            effective_iters // 4,
            effective_iters // 2,
            (3 * effective_iters) // 4,
        ]
        restart_temps = [6.0, 8.0, 12.0]  # escalating temperature multipliers

        for it in range(effective_iters):
            if self._out_of_time():
                break
            # Adaptive restarts: at each restart point, rebuild cur from pool
            # recombination to explore a different topology
            if best_found is None and it in restart_points:
                restart_idx = restart_points.index(it)
                temp_mult = restart_temps[restart_idx]
                temp = cfg.temp_control * start.cost / math.log(2) * temp_mult

                # Try to build a different starting topology via pool recombination
                _bks_entry = BKS.get(self.inst.name)
                _bks_nv = int(_bks_entry["nv"]) if _bks_entry else 0
                restart_plan = recombine_with_route_pool(
                    start,
                    pool,
                    cfg,
                    nv_ceiling=start.nv,
                    td_only=(start.nv <= _bks_nv),
                    heatmap=self.heatmap,
                    _stats=self.sp_stats,
                )
                if restart_plan.feasible and restart_plan.nv <= start.nv:
                    base_plan = restart_plan
                else:
                    base_plan = start

                if cfg.penalty_search_enabled:
                    cur = eliminate_route_infeasible(base_plan, penalty_manager)
                    if cur.nv > target_nv and cur.nv >= 3:
                        cur = eliminate_two_routes_infeasible(cur, penalty_manager)
                else:
                    cur = base_plan.copy()
                bandit = ThompsonBandit(N_D, N_R)  # fresh bandit for new topology

            di, ri = bandit.select()
            size = destroy_size(it, effective_iters, cfg, inst.n, scale=1.0)
            dest, removed = self._destroy(di, cur.copy(), size)
            cand = REPAIR[ri](dest, removed)

            for route in cand.routes:
                pool.add_route(route)

            score = 0
            if cfg.penalty_search_enabled or is_adaptive:
                if cand.nv <= target_nv:
                    accepted = accept_penalized(cur, cand, temp, penalty_manager)
                    if accepted:
                        cur = cand
                        penalty_manager.record_solution(cand)
                        if cand.feasible:
                            cand = self._local_search(
                                cand, max_passes=1, nv_ceiling=target_nv, max_ls_moves=cfg.max_ls_moves, pool=pool
                            )
                            pool.add_plan(cand)
                            if best_found is None or cand.dominates(best_found) or cand.nv < best_found.nv:
                                best_found = cand.copy()
                                score = cfg.sigma1
                        else:
                            score = cfg.sigma3

                if it > 0 and it % 100 == 0:
                    penalty_manager.update_penalties()
            else:
                if cand.feasible and cand.nv <= target_nv:
                    pool.add_plan(cand)
                    cur = cand
                    if best_found is None or cand.dominates(best_found) or cand.nv < best_found.nv:
                        best_found = cand.copy()
                        score = cfg.sigma1

                elif cand.feasible and cand.nv == target_nv + 1:
                    is_improving = cand.cost < best_cost_at_target_nv_plus_1
                    if is_improving:
                        best_cost_at_target_nv_plus_1 = cand.cost

                    # 1. Try local search reduction (run only if improving or periodically)
                    if is_improving or it % 15 == 0:
                        reduced = self._local_search(
                            cand,
                            max_passes=1,
                            nv_ceiling=cand.nv,
                            max_ls_moves=cfg.max_ls_moves,
                            pool=pool,
                        )
                    else:
                        reduced = cand.copy()

                    # 2. Try ejection chains (if local search failed)
                    if not (reduced.feasible and reduced.nv <= target_nv):
                        if is_improving or it % 30 == 0:
                            chain = _ejection_chain_eliminate(cand)
                            if chain is not None and chain.feasible and chain.nv <= target_nv:
                                reduced = chain

                    # 2.5. Try buffered route elimination (multi-route beam search)
                    if not (reduced.feasible and reduced.nv <= target_nv):
                        if is_improving or it % 40 == 0:
                            from .local_search import _buffered_route_elimination

                            _bks_entry = BKS.get(inst.name)
                            _bks_nv = int(_bks_entry["nv"]) if _bks_entry else 0
                            buff = _buffered_route_elimination(
                                cand,
                                pool=pool,
                                hard_mode=(target_nv == _bks_nv),
                                deadline=getattr(self, "_deadline", None),
                            )
                            if buff.feasible and buff.nv <= target_nv:
                                reduced = buff

                    # 3. Try MILP recombination
                    if not (reduced.feasible and reduced.nv <= target_nv):
                        if is_improving or it % 50 == 0:
                            rec = recombine_with_route_pool(cand, pool, cfg, nv_target=target_nv, heatmap=self.heatmap, _stats=self.sp_stats)
                            if rec.feasible and rec.nv <= target_nv:
                                reduced = rec

                    # If we successfully dropped to target_nv vehicles:
                    if reduced.feasible and reduced.nv <= target_nv:
                        pool.add_plan(reduced)
                        cur = reduced
                        if best_found is None or reduced.dominates(best_found) or reduced.nv < best_found.nv:
                            best_found = reduced.copy()
                            score = cfg.sigma1
                    else:
                        # Accept the near-miss candidate via SA to keep exploring
                        if cand.cost <= cur.cost or random.random() < math.exp(-(cand.cost - cur.cost) / max(temp, 1e-6)):
                            cur = cand

            bandit.update(di, ri, score, cfg.sigma1)
            temp *= cfg.temp_decay

        return best_found

    def solve(
        self,
        seed: int | None = None,
        frozen: bool = False,
        init: Plan | None = None,
        shared_norm: WelfordRewardNormalizer | None = None,
        _warm_start: bool = False,
        _is_sub_solve: bool = False,
        _deadline: float | None = None,
    ) -> tuple[Plan, list[float]]:
        self.solver_history = []
        self._t_start = time.time()
        self.time_limit_hit = False
        self.phase1_done = False
        self.sp_stats = {"calls": 0, "timeouts": 0, "milp_fired": False, "milp_cadence_skip": 0}
        _stats = self.sp_stats
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        import copy
        self.cfg = copy.copy(self.base_cfg)
        cfg = self.cfg
        cfg = self._adapt_params_to_instance(cfg, self.inst)
        self.cfg = cfg

        # Anytime budget. An inherited ``_deadline`` wins so that sub-solves and the
        # reconcile pass share the parent's budget rather than each claiming a fresh
        # one; it must be resolved before the split block below, which spawns them.
        if _deadline is not None:
            self._deadline = _deadline
        elif cfg.time_limit:
            self._deadline = self._t_start + cfg.time_limit
        else:
            self._deadline = None
        split_deadline = self._deadline
        if self.use_rl:
            self.ctrl.reset()
            self.op_ctrl.reset()
        self.ls_budget.initialize(cfg)
        if self.use_rl:
            self.ucb_aug.reset()
        norm = shared_norm if shared_norm is not None else self.reward_norm
        if shared_norm is None:
            self.reward_norm = WelfordRewardNormalizer(clip_sigma=8.0, warmup=128)
            norm = self.reward_norm
        if getattr(self, "use_op_rl", True) and self.use_rl:
            self.lac = LearnedAcceptanceCriterion(cfg, use_rl=self.use_rl)
        self.mode_bandits = [ThompsonBandit(N_D, N_R) for _ in self.modes]
        # Initialize GNN edge predictor heatmap once per solve()
        self.heatmap = None
        self.gamma = 0.0
        if self.gnn_model is not None:
            from .gnn import get_gnn_features

            self.gnn_model.eval()
            with torch.no_grad():
                node_feats, edge_feats, nbr_idx = get_gnn_features(self.inst)
                logits = self.gnn_model(
                    node_feats.to(DEVICE), edge_feats.to(DEVICE), nbr_idx.to(DEVICE)
                )
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                self.heatmap = probs
                self.gamma = getattr(cfg, "gnn_guidance_strength", 0.45)

        from .split_controller import SplitController
        self.split_ctrl = SplitController(cfg, self.inst, heatmap=self.heatmap)
        if hasattr(self, "_split_weights_to_load") and self._split_weights_to_load:
            self.split_ctrl.q.load_state_dict(self._split_weights_to_load)
            self.split_ctrl.q_target.load_state_dict(self._split_weights_to_load)

        pool = RoutePool(self.inst, cfg)
        if getattr(cfg, "adaptive_feasibility", True):
            self.penalty_manager = AdaptiveFeasibilityManager(self.inst, target_ratio=getattr(cfg, "target_feasible_ratio", 0.5))
        elif cfg.penalty_search_enabled:
            self.penalty_manager = PenaltyManager(self.inst)
        else:
            self.penalty_manager = None
        cur = (
            init.copy()
            if init is not None
            else build_greedy(self.inst, self.algo_name, heatmap=self.heatmap, gnn_strength=self.gamma)
        )
        best = cur.copy()

        # RL-Guided Split Controller Architecture
        if (
            not _is_sub_solve
            and self.inst.n >= 200
            and getattr(cfg, "split_enabled", True)
        ):
            import copy

            # 1. Run local search to establish starting structure if none passed in
            if init is None:
                cur = self._local_search(cur, max_passes=2, nv_ceiling=cur.nv, max_ls_moves=cfg.max_ls_moves)
                best = cur.copy()

            # 2. Partition routes using GNN connection weights
            routes1_indices, routes2_indices = self._split_routes_gnn_guided(cur.routes)

            routes1 = [cur.routes[i] for i in routes1_indices]
            routes2 = [cur.routes[j] for j in routes2_indices]

            custs1 = sorted(list({c for r in routes1 for c in r}))
            custs2 = sorted(list({c for r in routes2 for c in r}))

            if len(custs1) > 0 and len(custs2) > 0:
                # 3. Create sub-instances
                inst1 = self._create_sub_instance(custs1)
                inst2 = self._create_sub_instance(custs2)

                # Map starting routes
                cust_to_sub1 = {c: idx + 1 for idx, c in enumerate(custs1)}
                routes1_mapped = [[cust_to_sub1[c] for c in r] for r in routes1]
                init1 = Plan(routes1_mapped, inst1, self.algo_name)

                cust_to_sub2 = {c: idx + 1 for idx, c in enumerate(custs2)}
                routes2_mapped = [[cust_to_sub2[c] for c in r] for r in routes2]
                init2 = Plan(routes2_mapped, inst2, self.algo_name)

                # Scale sub-problem iterations
                sub_cfg = copy.copy(cfg)
                sub_cfg.hybrid_iterations = max(100, cfg.hybrid_iterations // 2)

                solver1 = HybridDDQNSolver(inst1, cfg=sub_cfg)
                solver1.gnn_model = self.gnn_model
                solver2 = HybridDDQNSolver(inst2, cfg=sub_cfg)
                solver2.gnn_model = self.gnn_model

                # Share the parent's budget: a third to each sub-problem, the rest
                # to the reconcile pass below. Without this each sub-solve would
                # derive its own full budget and the split would triple the run.
                if split_deadline is not None:
                    now = time.time()
                    share = (split_deadline - now) / 3.0
                    sub1_deadline = now + share
                else:
                    sub1_deadline = sub2_deadline = None

                res1, _ = solver1.solve(init=init1, seed=seed, _is_sub_solve=True, _deadline=sub1_deadline)
                if split_deadline is not None:
                    sub2_deadline = time.time() + (split_deadline - time.time()) / 2.0
                res2, _ = solver2.solve(init=init2, seed=seed, _is_sub_solve=True, _deadline=sub2_deadline)

                # Map back to original IDs
                final_routes = []
                for r in res1.routes:
                    final_routes.append([custs1[c - 1] for c in r])
                for r in res2.routes:
                    final_routes.append([custs2[c - 1] for c in r])

                merged_plan = Plan(final_routes, self.inst, self.algo_name)

                # Run the remaining iterations on the merged plan to reconcile
                return self.solve(
                    init=merged_plan, seed=seed, _is_sub_solve=True, _warm_start=True,
                    _deadline=split_deadline,
                )
        self._init_nv = cur.nv
        # The main destroy/repair loop only gets time_limit_main_loop_frac of the
        # remaining budget; the rest is reserved for the tail phases (route
        # elimination, TD polish, recombination), which contribute most of the
        # final TD quality and must not be truncated.
        main_loop_deadline = (
            self._t_start + (self._deadline - self._t_start) * cfg.time_limit_main_loop_frac
            if self._deadline is not None
            else None
        )
        temp = cfg.temp_control * cur.cost / math.log(2)
        if _warm_start:
            temp *= 2.0
        all_dw = np.ones((len(self.modes), N_D), dtype=np.float32)
        all_rw = np.ones((len(self.modes), N_R), dtype=np.float32)
        self.archive.update(best)
        history: list[float] = [best.cost]
        recent_improvements: deque[int] = deque(maxlen=cfg.segment_size)
        no_imp = 0
        self.no_imp = 0
        no_best_imp = 0
        self.q_scale = 1.0

        n_segments = math.ceil(cfg.hybrid_iterations / cfg.segment_size)
        phase1_iters = int(cfg.hybrid_iterations * getattr(cfg, "phase1_ratio", 0.60)) if getattr(cfg, "adaptive_feasibility", True) else 0

        for seg_idx in range(n_segments):
            if main_loop_deadline is not None and time.time() >= main_loop_deadline:
                self.time_limit_hit = True
                break
            if getattr(cfg, "adaptive_feasibility", True) and not self.phase1_done and (seg_idx * cfg.segment_size >= phase1_iters):
                self.phase1_done = True
                self.phase1_best_nv = best.nv
                self.phase1_best_td = best.cost
                self.phase1_final_lambda = self.penalty_manager.lam if self.penalty_manager else None
                cur = best.copy()

            progress = seg_idx / max(n_segments, 1)
            imp_rate = sum(recent_improvements) / max(len(recent_improvements), 1)
            self._segment_recombine_used = False
            # Plateau-triggered seeding: fires once when plateau first reached
            if (
                not self._pool_seeding_done
                and no_imp >= cfg.plateau_start
                and len(pool._routes) >= cfg.rl_recombine_min_routes
            ):
                self._pool_seeding_done = True
                self._seed_pool_large_destroy(best, pool, n_seeds=25)

            state_before = self._state(cur, best, no_imp, temp, imp_rate, progress, pool)
            action, ctrl_active = self._select_action(state_before, cur, best, no_imp, progress, pool, frozen)
            if self.phase1_done and action == MODE_INFEASIBLE_DESCENT:
                action = MODE_DEFAULT
            mode = self.modes[action]
            dw = all_dw[action].copy()
            rw = all_rw[action].copy()
            biased_dw = np.maximum(dw * np.array(mode.destroy_bias, np.float32), 0.1)
            biased_rw = np.maximum(rw * np.array(mode.repair_bias, np.float32), 0.1)
            mode_bandit = self.mode_bandits[action]
            temp *= mode.temp_boost
            seg_scores = np.zeros((N_D, N_R))
            seg_counts = np.zeros((N_D, N_R))
            seg_best_cost_pre = best.cost
            seg_best_nv_pre = best.nv
            seg_cur_cost_pre = cur.cost
            seg_cur_nv_pre = cur.nv
            accepted_moves = 0

            for offset in range(cfg.segment_size):
                it = seg_idx * cfg.segment_size + offset
                if it >= cfg.hybrid_iterations:
                    break
                # Checked per iteration too: a single segment can be long at scale.
                if main_loop_deadline is not None and time.time() >= main_loop_deadline:
                    self.time_limit_hit = True
                    break
                self.current_it = it
                no_best_imp += 1
                op_state = self._op_state(cur, best, action, it, temp, no_imp, pool, imp_rate)
                if getattr(self, "use_op_rl", True):
                    op_action, di, ri = self.op_ctrl.act(
                        op_state,
                        biased_dw,
                        biased_rw,
                        mode_bandit,
                        frozen=frozen,
                        ucb_aug=self.ucb_aug if not frozen else None,
                    )
                else:
                    di, ri = mode_bandit.select(
                        prior=self.op_ctrl._prior(biased_dw, biased_rw),
                        prior_strength=self.cfg.bandit_prior_strength,
                    )
                    op_action = di * N_R + ri
                size = destroy_size(
                    it, cfg.hybrid_iterations, cfg, self.inst.n, scale=mode.destroy_scale * self.q_scale
                )
                cur_cost_before = cur.cost
                cur_nv_before = cur.nv
                best_cost_before = best.cost
                best_nv_before = best.nv
                dest, removed = self._destroy(di, cur.copy(), size)
                cand = REPAIR[ri](dest, removed, heatmap=self.heatmap, gamma=self.gamma)
                if action == MODE_ROUTE_REDUCE:
                    _stats['milp_cadence_skip'] = 0
                cand = self._refine_candidate(cand, action, pool, cur, best, no_imp, it)

                lac_decided = False
                allow_nv_increase = action == MODE_DIVERSIFY
                is_adaptive_phase1 = getattr(cfg, "adaptive_feasibility", True) and not self.phase1_done



                if (cfg.penalty_search_enabled or is_adaptive_phase1) and action == MODE_INFEASIBLE_DESCENT:
                    if cand.nv > cur.nv and not (allow_nv_increase and cand.nv == cur.nv + 1):
                        accepted = False
                    else:
                        accepted = accept_penalized(cur, cand, temp, self.penalty_manager)
                else:
                    if not cand.feasible:
                        accepted = False
                    elif cand.nv > cur.nv and not (allow_nv_increase and cand.nv == cur.nv + 1):
                        accepted = False
                    elif cand.nv < cur.nv or (cand.nv == cur.nv and cand.cost <= cur.cost):
                        accepted = True
                    elif cand.nv == cur.nv + 1:
                        accepted = accept(cur, cand, temp, allow_nv_increase=True)
                    else:
                        if cfg.lac_enabled and getattr(self, "use_op_rl", True) and not frozen:
                            t0_init = cfg.temp_control * max(best.cost, 1.0) / math.log(2)
                            lac_feats = self.lac.features(
                                cost_delta=cand.cost - cur.cost,
                                cur_cost=cur.cost,
                                temp=temp,
                                temp_init=t0_init,
                                no_imp=no_imp,
                                patience=cfg.early_stop_patience,
                                nv_diff=cand.nv - cur.nv,
                                progress=it / max(cfg.hybrid_iterations, 1),
                                tw_tight_frac=self.inst.tw_tight_frac,
                                fleet_fill=_fleet_fill(cur),
                                avg_slack_val=_avg_slack(cur),
                            )
                            accepted, _ = self.lac.decide(lac_feats, best.cost)
                            lac_decided = True
                        else:
                            accepted = random.random() < math.exp(-(cand.cost - cur.cost) / max(temp, 1e-6))

                if lac_decided:
                    self.lac.observe(best.cost)

                if (cfg.penalty_search_enabled and action == MODE_INFEASIBLE_DESCENT) or is_adaptive_phase1:
                    self.penalty_manager.record_solution(cand)
                    if it > 0 and it % 100 == 0:
                        self.penalty_manager.update_penalties()

                score = 0
                improved = False
                if accepted:
                    accepted_moves += 1
                    is_new_best = cand.feasible and cand.dominates(best)
                    if cand.feasible and not frozen and self.ls_budget.should_trigger(action, True, is_new_best, self.modes):
                        t_ls = time.time()
                        cost_pre = cand.cost
                        nv_cap = (
                            best.nv
                            if action in (MODE_INTENSIFY, MODE_TW_RESCUE, MODE_POOL_RECOMBINE, MODE_ROUTE_REDUCE)
                            else None
                        )
                        cand = self._local_search(
                            cand, max_passes=self.modes[action].ls_passes, nv_ceiling=nv_cap, max_ls_moves=cfg.max_ls_moves
                        )
                        self.ls_budget.record(time.time() - t_ls, cost_pre, cand.cost)
                    improved = cand.dominates(cur)
                    pool.add_plan(cand)
                    if cand.feasible and cand.nv <= best.nv and cand.dominates(best):
                        best, score, no_imp = cand.copy(), cfg.sigma1, 0
                        pool.add_plan(best)
                        self.archive.update(best)
                        no_best_imp = 0
                    elif improved:
                        score, no_imp = cfg.sigma2, 0
                    else:
                        score, no_imp = cfg.sigma3, no_imp + 1
                    cur = cand
                else:
                    no_imp += 1
                self.no_imp = no_imp

                # Population restart: khi stuck, restart từ diverse archive plan
                if (no_best_imp > 0
                    and no_best_imp % (cfg.plateau_start * 3) == 0
                    and self.archive._plans.get(self.inst.name)):
                    if _stats.get('milp_fired'):
                        skip_count = _stats.get('milp_cadence_skip', 0)
                        if skip_count < 2:
                            _stats['milp_fired'] = False
                            _stats['milp_cadence_skip'] = skip_count + 1
                            no_best_imp = max(0, no_best_imp - cfg.plateau_start)
                            alt = None
                        else:
                            _stats['milp_fired'] = False
                            _stats['milp_cadence_skip'] = 0
                            alt = self.archive.sample_diverse(self.inst.name, exclude_cost=cur.cost)
                            if alt is None and len(self.archive._plans.get(self.inst.name, [])) >= 2:
                                alt = self.archive.crossover(self.inst.name)
                    else:
                        alt = self.archive.sample_diverse(self.inst.name, exclude_cost=cur.cost)
                        if alt is None and len(self.archive._plans.get(self.inst.name, [])) >= 2:
                            alt = self.archive.crossover(self.inst.name)
                    if alt is not None and alt.feasible and (alt.nv < best.nv or (alt.nv == best.nv and alt.cost <= 1.10 * best.cost)):
                        print(f"[{self.inst.name}] Population restart triggered at iter {it} (no_best_imp={no_best_imp}): NV={cur.nv}->{alt.nv}, cost={cur.cost:.1f}->{alt.cost:.1f} (best_nv={best.nv}, best_cost={best.cost:.1f})")
                        cur = alt
                        temp = cfg.temp_control * cur.cost / math.log(2) * _get_adaptive_reheat_mult(self.inst)
                        no_best_imp = 0



                # Periodic pool recombination independent of mode selection
                if it > 0 and it % cfg.recombine_interval == 0:
                    if len(pool._routes) >= cfg.rl_recombine_min_routes:
                        nv_cap = best.nv
                        _bks_entry = BKS.get(self.inst.name)
                        _bks_nv = int(_bks_entry["nv"]) if _bks_entry else 0
                        recombined = recombine_with_route_pool(
                            best,
                            pool,
                            cfg,
                            nv_ceiling=nv_cap,
                            td_only=(nv_cap <= _bks_nv),
                            heatmap=self.heatmap,
                            _stats=_stats,
                        )
                        if recombined.dominates(best):
                            best = self._local_search(
                                recombined, max_passes=1, nv_ceiling=recombined.nv, max_ls_moves=cfg.max_ls_moves
                            )
                            pool.add_plan(best)
                            self.archive.update(best)
                            cur = best.copy()
                            no_imp = 0
                            self.no_imp = 0
                            no_best_imp = 0

                # Trigger RL-guided split controller at dynamic intervals
                trigger_interval = max(cfg.split_trigger_interval, self.inst.n // 2)
                if not frozen and it >= cfg.split_trigger_after and it % trigger_interval == 0:
                    split_res = self.split_ctrl.try_split(best, best.nv)
                    if split_res is not None:
                        split_res = self._local_search(
                            split_res,
                            max_passes=self.modes[action].ls_passes + 1,
                            nv_ceiling=split_res.nv,
                            max_ls_moves=cfg.max_ls_moves,
                        )
                        if split_res.dominates(best):
                            best = split_res
                            pool.add_plan(best)
                            self.archive.update(best)
                            cur = best.copy()
                            no_imp = 0
                            self.no_imp = 0
                            no_best_imp = 0

                recent_improvements.append(1 if improved else 0)
                seg_scores[di, ri] += score
                seg_counts[di, ri] += 1
                mode_bandit.update(di, ri, score, cfg.sigma1)

                # Record decision history for XAI dashboard
                q_val = 0.0
                if getattr(self, "use_op_rl", True) and getattr(self.op_ctrl, "last_q", None) is not None:
                    try:
                        q_val = float(self.op_ctrl.last_q[op_action])
                    except Exception:
                        pass
                if len(self.solver_history) < 500:
                    self.solver_history.append(
                        {
                            "iteration": int(it),
                            "destroy_op": DESTROY[di].__name__,
                            "repair_op": REPAIR[ri].__name__,
                            "q_value": q_val,
                            "cost": float(cand.cost) if cand.feasible else float("inf"),
                            "best_cost": float(best.cost),
                            "accepted": bool(accepted),
                        }
                    )
                next_imp = sum(recent_improvements) / max(len(recent_improvements), 1)
                next_state = self._op_state(cur, best, action, it + 1, temp, no_imp, pool, next_imp)
                done = 1.0 if no_imp >= cfg.early_stop_patience else 0.0
                if not frozen and getattr(self, "use_op_rl", True):
                    iter_rew_raw = self._iteration_reward(
                        cur_cost_before,
                        cur_nv_before,
                        best_cost_before,
                        best_nv_before,
                        cur,
                        best,
                        accepted,
                    )
                    iter_rew_norm = norm.normalize(iter_rew_raw)
                    self.ucb_aug.update(op_action, iter_rew_raw)
                    if iter_rew_norm is not None:
                        self.op_ctrl.observe(
                            op_state,
                            op_action,
                            iter_rew_norm,
                            next_state,
                            done,
                        )
                    if (it + 1) % 4 == 0:
                        self.op_ctrl.train_step()

                # Adapt q_scale based on whether search is improving or stuck
                if no_imp == 0:
                    self.q_scale = max(0.6, self.q_scale * 0.98)
                else:
                    self.q_scale = min(1.6, self.q_scale * 1.005)

                temp *= cfg.temp_decay * mode.temp_decay_scale
                history.append(best.cost)
                if no_imp >= cfg.early_stop_patience:
                    break

            for mb in self.mode_bandits:
                mb.decay(cfg.bandit_decay)



            # Adaptive mode_bandit reset khi plateau segment
            if no_imp > 0 and no_imp % (cfg.plateau_start * 2) == 0:
                for mb in self.mode_bandits:
                    mb.reset()
            for d in range(N_D):
                for r in range(N_R):
                    if seg_counts[d, r] > 0:
                        avg = seg_scores[d, r] / seg_counts[d, r]
                        dw[d] = dw[d] * (1 - cfg.weight_decay) + avg * cfg.weight_decay
                        rw[r] = rw[r] * (1 - cfg.weight_decay) + avg * cfg.weight_decay
            all_dw[action] = np.maximum(dw, 0.1)
            all_rw[action] = np.maximum(rw, 0.1)

            state_after = self._state(
                cur,
                best,
                no_imp,
                temp,
                sum(recent_improvements) / max(len(recent_improvements), 1),
                min((seg_idx + 1) / max(n_segments, 1), 1.0),
                pool,
            )
            if ctrl_active:
                self.ctrl.observe(
                    state_before,
                    action,
                    self._segment_reward(
                        seg_best_cost_pre,
                        seg_best_nv_pre,
                        best,
                        seg_cur_cost_pre,
                        seg_cur_nv_pre,
                        cur,
                        accepted_moves,
                        action,
                    ),
                    state_after,
                    0.0,
                )
                self.ctrl.train_step()
            if no_imp >= cfg.early_stop_patience:
                break

        _bks_entry = BKS.get(self.inst.name)
        _bks_nv = int(_bks_entry["nv"]) if _bks_entry else 0

        if cfg.recombine_after_main_search and not self._out_of_time():
            recombined = recombine_with_route_pool(
                best,
                pool,
                cfg,
                nv_ceiling=best.nv,
                td_only=(best.nv <= _bks_nv),
                heatmap=self.heatmap,
                _stats=_stats,
            )
            if recombined.dominates(best):
                best = recombined
                pool.add_plan(best)
                history.append(best.cost)

        if not self._out_of_time():
            best = self._fixed_nv_polish(best, pool, inherited_bandit=self.mode_bandits[MODE_INTENSIFY])
            history.append(best.cost)

        if cfg.recombine_after_polish and not self._out_of_time():
            recombined = recombine_with_route_pool(
                best,
                pool,
                cfg,
                nv_ceiling=best.nv,
                td_only=(best.nv <= _bks_nv),
                heatmap=self.heatmap,
                _stats=_stats,
            )
            if recombined.dominates(best):
                best = self._local_search(
                    recombined, max_passes=cfg.polish_ls_passes, nv_ceiling=recombined.nv, max_ls_moves=cfg.max_ls_moves
                )
                history.append(best.cost)

        # Scale post-processing search effort based on hybrid_iterations budget
        budget_scale = max(0.005, cfg.hybrid_iterations / 1200.0)
        n_rand = max(1, int(10 * budget_scale))
        n_seeds_phc = max(1, int(20 * budget_scale))
        n_trials_1 = max(1, int(35 * budget_scale))
        n_trials_2 = max(1, int(20 * budget_scale))

        # ── Phase 0: dense column generation (generalized topology injection) ───
        if budget_scale > 0.05 and not self._out_of_time():
            self._generate_dense_pool_columns(best, pool)

        # ── Phase A: savings seeding (topology diversity ALNS can't generate) ──
        if not self._out_of_time():
            self._seed_savings_routes(pool, n_randomizations=n_rand)

        # ── Phase B: direct NV-targeted construction ──────────────────────
        _target_nv_floor = max(1, best.nv - 1)

        if best.nv > _target_nv_floor and not self._out_of_time():
            # Seed for both target and one above target (richer pool = better MILP)
            self._seed_nv_targeted_construction(pool, target_nv=_target_nv_floor, n_trials=n_trials_1)
            self._seed_nv_targeted_construction(pool, target_nv=best.nv - 1, n_trials=n_trials_2)

        # ── Phase C: large-destroy seeding (existing, unchanged) ──────────────
        if not self._out_of_time():
            self._seed_pool_large_destroy(best, pool, n_seeds=n_seeds_phc)

        # For very small iteration limits (e.g. smoke tests/quick checks), skip the heavy
        # NV-reduction heuristics and the TD polish alike.
        # The NV-reduction block is also the most expensive part of the tail, so it
        # is skipped outright once the hard deadline has passed; the cheaper TD
        # polish below still runs so the incumbent is always returned tidied up.
        # (It must therefore sit outside this guard, not inside it — each of its
        # own expensive phases carries its own _out_of_time() check, and
        # td_converge_polish degenerates to a copy once the deadline has passed.)
        run_tail = cfg.hybrid_iterations > 5
        if run_tail and not self._out_of_time():
            # ── Resolve BKS floor: never search below BKS NV ─────────────────────
            # Prevents 22-second committed search on instances already at optimal NV
            # (e.g. RC207 at NV=3 wasting time chasing NV=2 instead of fixing TD).
            _bks_entry = BKS.get(self.inst.name)
            _bks_nv = int(_bks_entry["nv"]) if _bks_entry else max(1, best.nv - 2)

            # NV-reduction loop stops at BKS floor (or at the hard deadline —
            # measured 8% budget overrun at n=1000 came from these tail phases
            # running unconditionally after a single entry check).
            for _target_nv in range(best.nv - 1, max(_bks_nv - 1, 0), -1):
                if self._out_of_time():
                    break
                _rec = recombine_with_route_pool(best, pool, cfg, nv_target=_target_nv, heatmap=self.heatmap, _stats=_stats)
                if not _rec.feasible or _rec.nv > _target_nv:
                    break
                _rec = self._local_search(
                    _rec,
                    max_passes=cfg.polish_ls_passes + 1,
                    nv_ceiling=_rec.nv,
                    max_ls_moves=cfg.max_ls_moves * 2,
                    pool=pool,
                )
                if _rec.feasible and _rec.nv <= _target_nv:
                    best = _rec
                    pool.add_plan(best)
                    history.append(best.cost)

            # Each elimination pass is skipped once the deadline has passed; a
            # skipped pass leaves ``best`` untouched, so the guards below no-op.
            chain_result = _ejection_chain_eliminate(best) if not self._out_of_time() else None
            if (
                chain_result is not None
                and chain_result.feasible
                and (chain_result.nv < best.nv or chain_result.dominates(best))
            ):
                best = chain_result
                pool.add_plan(best)
                history.append(best.cost)

            _one_over_bks = _bks_entry is not None and best.nv == _bks_nv + 1
            buffered = (
                _buffered_route_elimination(
                    best,
                    pool=pool,
                    hard_mode=_one_over_bks,
                    deadline=getattr(self, "_deadline", None),
                )
                if not self._out_of_time()
                else best
            )
            if buffered.feasible and (buffered.nv < best.nv or buffered.dominates(best)):
                best = buffered
                pool.add_plan(best)
                history.append(best.cost)

            eliminated = (
                _iterative_route_elimination(best, self.inst, pool=pool)
                if not self._out_of_time()
                else best
            )
            if eliminated.feasible and (eliminated.nv < best.nv or eliminated.dominates(best)):
                best = eliminated
                pool.add_plan(best)
                history.append(best.cost)

            # Committed NV search only if still above BKS floor
            if best.nv > _bks_nv and not self._out_of_time():
                committed = self._committed_nv_search(best, pool, target_nv=best.nv - 1)
                if (
                    committed is not None
                    and committed.feasible
                    and (committed.nv < best.nv or committed.dominates(best))
                ):
                    best = committed
                    pool.add_plan(best)
                    history.append(best.cost)
                    chain2 = _ejection_chain_eliminate(best)
                    if chain2 is not None and chain2.feasible and (chain2.nv < best.nv or chain2.dominates(best)):
                        best = chain2
                        pool.add_plan(best)
                        history.append(best.cost)

        # ── TD polish ────────────────────────────────
        if run_tail:
            # Phase A: convergent intra-route sequence optimization
            # Runs 2-opt + or-opt(1,2,3) per route to convergence with no move cap.
            # Critical for wide-TW instances (RC2, R2) where routes carry 30+ customers.
            is_wide_tw = self.inst.tw_tight_frac < 0.15
            n_intra_passes = 35 if is_wide_tw else 20
            best = td_converge_polish(
                best, max_passes=n_intra_passes, deadline=getattr(self, "_deadline", None)
            )
            if best.feasible:
                history.append(best.cost)

            # Phase B: inter-route local search + MILP recombination
            td_scale = 4 if is_wide_tw else 2
            td_polished = (
                self._local_search(
                    best,
                    max_passes=cfg.polish_ls_passes + td_scale,
                    nv_ceiling=best.nv,
                    max_ls_moves=cfg.max_ls_moves * (td_scale + 1),
                )
                if not self._out_of_time()
                else best
            )
            if td_polished.feasible and td_polished.cost + 1e-6 < best.cost:
                best = td_polished
                history.append(best.cost)

            td_rec = (
                recombine_with_route_pool(
                    best,
                    pool,
                    cfg,
                    nv_ceiling=best.nv,
                    td_only=True,
                    _stats=_stats,
                )
                if not self._out_of_time()
                else best
            )
            if td_rec.feasible and td_rec.cost + 1e-6 < best.cost:
                best = td_rec
                pool.add_plan(best)
                history.append(best.cost)

            # Phase C: Destroy-repair ALNS at fixed NV (wide-TW intensification)
            # After NV reduction, routes carry 30+ customers and local search
            # gets trapped in deep local minima. Large perturbations (40-60%
            # destroy) allow structural rearrangement while maintaining NV.
            # Runs only for wide-TW instances where the TD gap is significant.
            if is_wide_tw and not self._out_of_time():
                best = self._fixed_nv_polish(
                    best, pool, inherited_bandit=self.mode_bandits[MODE_INTENSIFY], initial_temp=1.0
                )
                history.append(best.cost)
                # Final intra-route convergence after destroy-repair
                best = td_converge_polish(
                    best, max_passes=n_intra_passes, deadline=getattr(self, "_deadline", None)
                )
                if best.feasible:
                    history.append(best.cost)

        # Final split controller attempt at the end of search
        if not frozen and self.split_ctrl is not None and not self._out_of_time():
            split_res = self.split_ctrl.try_split(best, best.nv)
            if split_res is not None:
                split_res = self._local_search(
                    split_res,
                    max_passes=cfg.polish_ls_passes,
                    nv_ceiling=split_res.nv,
                    max_ls_moves=cfg.max_ls_moves,
                )
                if split_res.dominates(best):
                    best = split_res
                    pool.add_plan(best)
                    history.append(best.cost)

        best.algo = self.algo_name
        self.archive.update(best)

        # Logging transition metrics for Adaptive Feasibility verification
        if getattr(cfg, "adaptive_feasibility", True) and not _is_sub_solve:
            import logging
            logger = logging.getLogger("vrptw.solvers")
            early_stop_iter = self.current_it if (no_imp >= cfg.early_stop_patience) else None
            log_data = {
                "phase1_best_nv": getattr(self, "phase1_best_nv", None),
                "phase1_best_td": getattr(self, "phase1_best_td", None),
                "phase1_final_lambda": getattr(self, "phase1_final_lambda", None),
                "early_stop_triggered_iter": early_stop_iter,
                "phase2_final_nv": best.nv,
                "phase2_final_td": best.cost
            }
            logger.info(log_data)
            print(f"[phase_transition_log] {log_data}")

        self.current_it = None
        return best, history

    def _split_routes_gnn_guided(self, routes: list[list[int]]) -> tuple[list[int], list[int]]:
        if len(routes) < 2:
            return [0], []

        num_r = len(routes)
        W = np.zeros((num_r, num_r), dtype=np.float32)
        for i in range(num_r):
            for j in range(num_r):
                if i != j:
                    val = 0.0
                    if self.heatmap is not None:
                        for u in routes[i]:
                            for v in routes[j]:
                                val += self.heatmap[u, v] + self.heatmap[v, u]
                    W[i, j] = val

        if self.heatmap is None:
            centroids = []
            for r in routes:
                xs = [self.inst.coords[c][0] for c in r]
                ys = [self.inst.coords[c][1] for c in r]
                centroids.append(np.array([np.mean(xs), np.mean(ys)]))
            for i in range(num_r):
                for j in range(num_r):
                    if i != j:
                        W[i, j] = -float(np.linalg.norm(centroids[i] - centroids[j]))

        best_seed = 0
        min_conn = float("inf")
        for i in range(num_r):
            conn = sum(W[i, j] for j in range(num_r))
            if conn < min_conn:
                min_conn = conn
                best_seed = i

        V1 = {best_seed}
        V2 = set(range(num_r)) - V1
        target_size = num_r // 2

        while len(V1) < target_size:
            best_candidate = -1
            max_conn = -float("inf")
            for u in V2:
                conn = sum(W[u, v] for v in V1)
                if conn > max_conn:
                    max_conn = conn
                    best_candidate = u
            V1.add(best_candidate)
            V2.remove(best_candidate)

        return list(V1), list(V2)

    def _create_sub_instance(self, cust_ids: list[int]) -> Inst:
        raw_data = np.zeros((len(cust_ids) + 1, 7), dtype=np.float32)

        # Depot (0)
        raw_data[0, 0] = 0
        raw_data[0, 1:3] = self.inst.coords[0]
        raw_data[0, 3] = self.inst.demands[0]
        raw_data[0, 4] = self.inst.ready_times[0]
        raw_data[0, 5] = self.inst.due_times[0]
        raw_data[0, 6] = self.inst.service_times[0]

        # Customers
        for i, c in enumerate(cust_ids):
            raw_data[i + 1, 0] = i + 1
            raw_data[i + 1, 1:3] = self.inst.coords[c]
            raw_data[i + 1, 3] = self.inst.demands[c]
            raw_data[i + 1, 4] = self.inst.ready_times[c]
            raw_data[i + 1, 5] = self.inst.due_times[c]
            raw_data[i + 1, 6] = self.inst.service_times[c]

        raw = {
            "name": f"{self.inst.name}_sub_{len(cust_ids)}",
            "capacity": self.inst.capacity,
            "data": raw_data
        }
        return Inst(raw)

    def solve_multi_run(
        self,
        n_runs: int = 5,
        base_seed: int = 42,
        shared_norm: WelfordRewardNormalizer | None = None,
    ) -> tuple[Plan, list[list[float]]]:
        """
        Cascade warm-start: each run after the first initialises from the
        best solution found so far.  Temperature is doubled for warm-started
        runs to prevent premature convergence from the already-good init.

        Generalisation guarantee: the cascade only changes the *starting point*,
        not the search logic — no instance-specific tuning.

        Returns: (best plan across all runs, per-run history lists)
        """
        best_overall: Plan | None = None
        all_histories: list[list[float]] = []

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            init = best_overall.copy() if best_overall is not None else None
            plan, history = self.solve(
                seed=seed,
                init=init,
                shared_norm=shared_norm,
                _warm_start=init is not None,
            )
            all_histories.append(history)
            if best_overall is None or plan.dominates(best_overall) or plan.nv < best_overall.nv:
                best_overall = plan.copy()
                self.archive.update(best_overall)

        assert best_overall is not None
        return best_overall, all_histories


# ---------------------------------------------------------------------------
# Hybrid-Fixed
# ---------------------------------------------------------------------------
class HybridFixedSolver(HybridDDQNSolver):
    algo_name = ALGO_HYBRID_FIXED
    use_op_rl = False
    use_rl = False

    def _select_action(self, state_before, cur, best, no_imp, progress, pool, frozen):
        del state_before, best, progress, pool, frozen
        if self._route_reduce_trigger(cur, no_imp):
            return MODE_ROUTE_REDUCE, False
        return MODE_DEFAULT, False

    def solve(
        self,
        seed: int | None = None,
        frozen: bool = True,
        init: Plan | None = None,
        shared_norm: WelfordRewardNormalizer | None = None,
        _is_sub_solve: bool = False,
        _warm_start: bool = False,
        _deadline: float | None = None,
    ) -> tuple[Plan, list[float]]:
        plan, history = super().solve(
            seed=seed,
            frozen=frozen,
            init=init,
            shared_norm=shared_norm,
            _is_sub_solve=_is_sub_solve,
            _warm_start=_warm_start,
            _deadline=_deadline,
        )
        plan.algo = self.algo_name
        return plan, history


# ---------------------------------------------------------------------------
# Hybrid-Rule
# ---------------------------------------------------------------------------
class HybridRuleSolver(HybridDDQNSolver):
    algo_name = ALGO_HYBRID_RULE
    use_op_rl = False
    use_rl = False

    def _select_action(self, state_before, cur, best, no_imp, progress, pool, frozen) -> tuple[int, bool]:
        del state_before, best, frozen
        if self._route_reduce_trigger(cur, no_imp):
            return MODE_ROUTE_REDUCE, False
        pool_ready = len(pool._routes) >= max(self.cfg.rl_recombine_min_routes, max(12, cur.nv * 2))
        fleet_fill = _fleet_fill(cur)
        slack = _avg_slack(cur)
        if pool_ready and no_imp >= max(10, self.cfg.ctrl_start // 2) and fleet_fill >= 0.66 and progress < 0.92:
            return MODE_POOL_RECOMBINE, False
        if self.inst.tw_tight_frac >= 0.18 and slack < 0.16 and no_imp >= max(8, self.cfg.ctrl_start // 2):
            return MODE_TW_RESCUE, False
        if no_imp >= max(self.cfg.ctrl_start_floor, self.cfg.ctrl_start // 2):
            return (MODE_DIVERSIFY if progress < 0.45 else MODE_INTENSIFY), False
        return MODE_DEFAULT, False

    def solve(
        self,
        seed: int | None = None,
        frozen: bool = True,
        init: Plan | None = None,
        shared_norm: WelfordRewardNormalizer | None = None,
        _is_sub_solve: bool = False,
        _warm_start: bool = False,
        _deadline: float | None = None,
    ) -> tuple[Plan, list[float]]:
        plan, history = super().solve(
            seed=seed,
            frozen=frozen,
            init=init,
            shared_norm=shared_norm,
            _is_sub_solve=_is_sub_solve,
            _warm_start=_warm_start,
            _deadline=_deadline,
        )
        plan.algo = self.algo_name
        return plan, history


PlateauHybridSolver = HybridDDQNSolver
ScheduledHybridSolver = HybridRuleSolver
RLALNSSolver = HybridDDQNSolver


def run_ortools(inst: Inst, cfg: Config) -> tuple[Plan | None, float]:
    if not ORTOOLS_OK:
        print("  [OR-Tools] not installed — skipping")
        return None, 0.0
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    scale = 100000  # Increased scale factor from 100 to 100000 for precision
    n_nodes = inst.n + 1
    n_vehicles = inst.n
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Use np.round to avoid truncation/underflow errors
    dist_mat = np.round(inst.dist * scale).astype(np.int64)
    serv_int = np.round(inst.service_times * scale).astype(np.int64)

    # 1. Cost Evaluator: pure travel distance (avoids GLS search bias from service times)
    cost_matrix = dist_mat.tolist()
    cost_idx = routing.RegisterTransitMatrix(cost_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

    # Set a large fixed cost per vehicle to ensure OR-Tools prioritizes vehicle count minimization (fleet size)
    routing.SetFixedCostOfAllVehicles(int(100000 * scale))
    demands_int = inst.demands.astype(int)

    def demand_cb(fi):
        return int(demands_int[manager.IndexToNode(fi)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(inst.capacity)] * n_vehicles, True, "Capacity")

    # 2. Time Dimension: travel distance + service time (required for time windows)
    transit_matrix = (dist_mat + serv_int[:, None]).tolist()
    transit_idx = routing.RegisterTransitMatrix(transit_matrix)
    routing.AddDimension(
        transit_idx, int(np.round(inst.horizon * scale)), int(np.round(inst.horizon * scale)), False, "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, inst.n + 1):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(
            int(np.round(inst.ready_times[node] * scale)), int(np.round(inst.due_times[node] * scale))
        )
    for v in range(n_vehicles):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = int(cfg.ortools_time_limit)
    params.log_search = False
    t0 = time.time()
    solution = routing.SolveWithParameters(params)
    elapsed = time.time() - t0
    if not solution:
        print(f"  [OR-Tools] no solution ({elapsed:.1f}s)")
        return None, elapsed
    routes: list[list[int]] = []
    for v in range(n_vehicles):
        route: list[int] = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        if route:
            routes.append(route)
    plan = Plan(routes, inst, ALGO_ORTOOLS)
    if not plan.feasible:
        print(f"  [OR-Tools] infeasible ({elapsed:.1f}s)")
        return None, elapsed
    return plan, elapsed


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int):
        s, a, r, ns, d = zip(*random.sample(self.buf, batch_size))
        return (
            np.array(s, np.float32),
            np.array(a, np.int64),
            np.array(r, np.float32),
            np.array(ns, np.float32),
            np.array(d, np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class DQNNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_adaptive_reheat_mult(inst) -> float:
    """Computes an instance-adaptive restart reheat temperature multiplier.
    Tight TW instances (tw_tight_frac ~ 1.0, e.g. R1, RC1, C1) -> 1.5x (protects TW feasibility)
    Wide TW instances (tw_tight_frac ~ 0.0, e.g. R2, RC2, C2) -> 4.0x (aggressive exploration)
    """
    tightness = max(0.0, min(1.0, getattr(inst, "tw_tight_frac", 0.5)))
    return 1.5 + 2.5 * (1.0 - tightness)


class DQNSolver:
    """Pure constructive RL — ablation study only. Not competitive."""

    def __init__(self, inst: Inst, cfg: Config):
        self.inst = inst
        self.cfg = cfg
        self.q = DQNNet(cfg.dqn_state_dim, inst.n + 1, cfg.dqn_hidden).to(DEVICE)
        self.q_t = DQNNet(cfg.dqn_state_dim, inst.n + 1, cfg.dqn_hidden).to(DEVICE)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.dqn_lr)
        self.buf = ReplayBuffer(cfg.dqn_buffer)
        self.eps = cfg.dqn_eps_start

    def _state(self, node: int, visited: set[int], load: float, t: float) -> np.ndarray:
        inst = self.inst
        uv = inst.n - len(visited)
        feas = [
            n
            for n in range(1, inst.n + 1)
            if n not in visited
            and load + inst.demands[n] <= inst.capacity
            and t + inst.dist[node, n] <= inst.due_times[n]
        ]
        nf = len(feas)
        if feas:
            slacks = [inst.due_times[n] - (t + inst.dist[node, n]) for n in feas]
            ms = min(slacks) / max(inst.horizon, 1)
            av = (sum(slacks) / nf) / max(inst.horizon, 1)
            uf = sum(1 for s in slacks if s < 0.1 * inst.horizon) / max(nf, 1)
            aw = (sum(inst.tw_width[n] for n in feas) / nf) / max(inst.max_tw_width, 1)
        else:
            ms = av = uf = aw = 0.0
        return np.array(
            [
                load / inst.capacity,
                t / max(inst.horizon, 1),
                len(visited) / inst.n,
                (inst.capacity - load) / inst.capacity,
                uv / inst.n,
                nf / max(uv, 1),
                inst.coords[node, 0] / 100,
                inst.coords[node, 1] / 100,
                inst.demands[node] / inst.capacity,
                ms,
                av,
                uf,
                aw,
            ],
            dtype=np.float32,
        )

    def _acts(self, node: int, visited: set[int], load: float, t: float) -> list[int]:
        inst = self.inst
        acts = [0]
        for n in range(1, inst.n + 1):
            if (
                n not in visited
                and load + inst.demands[n] <= inst.capacity
                and t + inst.dist[node, n] <= inst.due_times[n]
            ):
                acts.append(n)
        return acts

    def _sel(self, state: np.ndarray, feasible: list[int]) -> int:
        if random.random() < self.eps:
            return random.choice(feasible)
        with torch.no_grad():
            q = self.q(torch.tensor(state, device=DEVICE).unsqueeze(0)).cpu().numpy()[0]
        return max(feasible, key=lambda a: q[a])

    def _train(self) -> None:
        if len(self.buf) < self.cfg.dqn_batch:
            return
        s, a, r, ns, d = self.buf.sample(self.cfg.dqn_batch)
        s_t = torch.tensor(s, device=DEVICE)
        a_t = torch.tensor(a, dtype=torch.long, device=DEVICE)
        r_t = torch.tensor(r, device=DEVICE)
        ns_t = torch.tensor(ns, device=DEVICE)
        d_t = torch.tensor(d, device=DEVICE)
        qp = self.q(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            tgt = r_t + self.cfg.dqn_gamma * self.q_t(ns_t).max(1)[0] * (1 - d_t)
        loss = F.mse_loss(qp, tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

    def _episode(self) -> tuple[Plan, list]:
        inst = self.inst
        visited: set[int] = set()
        routes: list[list[int]] = []
        trans: list = []
        while len(visited) < inst.n:
            route: list[int] = []
            node = 0
            load = 0.0
            t = 0.0
            is_new = True
            while True:
                state = self._state(node, visited, load, t)
                feas = self._acts(node, visited, load, t)
                if len(feas) == 1:
                    break
                action = self._sel(state, feas)
                if action == 0:
                    break
                dv = inst.dist[node, action]
                rew = -dv / max(inst.max_dist, 1)
                if is_new and routes:
                    rew -= self.cfg.dqn_vehicle_penalty / inst.n
                is_new = False
                load += inst.demands[action]
                t = max(t + dv, inst.ready_times[action]) + inst.service_times[action]
                visited.add(action)
                route.append(action)
                ns = self._state(action, visited, load, t)
                done = float(len(visited) == inst.n)
                trans.append((state, action, rew, ns, done))
                node = action
            if route:
                routes.append(route)
        return Plan(routes, inst, ALGO_DQN), trans

    def solve(self, seed: int | None = None) -> tuple[Plan, list[float]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        cfg = self.cfg
        best = None
        bc = float("inf")
        hist: list[float] = []
        self.eps = cfg.dqn_eps_start
        n_eps = max(50, cfg.alns_iterations // self.inst.n)
        for ep in range(n_eps):
            plan, trans = self._episode()
            if plan.feasible and trans:
                bonus = max(0.0, (bc - plan.cost) / bc * 10) if bc < float("inf") else 1.0
                s, a, r, ns, d = trans[-1]
                trans[-1] = (s, a, r + bonus, ns, d)
                if plan.cost < bc:
                    bc = plan.cost
                    best = plan.copy()
            for tr in trans:
                self.buf.push(*tr)
            if ep % cfg.dqn_train_freq == 0:
                for _ in range(min(5, len(self.buf) // max(cfg.dqn_batch, 1))):
                    self._train()
            if ep % cfg.dqn_target_freq == 0:
                self.q_t.load_state_dict(self.q.state_dict())
            self.eps = max(cfg.dqn_eps_end, self.eps * cfg.dqn_eps_decay)
            hist.append(bc if bc < float("inf") else float("nan"))
        if best is None:
            best = build_greedy(self.inst, ALGO_DQN)
        best.algo = ALGO_DQN
        return best, hist

