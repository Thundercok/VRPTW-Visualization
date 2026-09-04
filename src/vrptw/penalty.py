from __future__ import annotations


class PenaltyManager:
    def __init__(self, inst):
        self.inst = inst
        # Initial values: alpha_q (capacity) and alpha_tw (time window)
        # Adjusted by instance characteristics
        self.alpha_q = 10.0
        self.alpha_tw = 1.0

        self.cap_history = []
        self.tw_history = []

    def penalized_cost(self, plan) -> float:
        cap_viol = plan.violation_capacity
        tw_viol = plan.violation_tw
        return plan.cost + self.alpha_q * cap_viol + self.alpha_tw * tw_viol

    def record_solution(self, plan) -> None:
        cap_ok = plan.violation_capacity < 1e-6
        tw_ok = plan.violation_tw < 1e-6
        self.cap_history.append(cap_ok)
        self.tw_history.append(tw_ok)
        if len(self.cap_history) > 100:
            self.cap_history.pop(0)
        if len(self.tw_history) > 100:
            self.tw_history.pop(0)

    def update_penalties(self) -> None:
        if len(self.cap_history) < 10:
            return

        cap_infeasible_rate = sum(1 for ok in self.cap_history if not ok) / len(self.cap_history)
        tw_infeasible_rate = sum(1 for ok in self.tw_history if not ok) / len(self.tw_history)

        # Self-adjusting capacity penalty
        if cap_infeasible_rate > 0.50:
            self.alpha_q = min(10000.0, self.alpha_q * 2.0)
        elif cap_infeasible_rate < 0.10:
            self.alpha_q = max(0.01, self.alpha_q / 2.0)

        # Self-adjusting time window penalty
        if tw_infeasible_rate > 0.50:
            self.alpha_tw = min(10000.0, self.alpha_tw * 2.0)
        elif tw_infeasible_rate < 0.10:
            self.alpha_tw = max(0.01, self.alpha_tw / 2.0)


def eliminate_route_infeasible(plan, penalty_manager: PenaltyManager):
    """
    Greedily eliminates the smallest route in the plan (the one with the fewest customers)
    and distributes its customers into the remaining routes at the positions that minimize
    the increase in penalized cost.
    """
    from .core import Plan

    inst = plan.inst
    routes = [r[:] for r in plan.routes]
    if len(routes) <= 1:
        return plan.copy()

    # Find the index of the route with the fewest customers
    route_idx = min(range(len(routes)), key=lambda i: len(routes[i]))
    removed_customers = routes.pop(route_idx)

    # Create a plan with the remaining routes
    new_plan = Plan(routes, inst, plan.algo)

    # Insert each removed customer into the best position
    for cust in removed_customers:
        best_plan = None
        best_cost = float("inf")

        # Try inserting cust at every position of every route
        for r_idx in range(len(new_plan.routes)):
            for pos in range(len(new_plan.routes[r_idx]) + 1):
                # Create a copy of the routes and insert the customer
                temp_routes = [r[:] for r in new_plan.routes]
                temp_routes[r_idx].insert(pos, cust)
                temp_plan = Plan(temp_routes, inst, plan.algo)

                cost = penalty_manager.penalized_cost(temp_plan)
                if cost < best_cost:
                    best_cost = cost
                    best_plan = temp_plan

        if best_plan is not None:
            new_plan = best_plan

    return new_plan


def eliminate_two_routes_infeasible(plan, penalty_manager: PenaltyManager):
    """
    Eliminates the two smallest routes simultaneously and redistributes
    all their customers into remaining routes via penalized insertion.
    More powerful than single-route eliminate when NV is near BKS.
    """
    from .core import Plan

    inst = plan.inst
    routes = [r[:] for r in plan.routes]
    if len(routes) <= 2:
        return plan.copy()

    # Find two smallest routes by customer count
    sorted_idx = sorted(range(len(routes)), key=lambda i: len(routes[i]))
    idx1, idx2 = sorted_idx[0], sorted_idx[1]

    # Remove both (higher index first to avoid index shift)
    removed_customers = routes[max(idx1, idx2)] + routes[min(idx1, idx2)]
    for i in sorted([idx1, idx2], reverse=True):
        routes.pop(i)

    if not routes:
        return plan.copy()

    # Create a plan with the remaining routes
    new_plan = Plan(routes, inst, plan.algo)

    # Redistribute all removed customers via penalized insertion
    for cust in removed_customers:
        best_plan = None
        best_cost = float("inf")

        # Try inserting cust at every position of every route
        for r_idx in range(len(new_plan.routes)):
            for pos in range(len(new_plan.routes[r_idx]) + 1):
                temp_routes = [r[:] for r in new_plan.routes]
                temp_routes[r_idx].insert(pos, cust)
                temp_plan = Plan(temp_routes, inst, plan.algo)

                cost = penalty_manager.penalized_cost(temp_plan)
                if cost < best_cost:
                    best_cost = cost
                    best_plan = temp_plan

        if best_plan is not None:
            new_plan = best_plan

    return new_plan


class AdaptiveFeasibilityManager:
    def __init__(self, inst, target_ratio: float = 0.5, alpha_ema: float = 0.1):
        self.inst = inst
        self.target = target_ratio
        self.alpha = alpha_ema
        self.feasible_ema = target_ratio  # Initialize at target to avoid cold-start bias

        # Strategy 3: Time-window tightness calculation
        # Width of time-windows for customers (excluding depot at index 0)
        widths = [inst.due_times[i] - inst.ready_times[i] for i in range(1, inst.n + 1)]
        # Total horizon of the depot
        horizon = inst.due_times[0] - inst.ready_times[0]
        tightness = sum(widths) / (len(widths) * horizon) if horizon > 0 else 0.0

        # Seed initial penalty weight lambda and boundaries based on tightness:
        # Tight TW (R1/RC1) -> start high lambda, floor high -> acts like Baseline
        # Wide TW  (R2/RC2) -> start low lambda, floor 0.5 -> acts like Penalty but preserves NV pressure
        if tightness < 0.25:
            self.lam = 50.0
            self.lam_min = 5.0
            self.lam_max = 200.0
        else:
            self.lam = 1.0
            self.lam_min = 0.5
            self.lam_max = 50.0

    def penalized_cost(self, plan) -> float:
        viol = plan.violation_capacity + plan.violation_tw
        return plan.cost + self.lam * viol

    def record_solution(self, plan) -> None:
        # Check if plan is feasible
        is_feasible = (plan.violation_capacity < 1e-6) and (plan.violation_tw < 1e-6)
        # Update EMA of feasibility ratio
        self.feasible_ema = (1.0 - self.alpha) * self.feasible_ema + self.alpha * float(is_feasible)

    def update_penalties(self) -> None:
        # Strategy 1: Adjust self.lam dynamically
        if self.feasible_ema > self.target:
            # Too many feasible plans -> reduce penalty weight to encourage exploration
            self.lam *= 0.90
        else:
            # Too many infeasible plans -> increase penalty weight to push back to feasibility
            self.lam *= 1.15

        # Keep lam in a stable numeric range based on instance type
        self.lam = min(max(self.lam, self.lam_min), self.lam_max)


class LagrangianPenaltyController:
    """Held-Karp / Polyak Subgradient Dual Ascent Controller for VRPTW.

    Maintains separate dual multipliers lambda_cap and lambda_tw, updating them
    proportionally to constraint violation magnitudes scaled by step size t_k:
        lambda_cap <- max(0, lambda_cap + t_k * v_cap)
        lambda_tw  <- max(lambda_tw_min, min(lambda_tw_max, lambda_tw + t_k * v_tw))
    where t_k = theta * (Z(plan) - Z_LB) / (v_cap^2 + v_tw^2).
    """

    def __init__(self, inst, theta: float = 2.0, stall_limit: int = 20):
        self.inst = inst
        self.theta = theta
        self.stall_limit = stall_limit
        self._stall_count = 0
        self._best_feasible_cost = float("inf")

        # Instance Tightness Prior: derive initial multipliers and bounds from instance horizon
        widths = [inst.due_times[i] - inst.ready_times[i] for i in range(1, inst.n + 1)]
        horizon = inst.due_times[0] - inst.ready_times[0]
        tightness = sum(widths) / (len(widths) * horizon) if horizon > 0 else 0.0

        if tightness < 0.25:
            # Tight TW (e.g. R1, RC1): High time-window penalty prior
            self.lam_tw = 50.0
            self.lam_tw_min = 5.0
            self.lam_tw_max = 500.0
            self.lam_cap = 10.0
        else:
            # Wide TW (e.g. R2, RC2): Low time-window penalty prior
            self.lam_tw = 1.0
            self.lam_tw_min = 0.5
            self.lam_tw_max = 100.0
            self.lam_cap = 2.0

        self.lam_cap_min = 0.1
        self.lam_cap_max = 1000.0
        self.update_interval = 1
        self._total_calls = 0
        self._infeasible_count = 0
        self._max_v_cap = 0.0
        self._max_v_tw = 0.0

    def penalized_cost(self, plan) -> float:
        v_cap = getattr(plan, "violation_capacity", 0.0)
        v_tw = getattr(plan, "violation_tw", 0.0)
        return plan.cost + self.lam_cap * v_cap + self.lam_tw * v_tw

    def update(self, plan) -> None:
        self._total_calls += 1
        v_cap = getattr(plan, "violation_capacity", 0.0)
        v_tw = getattr(plan, "violation_tw", 0.0)
        is_feasible = (v_cap < 1e-6) and (v_tw < 1e-6)

        if not is_feasible:
            self._infeasible_count += 1
            self._max_v_cap = max(self._max_v_cap, float(v_cap))
            self._max_v_tw = max(self._max_v_tw, float(v_tw))

        if is_feasible:
            improved = plan.cost + 1e-6 < self._best_feasible_cost
            self._best_feasible_cost = min(self._best_feasible_cost, plan.cost)
            self._stall_count = 0 if improved else self._stall_count + 1
        else:
            self._stall_count += 1

        if self._stall_count >= self.stall_limit:
            self.theta = max(self.theta / 2.0, 1e-3)
            self._stall_count = 0

        # Subgradient dual step calculation
        if self._best_feasible_cost == float("inf"):
            # No feasible incumbent yet: use small default positive step
            if v_cap > 1e-6:
                self.lam_cap = min(self.lam_cap_max, self.lam_cap * 1.10)
            if v_tw > 1e-6:
                self.lam_tw = min(self.lam_tw_max, self.lam_tw * 1.10)
            return

        denom = (v_cap**2) + (v_tw**2)
        if denom < 1e-9:
            # Plan is feasible: gradually decay multipliers toward lower bounds
            self.lam_cap = max(self.lam_cap_min, self.lam_cap * 0.98)
            self.lam_tw = max(self.lam_tw_min, self.lam_tw * 0.98)
            return

        # Polyak step size
        cost_diff = max(0.0, plan.cost - self._best_feasible_cost)
        if cost_diff < 1e-6:
            cost_diff = 1.0  # nominal gradient scale when plan cost equals lower bound

        step = self.theta * cost_diff / denom
        self.lam_cap = min(self.lam_cap_max, max(self.lam_cap_min, self.lam_cap + step * v_cap))
        self.lam_tw = min(self.lam_tw_max, max(self.lam_tw_min, self.lam_tw + step * v_tw))

    def record_solution(self, plan) -> None:
        """Alias for compatibility with the search loop."""
        self.update(plan)

    def update_penalties(self) -> None:
        """Periodic hook for compatibility with the search loop cadence."""
        pass
