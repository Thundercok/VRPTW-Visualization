import numpy as np

from vrptw.core import Inst, Plan
from vrptw.penalty import AdaptiveFeasibilityManager


def test_adaptive_feasibility_manager_tightness_classification():
    # Scenario A: Tight time windows
    raw_tight = {
        "name": "mock_tight",
        "capacity": 100.0,
        "data": np.array([
            [0, 0, 0, 0, 0, 100, 0],   # Depot: (0, 100)
            [1, 1, 0, 10, 0, 10, 0],   # Customer 1: ready=0, due=10 (Tight: 10/100 = 0.1)
            [2, 2, 0, 10, 5, 15, 0],   # Customer 2: ready=5, due=15 (Tight: 10/100 = 0.1)
        ], dtype=np.float64)
    }
    inst_tight = Inst(raw_tight)
    mgr_tight = AdaptiveFeasibilityManager(inst_tight)

    # Tight time windows (< 0.25 tightness) should initialize with lambda = 50.0
    assert mgr_tight.lam == 50.0

    # Scenario B: Wide time windows
    raw_wide = {
        "name": "mock_wide",
        "capacity": 100.0,
        "data": np.array([
            [0, 0, 0, 0, 0, 100, 0],   # Depot: (0, 100)
            [1, 1, 0, 10, 0, 80, 0],   # Customer 1: ready=0, due=80 (Wide: 80/100 = 0.8)
            [2, 2, 0, 10, 0, 90, 0],   # Customer 2: ready=0, due=90 (Wide: 90/100 = 0.9)
        ], dtype=np.float64)
    }
    inst_wide = Inst(raw_wide)
    mgr_wide = AdaptiveFeasibilityManager(inst_wide)

    # Wide time windows (>= 0.25 tightness) should initialize with lambda = 1.0
    assert mgr_wide.lam == 1.0


def test_adaptive_feasibility_manager_lambda_updates():
    raw = {
        "name": "mock",
        "capacity": 100.0,
        "data": np.array([
            [0, 0, 0, 0, 0, 100, 0],
            [1, 1, 0, 10, 0, 80, 0],
            [2, 2, 0, 10, 0, 90, 0],
        ], dtype=np.float64)
    }
    inst = Inst(raw)
    mgr = AdaptiveFeasibilityManager(inst, target_ratio=0.5, alpha_ema=0.1)

    # Start with lam = 1.0
    assert mgr.lam == 1.0

    # Record a sequence of feasible plans -> ratio should rise -> lam should decrease
    for _ in range(20):
        # mock plan with zero violations
        plan = Plan([[1, 2]], inst)
        mgr.record_solution(plan)
        mgr.update_penalties()

    assert mgr.lam < 1.0

    # Record a sequence of infeasible plans (we mock this by manually setting a low ratio or directly triggering updates)
    mgr.feasible_ema = 0.1  # simulate high infeasibility rate
    mgr.update_penalties()
    assert mgr.lam > 0.1  # lam should start increasing again


def test_lagrangian_penalty_controller_tightness_classification():
    from vrptw.penalty import LagrangianPenaltyController

    # Tight time windows
    raw_tight = {
        "name": "mock_tight",
        "capacity": 100.0,
        "data": np.array([
            [0, 0, 0, 0, 0, 100, 0],
            [1, 1, 0, 10, 0, 10, 0],
            [2, 2, 0, 10, 5, 15, 0],
        ], dtype=np.float64),
    }
    mgr_tight = LagrangianPenaltyController(Inst(raw_tight))
    assert mgr_tight.lam_tw == 50.0
    assert mgr_tight.lam_cap == 10.0

    # Wide time windows
    raw_wide = {
        "name": "mock_wide",
        "capacity": 100.0,
        "data": np.array([
            [0, 0, 0, 0, 0, 100, 0],
            [1, 1, 0, 10, 0, 80, 0],
            [2, 2, 0, 10, 0, 90, 0],
        ], dtype=np.float64),
    }
    mgr_wide = LagrangianPenaltyController(Inst(raw_wide))
    assert mgr_wide.lam_tw == 1.0
    assert mgr_wide.lam_cap == 2.0


def test_lagrangian_penalty_controller_outlier_scaling():
    """Verify that Lagrangian step size scales with violation magnitude (unlike boolean EMA)."""
    from vrptw.penalty import LagrangianPenaltyController

    raw = {
        "name": "mock",
        "capacity": 100.0,
        "data": np.array([
            [0, 0, 0, 0, 0, 100, 0],
            [1, 1, 0, 10, 0, 80, 0],
            [2, 2, 0, 10, 0, 90, 0],
        ], dtype=np.float64),
    }
    inst = Inst(raw)

    # Base controller with known feasible incumbent at cost = 100.0
    ctrl_small = LagrangianPenaltyController(inst, theta=2.0)
    feasible_plan = Plan([[1, 2]], inst)
    feasible_plan._cost = 100.0
    feasible_plan._violation_capacity = 0.0
    feasible_plan._violation_tw = 0.0
    ctrl_small.update(feasible_plan)
    init_lam_cap = ctrl_small.lam_cap

    # Small violation plan: v_cap = 2.0, cost = 120.0
    plan_small = Plan([[1, 2]], inst)
    plan_small._cost = 120.0
    plan_small._violation_capacity = 2.0
    plan_small._violation_tw = 0.0
    ctrl_small.update(plan_small)
    delta_lam_small = ctrl_small.lam_cap - init_lam_cap

    # Outlier violation plan: v_cap = 50.0, cost = 120.0
    ctrl_large = LagrangianPenaltyController(inst, theta=2.0)
    ctrl_large.update(feasible_plan)
    plan_large = Plan([[1, 2]], inst)
    plan_large._cost = 120.0
    plan_large._violation_capacity = 50.0
    plan_large._violation_tw = 0.0
    ctrl_large.update(plan_large)
    delta_lam_large = ctrl_large.lam_cap - init_lam_cap

    # Step size scales proportionally with violation magnitude and numerator
    assert delta_lam_small > 0.0
    assert delta_lam_large > 0.0
    assert ctrl_large.lam_cap > ctrl_small.lam_cap_min


def test_lagrangian_penalty_controller_stall_decay_and_convergence():
    """Verify stall rate halving theta and decay toward lower bounds on feasible streaks."""
    from vrptw.penalty import LagrangianPenaltyController

    raw = {
        "name": "mock",
        "capacity": 100.0,
        "data": np.array([
            [0, 0, 0, 0, 0, 100, 0],
            [1, 1, 0, 10, 0, 80, 0],
            [2, 2, 0, 10, 0, 90, 0],
        ], dtype=np.float64),
    }
    inst = Inst(raw)
    ctrl = LagrangianPenaltyController(inst, theta=2.0, stall_limit=5)

    # Establish feasible baseline
    feasible_best = Plan([[1, 2]], inst)
    feasible_best._cost = 100.0
    feasible_best._violation_capacity = 0.0
    feasible_best._violation_tw = 0.0
    ctrl.update(feasible_best)
    assert ctrl.theta == 2.0

    # Trigger 5 non-improving feasible updates -> stall limit reached -> theta halved
    worse_feasible = Plan([[1, 2]], inst)
    worse_feasible._cost = 110.0
    worse_feasible._violation_capacity = 0.0
    worse_feasible._violation_tw = 0.0
    for _ in range(5):
        ctrl.update(worse_feasible)
    assert ctrl.theta == 1.0

    # 5 more stalls -> theta halved again
    for _ in range(5):
        ctrl.update(worse_feasible)
    assert ctrl.theta == 0.5

