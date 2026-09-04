import numpy as np

from vrptw import Config, FlatDDQNSolver, Inst


def test_flat_ddqn_execution():
    """Verify FlatDDQNSolver runs with flat macro regime (MODE_DEFAULT) and active micro-DDQN."""
    np.random.seed(42)
    n = 15
    # Build data matrix: [id, x, y, demand, ready, due, service]
    data = np.zeros((n + 1, 7), dtype=np.float64)
    data[:, 0] = np.arange(n + 1)
    data[:, 1:3] = np.random.uniform(0, 50, size=(n + 1, 2))
    data[1:, 3] = np.random.randint(1, 5, size=n)
    data[0, 3] = 0
    data[:, 4] = 0.0
    data[:, 5] = 500.0  # horizon
    data[1:, 6] = 5.0
    data[0, 6] = 0.0

    inst = Inst(
        {
            "name": "test_flat_15",
            "capacity": 30.0,
            "data": data,
        }
    )

    cfg = Config(
        alns_iterations=30,
        hybrid_iterations=30,
    )

    solver = FlatDDQNSolver(inst, cfg)
    plan, history = solver.solve(seed=42)

    assert plan is not None
    assert plan.feasible
    assert plan.algo == "Flat-DDQN"
    assert len(history) > 0
    assert solver.mode_trace[0] >= 0
