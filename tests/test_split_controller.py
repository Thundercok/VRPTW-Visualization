import numpy as np
import pytest

from vrptw.config import Config
from vrptw.core import Inst, Plan
from vrptw.split_controller import SplitController, extract_giant_tour


@pytest.fixture
def dummy_instance():
    # Coords of depot (0) and 3 customers (1, 2, 3)
    coords = np.array(
        [
            [50.0, 50.0],  # depot
            [50.0, 60.0],  # north
            [60.0, 50.0],  # east
            [40.0, 50.0],  # west
        ],
        dtype=float,
    )
    demands = np.array([0.0, 5.0, 10.0, 5.0])
    ready = np.array([0.0, 0.0, 10.0, 5.0])
    due = np.array([100.0, 80.0, 80.0, 80.0])
    service = np.array([0.0, 5.0, 5.0, 5.0])

    ids = np.array([0.0, 1.0, 2.0, 3.0])
    data = np.column_stack([ids, coords[:, 0], coords[:, 1], demands, ready, due, service])

    return Inst({"name": "DUMMY", "capacity": 20.0, "data": data})


def test_extract_giant_tour(dummy_instance):
    # Construct a plan
    routes = [[1, 2], [3]]
    plan = Plan(routes, dummy_instance)

    tour = extract_giant_tour(plan)

    assert len(tour) == 3
    assert set(tour) == {1, 2, 3}

    # Verify ordering by polar angle:
    # North (1): dx=0, dy=10 -> angle = pi/2 (1.57)
    # East (2): dx=10, dy=0 -> angle = 0 (0.0)
    # West (3): dx=-10, dy=0 -> angle = pi (3.14) or -pi
    # Sorted order of angles:
    # West (-pi) or East (0) -> North (pi/2) -> West (pi)
    # Let's check math.atan2(dy, dx):
    # dx=10, dy=0 -> angle = 0.0
    # dx=0, dy=10 -> angle = pi/2
    # dx=-10, dy=0 -> angle = pi
    # So the sorted order should be: East (2), North (1), West (3).
    assert tour == [2, 1, 3]


def test_split_controller_state_and_action(dummy_instance):
    cfg = Config()
    controller = SplitController(cfg, dummy_instance)

    # 1. State size check
    tour = [2, 1, 3]
    state, can_continue = controller._build_state(
        current_route=[2], next_cust=1, tour=tour, tour_idx=1, routes_so_far=0
    )

    assert state.shape == (18,)
    # Load condition: 10 demand of 2 + 5 demand of 1 = 15 <= capacity (20) -> should be feasible
    assert can_continue

    # If we add customer 3 (demand 5), total load = 15 + 5 = 20 <= 20
    # But let's check with exceeding capacity:
    state_exceed, can_continue_exceed = controller._build_state(
        current_route=[2, 1], next_cust=3, tour=tour, tour_idx=2, routes_so_far=0
    )
    # Customer 2 (10) + Customer 1 (5) + Customer 3 (5) = 20 (exactly capacity).
    # Since load is exactly capacity, it's still feasible.
    assert can_continue_exceed

    # If capacity is exceeded (e.g. if we artificially double demand of customer 3)
    dummy_instance.demands[3] = 15.0
    state_exceed2, can_continue_exceed2 = controller._build_state(
        current_route=[2, 1], next_cust=3, tour=tour, tour_idx=2, routes_so_far=0
    )
    assert not can_continue_exceed2

    # Verify action mask choice: when can_continue is False, action is forced to 1 (SPLIT)
    action = controller.act(state_exceed2, can_continue=can_continue_exceed2)
    assert action == 1


def test_try_split(dummy_instance):
    cfg = Config(split_eps_start=0.0)  # no exploration for deterministic test
    controller = SplitController(cfg, dummy_instance)

    # Set weights to make it predict action 0 (continue) or 1 (split)
    # We just run a simple forward and backward training step
    plan = Plan([[2], [1, 3]], dummy_instance)

    # Push dummy transitions to buffer to allow training step
    for _ in range(50):
        s = np.random.randn(18).astype(np.float32)
        ns = np.random.randn(18).astype(np.float32)
        controller.buf.push(s, 0, 1.0, ns, 0.0)

    assert len(controller.buf) == 50
    controller.train_step()

    # Try split execution
    result = controller.try_split(plan, incumbent_nv=2)
    # Since it is randomized/initialized randomly, it might return None or a plan.
    # We just check that it runs and returns either None or a Plan.
    if result is not None:
        assert isinstance(result, Plan)
        assert result.feasible
