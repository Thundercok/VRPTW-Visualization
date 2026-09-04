import numpy as np
import torch

from vrptw.config import Config
from vrptw.rl import MILPColumnController


def test_milp_column_controller_initialization():
    cfg = Config()
    controller = MILPColumnController(cfg, use_rl=True)

    assert controller.N_ACTIONS == 8
    assert controller.state_dim == 14
    assert controller.eps == 0.30

    # Verify action specs
    for action_id in range(8):
        spec = controller.get_action_spec(action_id)
        assert "p_vehicle" in spec
        assert "max_time" in spec
        assert "route_filter" in spec


def test_milp_column_controller_forward_and_act():
    cfg = Config()
    controller = MILPColumnController(cfg, use_rl=True)
    controller.eps = 0.0  # Force greedy exploitation to test Q-net output determinism

    state = np.random.randn(14).astype(np.float32)
    action = controller.act(state, frozen=False)

    assert isinstance(action, int)
    assert 0 <= action < 8


def test_milp_column_controller_replay_and_train():
    cfg = Config()
    controller = MILPColumnController(cfg, use_rl=True)

    # Push 40 synthetic transitions to cross the batch threshold (32)
    for _ in range(40):
        s = np.random.randn(14).astype(np.float32)
        a = np.random.randint(0, 8)
        r = float(np.random.randn())
        ns = np.random.randn(14).astype(np.float32)
        done = 0.0
        controller.observe(s, a, r, ns, done)

    assert len(controller.buf) == 40

    # Store initial Q-network weights
    w_before = controller.q.trunk[0].weight.clone()

    # Run training step
    controller.train_step()
    w_after = controller.q.trunk[0].weight.clone()

    # Verify weights actually updated (non-zero gradient step)
    assert not torch.equal(w_before, w_after)
