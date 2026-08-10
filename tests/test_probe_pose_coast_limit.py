"""The ESKF coast limit: how long a degenerate head may keep the cue alive.

`ProbePosePredictor.predict` never feeds a degenerate (det<=0) raw rotation to
the filter. It extrapolates instead, which is right for isolated dropouts and
wrong forever: without a bound it reports valid=True indefinitely, so a head
that has stopped producing sane rotations still drives a confident-looking
ghost probe built purely from filter momentum.

These tests drive the real `predict()` with a stubbed head so the degenerate/
valid sequence is exact and deterministic — the behaviour under test is the
state machine around the filter, not the network.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from inference.probe_pose.probe_pose_predictor import ProbePosePredictor  # noqa: E402

A4C = 2
FRAME = np.zeros((64, 64), dtype=np.uint8)

# Frame-sequence literals below read as a timeline, so name the two states
# rather than writing bare booleans — a bare True is ambiguous about whether it
# means "good frame" or "degenerate frame".
GOOD = False   # head returned a usable rotation (det > 0)
BAD = True     # head returned a degenerate one (det <= 0) -> filter coasts


class _StubHead:
    """Returns a caller-controlled 12-vector [R(9) | T(3)].

    `degenerate` swaps the identity rotation for an all-zero block, whose
    determinant is 0 — the `det(R_raw) > 0` test the predictor uses to decide
    whether the head produced anything usable.
    """

    def __init__(self):
        self.degenerate = False

    def __call__(self, image_tensor, onehot):
        R = np.zeros((3, 3)) if self.degenerate else np.eye(3)
        return torch.tensor(
            np.concatenate([R.flatten(), [0.0, 0.0, 0.0]]),
            dtype=torch.float32,
        ).unsqueeze(0)


class _StubNormalizer:
    def denormalize(self, pose):
        return pose


def _predictor(max_coast_frames):
    """A predictor wired for the filter path, with the network stubbed out."""
    p = object.__new__(ProbePosePredictor)
    p.device = "cpu"
    p.mirror_input = False
    p.conditioning_dim = 12
    p.autocast_fp16 = False
    p.model = _StubHead()
    p.normalizer = _StubNormalizer()
    p.use_filter = True
    p._filter_kwargs = {}
    p._filter_view_idx = None
    p._last_ts = None
    p._coast_frames = 0
    p.max_coast_frames = max_coast_frames

    from inference.probe_pose.filter.pose_filter import PoseFilter
    p._filter = PoseFilter()
    return p


def _run(p, timeline, t0=0.0, dt=0.0625):
    """Feed one frame per GOOD/BAD entry; return the `valid` each produced."""
    out = []
    for i, deg in enumerate(timeline):
        p.model.degenerate = deg
        pred = p.predict(FRAME, A4C, t0 + i * dt)
        assert pred is not None, f"frame {i} returned None"
        out.append(pred.valid)
    return out


def test_isolated_dropouts_still_coast():
    """A short degenerate run is bridged — that is the feature, not the bug."""
    p = _predictor(max_coast_frames=5)
    valid = _run(p, [GOOD, GOOD, BAD, BAD, GOOD, GOOD, BAD])
    assert all(valid), f"short dropouts should stay valid, got {valid}"


def test_sustained_degeneracy_stops_being_valid():
    """Past the limit the cue is filter momentum, so it must report invalid."""
    p = _predictor(max_coast_frames=3)
    valid = _run(p, [GOOD] + [BAD] * 6)

    assert valid[0] is True
    # frames 1..3 are coast 1,2,3 -> within the limit
    assert valid[1:4] == [True, True, True], valid
    # frames 4..6 are coast 4,5,6 -> past it
    assert valid[4:] == [False, False, False], valid


def test_counter_resets_after_a_good_frame():
    """One usable frame re-earns a full coast budget."""
    p = _predictor(max_coast_frames=2)
    _run(p, [GOOD, BAD, BAD, BAD])                # exhausts the budget
    assert p.coast_frames == 3

    assert _run(p, [GOOD]) == [True]
    assert p.coast_frames == 0
    # ...and the budget is whole again
    assert _run(p, [BAD, BAD]) == [True, True]


def test_view_switch_resets_the_counter():
    """The filter is rebuilt on a view change, so its coast run restarts too."""
    p = _predictor(max_coast_frames=2)
    _run(p, [GOOD, BAD, BAD, BAD])
    assert p.coast_frames == 3

    p.model.degenerate = GOOD
    p.predict(FRAME, A4C + 1, 10.0)               # different target view
    assert p.coast_frames == 0


def test_none_restores_unbounded_coasting():
    """Escape hatch: the previous behaviour is still reachable."""
    p = _predictor(max_coast_frames=None)
    valid = _run(p, [GOOD] + [BAD] * 30)
    assert all(valid), "max_coast_frames=None should never invalidate"


def test_zero_disables_coasting_entirely():
    p = _predictor(max_coast_frames=0)
    valid = _run(p, [GOOD, BAD, BAD])
    assert valid == [True, False, False], valid
