"""The reference body and the calibrated probe must lie along the same axis.

``_compute_calibration_target_world`` aims the calibrated probe's long axis at
the patient's head direction, preferring the axes the host supplies via
``set_calibration_body_axes``. ``_capture_body_world_orientation`` orients the
static reference body. When only the first honoured the supplied axes, the two
disagreed by the angle between the patient's head direction and the camera --
75 deg at the cardiac widget's shipped view (azimuth 105) -- so a probe
calibrated against the patient rendered across a body laid out against the
camera.

Constructed without ``__init__`` on purpose: both methods read three plain
attributes and nothing else, while the constructor loads meshes and needs a GL
context this rule does not.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from guidance.rendering.guidance_widget import GuidanceWidgetRenderer3D  # noqa: E402


def _renderer(camera_azimuth_deg=105.0, elevation_deg=45.0, body_axes=None):
    """A renderer stub carrying only what the two axis methods read."""
    r = GuidanceWidgetRenderer3D.__new__(GuidanceWidgetRenderer3D)
    az, el = np.deg2rad(camera_azimuth_deg), np.deg2rad(elevation_deg)
    d = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    forward = -d / np.linalg.norm(d)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    r.camera_pose = np.eye(4)
    r.camera_pose[:3, 0] = right
    r.camera_pose[:3, 1] = up
    r.camera_pose[:3, 2] = -forward
    r.calibration_head_dir_world = None if body_axes is None else np.asarray(body_axes[0], float)
    r.calibration_up_world = None if body_axes is None else np.asarray(body_axes[1], float)
    return r


# Scene axes as the cardiac host supplies them: head at -X, anterior at +Y.
PATIENT_AXES = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _long_axes(r):
    """(probe long axis, body long axis) in world coords."""
    probe_long = r._compute_calibration_target_world()[:, 0]   # IMU +X = tip
    body_long = r._capture_body_world_orientation()[:, 1]      # body +Y = head
    return probe_long, body_long


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _sin_angle(a, b):
    """Parallelism measured by |a x b|, not by arccos(a.b).

    arccos is ill-conditioned at exactly the answer we want: two bit-identical
    unit vectors come back ~1e-4 deg apart purely from the dot product landing
    one ulp below 1. The cross product is well-conditioned there."""
    return float(np.linalg.norm(np.cross(a, b)))


@pytest.mark.parametrize('azimuth', [105.0, 135.0, 0.0, 250.0])
def test_body_long_axis_is_parallel_to_the_calibrated_probe(azimuth):
    """The whole point: at ANY camera angle, once the patient's axes are known
    the body lies along the axis the probe is calibrated to."""
    probe_long, body_long = _long_axes(_renderer(azimuth, body_axes=PATIENT_AXES))
    assert _sin_angle(probe_long, body_long) < 1e-12
    assert np.dot(probe_long, body_long) > 0, "parallel but pointing opposite ways"


def test_without_patient_axes_both_fall_back_to_the_camera_together():
    """No supplied axes is not a failure -- it just means the view is the only
    reference there is. They must still agree with each other."""
    probe_long, body_long = _long_axes(_renderer(105.0))
    assert _sin_angle(probe_long, body_long) < 1e-12
    assert np.dot(probe_long, body_long) > 0


def test_the_camera_fallback_really_does_track_the_camera():
    """Guards the test above from passing because nothing depends on anything."""
    _, body_105 = _long_axes(_renderer(105.0))
    _, body_135 = _long_axes(_renderer(135.0))
    assert _angle_deg(body_105, body_135) > 25.0


def test_patient_axes_override_the_camera_rather_than_blending_with_it():
    """The body must follow the patient at every view, not drift with it."""
    _, body_a = _long_axes(_renderer(105.0, body_axes=PATIENT_AXES))
    _, body_b = _long_axes(_renderer(250.0, body_axes=PATIENT_AXES))
    assert _sin_angle(body_a, body_b) < 1e-12
    assert np.allclose(body_a, [-1.0, 0.0, 0.0], atol=1e-12)


def test_the_body_basis_is_a_rotation_even_for_skewed_input_axes():
    """A host deriving 'head' and 'anterior' independently can hand over a pair
    that is not quite perpendicular; a skewed basis is not a rotation and would
    shear the mesh. The HEAD axis is the one kept exact, since that is what has
    to stay parallel to the probe."""
    head = np.array([-0.933, -0.356, 0.0])       # a bed-stage head dir, not _|_ to up
    head /= np.linalg.norm(head)
    r = _renderer(105.0, body_axes=(head, (0.0, 1.0, 0.0)))
    R = r._capture_body_world_orientation()
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9), "basis is not orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9), "basis is not right-handed"
    assert np.allclose(R[:, 1], head, atol=1e-9), "head axis was not preserved"


def test_supplying_axes_marks_the_body_for_re_laying_on_the_render_thread():
    """Orienting from the patient is worthless if it never reaches the scene --
    and it must not be pushed from the caller's thread, which is the host's pose
    callback. Pins the deferral, not just the maths."""
    import threading
    r = _renderer(105.0)
    r._live_lock = threading.Lock()
    r._body_orientation_dirty = False
    r._pending_body_orientation = None
    r.set_calibration_body_axes(*PATIENT_AXES)
    assert r._body_orientation_dirty, "render thread was never told to re-lay it"
    assert r._pending_body_orientation is None, "should recompute, not use a stale pose"
