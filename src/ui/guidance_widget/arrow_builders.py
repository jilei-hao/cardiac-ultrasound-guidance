"""
Procedural flat-arrow builders for the guidance widget.

- ``create_straight_arrow``: rectangular shaft + triangular head, extruded
  to a small thickness, for translation cues.
- ``create_curved_arrow``: ribbon following a circular arc + tangent
  triangular head, all in one plane with a small out-of-plane thickness,
  for rotation cues.

Both return a single ``trimesh.Trimesh`` so they slot into the same
``pyrender.Mesh.from_trimesh`` flow as the coordinate axes. Coordinates
are unitless — pass values in whatever unit the caller's scene uses.
"""

from typing import Optional, Tuple

import numpy as np
import trimesh


def create_straight_arrow(
    length: float = 0.10,
    shaft_width: float = 0.012,
    shaft_thickness: float = 0.004,
    head_length: float = 0.025,
    head_width: float = 0.024,
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    flat_normal: Optional[Tuple[float, float, float]] = None,
) -> trimesh.Trimesh:
    """Build a flat 3D arrow: rectangular shaft + triangular head.

    The arrow's tail sits at the origin and the tip sits at
    ``length * normalized(direction)``. The flat plane is perpendicular
    to ``flat_normal`` (small ``shaft_thickness`` along that axis); set
    ``flat_normal`` to control which way the flat face points (e.g.,
    ``(1, 0, 0)`` to have all four cardinal arrows lie flat in the YZ
    plane).

    Args:
        length: Total tail-to-tip distance (head_length is included).
        shaft_width: In-plane width of the rectangular shaft.
        shaft_thickness: Out-of-plane thickness of the entire arrow.
        head_length: Length of the triangular head, included in ``length``.
        head_width: Base width of the triangle (typically ~2x shaft_width).
        direction: 3-vector the arrow points along; auto-normalized.
        flat_normal: Out-of-plane axis. If None, an arbitrary axis
            perpendicular to ``direction`` is picked.

    Returns:
        Trimesh of the combined shaft + head.
    """
    head_length = min(head_length, length)
    shaft_length = max(length - head_length, 0.0)
    half_sw = shaft_width / 2.0
    half_hw = head_width / 2.0

    # 2D outline in XY: arrow length runs along +Y, width along ±X.
    # ``extrude_triangulation`` extrudes along Z, which becomes the
    # thickness axis; the arrow is centered on z=0 below.
    verts_2d = np.array([
        [-half_sw, 0.0],            # 0: shaft tail-left
        [+half_sw, 0.0],            # 1: shaft tail-right
        [+half_sw, shaft_length],   # 2: shaft top-right
        [+half_hw, shaft_length],   # 3: head base outer-right (shoulder)
        [0.0,      length],         # 4: tip (apex)
        [-half_hw, shaft_length],   # 5: head base outer-left (shoulder)
        [-half_sw, shaft_length],   # 6: shaft top-left
    ])
    # CCW triangulation. Vertices 2 and 6 (shaft top corners) lie on the
    # head triangle's base (3..5), so the head fan-triangulates from the
    # apex through 5→6→2→3 to keep the shoulders connected to the shaft
    # without T-junctions; the resulting mesh is one watertight piece.
    faces_2d = np.array([
        [0, 1, 2],   # shaft right
        [0, 2, 6],   # shaft left
        [5, 6, 4],   # head left wing
        [6, 2, 4],   # head middle (shares edge 2–6 with shaft top)
        [2, 3, 4],   # head right wing
    ])

    arrow = trimesh.creation.extrude_triangulation(
        verts_2d, faces_2d, height=shaft_thickness
    )
    # ``extrude_triangulation`` puts the bottom face at z=0; recenter so
    # thickness straddles z=0 for symmetric lighting and easy positioning.
    arrow.apply_translation((0.0, 0.0, -shaft_thickness / 2.0))

    arrow.apply_transform(_orient_local_yz(direction, flat_normal))
    return arrow


def create_curved_arrow(
    arc_radius: float = 0.05,
    arc_angle_deg: float = 200.0,
    ribbon_width: float = 0.012,
    thickness: float = 0.004,
    head_length: float = 0.025,
    head_width: float = 0.024,
    plane_normal: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    start_dir: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    arc_segments: int = 48,
    orientation: str = 'flat',
) -> trimesh.Trimesh:
    """Build a flat curved arrow: ribbon along a circular arc + tangent head.

    The arc lies in the plane perpendicular to ``plane_normal`` with its
    center at the origin. ``start_dir`` is the direction (within the
    plane) from the arc's center to the start of the ribbon. The arrow
    sweeps right-handed around ``plane_normal`` for ``arc_angle_deg``;
    the last bit of arc is reserved for the triangular head so the tip
    isn't a flat ribbon edge.

    Args:
        arc_radius: Distance from origin to the ribbon's centerline.
        arc_angle_deg: Total sweep angle, head included.
        ribbon_width: Radial width of the ribbon (in-plane).
        thickness: Out-of-plane thickness (along ``plane_normal``).
        head_length: Arc-length consumed by the triangular head.
        head_width: Radial width of the head's base (~2x ribbon_width).
        plane_normal: Axis the arc sweeps around (right-hand rule).
        start_dir: Vector from origin to the ribbon's start point;
            projected onto the plane perpendicular to ``plane_normal``.
        arc_segments: Number of samples along the ribbon arc.
        orientation: ``'flat'`` (default) bends along the thin edge — the
            wide ribbon face stays parallel to the arc plane, like a flag
            laid flat on a table. ``'tall'`` bends along the flat face —
            the wide ribbon face stands perpendicular to the arc plane,
            like a strip of tape standing on edge and curling around the
            rotation axis. In ``'tall'`` mode ``ribbon_width`` is the
            extent along ``plane_normal`` (tall side) and ``thickness``
            is the radial extent in the arc plane (thin side).

    Returns:
        Trimesh of the combined ribbon + triangular head.
    """
    if orientation == 'tall':
        return _build_curved_arrow_tall(
            arc_radius=arc_radius,
            arc_angle_deg=arc_angle_deg,
            ribbon_width=ribbon_width,
            thickness=thickness,
            head_length=head_length,
            head_width=head_width,
            plane_normal=plane_normal,
            start_dir=start_dir,
            arc_segments=arc_segments,
        )
    if orientation != 'flat':
        raise ValueError(
            f"orientation must be 'flat' or 'tall', got {orientation!r}"
        )

    arc_total_rad = np.deg2rad(arc_angle_deg)
    head_arc_rad = head_length / max(arc_radius, 1e-9)
    # Cap the head's arc-length so it never eats the whole sweep.
    head_arc_rad = min(head_arc_rad, arc_total_rad * 0.95)
    tube_end_rad = arc_total_rad - head_arc_rad

    r_outer = arc_radius + ribbon_width / 2.0
    r_inner = arc_radius - ribbon_width / 2.0
    r_outer_head = arc_radius + head_width / 2.0
    r_inner_head = arc_radius - head_width / 2.0

    n = max(2, arc_segments)
    thetas = np.linspace(0.0, tube_end_rad, n)

    # 2D vertex layout (XY plane). Local frame: arc center at origin,
    # start direction along +X, sweep CCW (toward +Y), thickness along Z.
    # Indexing:
    #   [0 .. n-1]      outer-arc samples (last one == ribbon outer end)
    #   [n .. 2n-1]     inner-arc samples (last one == ribbon inner end)
    #   2n              head outer-shoulder (radius r_outer_head, angle θ_e)
    #   2n+1            head apex
    #   2n+2            head inner-shoulder (radius r_inner_head, angle θ_e)
    verts_2d = np.empty((2 * n + 3, 2), dtype=float)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    verts_2d[0:n, 0] = r_outer * cos_t
    verts_2d[0:n, 1] = r_outer * sin_t
    verts_2d[n:2 * n, 0] = r_inner * cos_t
    verts_2d[n:2 * n, 1] = r_inner * sin_t
    cos_e, sin_e = np.cos(tube_end_rad), np.sin(tube_end_rad)
    cos_a, sin_a = np.cos(arc_total_rad), np.sin(arc_total_rad)
    verts_2d[2 * n + 0] = (r_outer_head * cos_e, r_outer_head * sin_e)
    verts_2d[2 * n + 1] = (arc_radius * cos_a, arc_radius * sin_a)
    verts_2d[2 * n + 2] = (r_inner_head * cos_e, r_inner_head * sin_e)

    # Ribbon strip + head fan from apex. The head fan sweeps from the
    # inner shoulder through both ribbon-end vertices to the outer
    # shoulder, sharing the ribbon's end-cap edge so the whole arrow is
    # one watertight piece (no T-junction at the shoulder).
    ribbon_outer_end = n - 1
    ribbon_inner_end = 2 * n - 1
    head_outer = 2 * n
    apex = 2 * n + 1
    head_inner = 2 * n + 2

    faces_2d = []
    for i in range(n - 1):
        outer_i, outer_j = i, i + 1
        inner_i, inner_j = n + i, n + i + 1
        faces_2d.append([outer_i, outer_j, inner_j])
        faces_2d.append([outer_i, inner_j, inner_i])
    faces_2d.append([head_inner, ribbon_inner_end, apex])
    faces_2d.append([ribbon_inner_end, ribbon_outer_end, apex])
    faces_2d.append([ribbon_outer_end, head_outer, apex])
    faces_2d = np.asarray(faces_2d)

    arrow = trimesh.creation.extrude_triangulation(
        verts_2d, faces_2d, height=thickness
    )
    arrow.apply_translation((0.0, 0.0, -thickness / 2.0))

    arrow.apply_transform(_orient_local_xz(plane_normal, start_dir))
    return arrow


def _build_curved_arrow_tall(
    arc_radius: float,
    arc_angle_deg: float,
    ribbon_width: float,
    thickness: float,
    head_length: float,
    head_width: float,
    plane_normal: Tuple[float, float, float],
    start_dir: Tuple[float, float, float],
    arc_segments: int,
) -> trimesh.Trimesh:
    """Curved arrow bent along its flat face.

    The cross-section at every arc sample is the same rectangle — wide
    along ``plane_normal`` (``ribbon_width``), thin along the radial
    direction in the arc plane (``thickness``). The head is a wedge that
    starts from a (possibly wider) ``head_width`` rectangle at the base
    and tapers to an apex line in the radial direction.

    Built directly from explicit vertex/face arrays (no
    ``extrude_triangulation``) because the cross-section frame rotates
    with the tangent — there is no constant 2D plane that captures the
    sweep.
    """
    n_dir = _normalize(plane_normal)
    s_in = np.asarray(start_dir, dtype=float)
    s_dir = s_in - n_dir * float(np.dot(s_in, n_dir))
    if float(np.linalg.norm(s_dir)) < 1e-9:
        raise ValueError(
            "start_dir is parallel to plane_normal; pick a perpendicular "
            "start_dir."
        )
    s_dir = s_dir / float(np.linalg.norm(s_dir))
    t_dir = np.cross(n_dir, s_dir)

    arc_total_rad = np.deg2rad(arc_angle_deg)
    head_arc_rad = head_length / max(arc_radius, 1e-9)
    head_arc_rad = min(head_arc_rad, arc_total_rad * 0.95)
    tube_end_rad = arc_total_rad - head_arc_rad

    n = max(2, arc_segments)
    thetas = np.linspace(0.0, tube_end_rad, n)

    half_thin = thickness / 2.0
    half_wide = ribbon_width / 2.0
    # If head_width < ribbon_width we'd get an "inverted shoulder"
    # (ribbon overhangs head); just clamp up so the head is at least as
    # wide as the ribbon.
    half_head = max(head_width, ribbon_width) / 2.0

    # Vertex layout. Per ribbon sample i the four corners are at
    # 4*i + k for k in {0:outer_top, 1:outer_bot, 2:inner_top, 3:inner_bot}
    # where "outer/inner" is along ±radial and "top/bottom" is along
    # ±plane_normal. After the ribbon, four head-base vertices and two
    # apex vertices.
    verts = np.empty((4 * n + 6, 3), dtype=float)
    for i, theta in enumerate(thetas):
        r_dir = np.cos(theta) * s_dir + np.sin(theta) * t_dir
        center = arc_radius * r_dir
        verts[4 * i + 0] = center + r_dir * half_thin + n_dir * half_wide
        verts[4 * i + 1] = center + r_dir * half_thin - n_dir * half_wide
        verts[4 * i + 2] = center - r_dir * half_thin + n_dir * half_wide
        verts[4 * i + 3] = center - r_dir * half_thin - n_dir * half_wide

    end_r_dir = np.cos(tube_end_rad) * s_dir + np.sin(tube_end_rad) * t_dir
    end_center = arc_radius * end_r_dir
    apex_r_dir = np.cos(arc_total_rad) * s_dir + np.sin(arc_total_rad) * t_dir
    apex_center = arc_radius * apex_r_dir

    H_OUT_TOP = 4 * n + 0
    H_OUT_BOT = 4 * n + 1
    H_IN_TOP = 4 * n + 2
    H_IN_BOT = 4 * n + 3
    A_OUT = 4 * n + 4
    A_IN = 4 * n + 5

    verts[H_OUT_TOP] = end_center + end_r_dir * half_thin + n_dir * half_head
    verts[H_OUT_BOT] = end_center + end_r_dir * half_thin - n_dir * half_head
    verts[H_IN_TOP] = end_center - end_r_dir * half_thin + n_dir * half_head
    verts[H_IN_BOT] = end_center - end_r_dir * half_thin - n_dir * half_head
    verts[A_OUT] = apex_center + apex_r_dir * half_thin
    verts[A_IN] = apex_center - apex_r_dir * half_thin

    faces = []

    # Ribbon strip — 4 side faces between consecutive samples. Windings
    # were derived by computing the cross product of the first two edges
    # for a sample example; the same indexing works at every θ because
    # the cross-section frame is consistent.
    for i in range(n - 1):
        a = 4 * i
        b = 4 * (i + 1)
        # +plane_normal (top) face
        faces.append([a + 0, b + 0, b + 2])
        faces.append([a + 0, b + 2, a + 2])
        # -plane_normal (bottom) face
        faces.append([a + 1, a + 3, b + 3])
        faces.append([a + 1, b + 3, b + 1])
        # +radial (outer) face
        faces.append([a + 0, b + 1, b + 0])
        faces.append([a + 0, a + 1, b + 1])
        # -radial (inner) face
        faces.append([a + 2, b + 2, b + 3])
        faces.append([a + 2, b + 3, a + 3])

    # Start cap at θ=0, normal = -tangent at θ=0 = -t_dir.
    faces.append([0, 2, 3])
    faces.append([0, 3, 1])

    # Shoulder strips at θ_end. Only meaningful when head_width >
    # ribbon_width; otherwise fold the head onto the ribbon's end-cap
    # vertices so there's no gap.
    if half_head > half_wide + 1e-9:
        rt_outer = 4 * (n - 1) + 0
        rt_inner = 4 * (n - 1) + 2
        rb_outer = 4 * (n - 1) + 1
        rb_inner = 4 * (n - 1) + 3
        # Top shoulder rectangle, normal -tangent at θ_end.
        faces.append([rt_outer, H_OUT_TOP, H_IN_TOP])
        faces.append([rt_outer, H_IN_TOP, rt_inner])
        # Bottom shoulder rectangle, normal -tangent at θ_end.
        faces.append([rb_outer, H_IN_BOT, H_OUT_BOT])
        faces.append([rb_outer, rb_inner, H_IN_BOT])
    else:
        # Re-point head base to ribbon-end indices; the originally placed
        # H_* vertices become unused (process=True drops them).
        H_OUT_TOP = 4 * (n - 1) + 0
        H_OUT_BOT = 4 * (n - 1) + 1
        H_IN_TOP = 4 * (n - 1) + 2
        H_IN_BOT = 4 * (n - 1) + 3

    # Head — wedge from a rectangle (at θ_end) to an apex line (at θ_apex,
    # axial=0). Top/bottom slopes are always 2 triangles each. The
    # ±radial sides are split into 3 triangles each when a shoulder
    # exists, so their base edges pass through ``rt_outer``/``rb_outer``
    # (resp. ``rt_inner``/``rb_inner``) and match the shoulder edges
    # without leaving T-junctions; otherwise a single triangle suffices.
    faces.append([H_OUT_TOP, A_OUT, A_IN])
    faces.append([H_OUT_TOP, A_IN, H_IN_TOP])
    faces.append([H_OUT_BOT, A_IN, A_OUT])
    faces.append([H_OUT_BOT, H_IN_BOT, A_IN])
    if half_head > half_wide + 1e-9:
        rt_outer_v = 4 * (n - 1) + 0
        rb_outer_v = 4 * (n - 1) + 1
        rt_inner_v = 4 * (n - 1) + 2
        rb_inner_v = 4 * (n - 1) + 3
        # +radial side
        faces.append([H_OUT_TOP, rt_outer_v, A_OUT])
        faces.append([rt_outer_v, rb_outer_v, A_OUT])
        faces.append([rb_outer_v, H_OUT_BOT, A_OUT])
        # -radial side
        faces.append([H_IN_TOP, A_IN, rt_inner_v])
        faces.append([rt_inner_v, A_IN, rb_inner_v])
        faces.append([rb_inner_v, A_IN, H_IN_BOT])
    else:
        faces.append([H_OUT_TOP, H_OUT_BOT, A_OUT])
        faces.append([H_IN_TOP, A_IN, H_IN_BOT])

    return trimesh.Trimesh(
        vertices=verts, faces=np.asarray(faces), process=True,
    )


def _orient_local_yz(
    direction, flat_normal: Optional[Tuple[float, float, float]]
) -> np.ndarray:
    """4x4 rotation that maps local +Y → ``direction`` and local +Z →
    ``flat_normal`` (projected onto the plane perpendicular to ``direction``).

    If ``flat_normal`` is None or parallel to ``direction``, only the +Y
    alignment is applied; the +Z axis ends up on whatever perpendicular
    falls out of that rotation.
    """
    direction = _normalize(direction)
    R1 = _rotation_between(np.array([0.0, 1.0, 0.0]), direction)

    if flat_normal is None:
        return _to_4x4(R1)

    fn = np.asarray(flat_normal, dtype=float)
    fn_perp = fn - direction * float(np.dot(fn, direction))
    fn_norm = float(np.linalg.norm(fn_perp))
    if fn_norm < 1e-9:
        return _to_4x4(R1)
    fn_perp = fn_perp / fn_norm

    current_z = R1 @ np.array([0.0, 0.0, 1.0])
    R2 = _rotation_around_axis(direction, current_z, fn_perp)
    return _to_4x4(R2 @ R1)


def _orient_local_xz(plane_normal, start_dir) -> np.ndarray:
    """4x4 rotation: maps local +Z → ``plane_normal`` and local +X →
    ``start_dir`` (projected onto the plane perpendicular to ``plane_normal``).
    """
    n = _normalize(plane_normal)
    R1 = _rotation_between(np.array([0.0, 0.0, 1.0]), n)

    s = np.asarray(start_dir, dtype=float)
    s_perp = s - n * float(np.dot(s, n))
    s_norm = float(np.linalg.norm(s_perp))
    if s_norm < 1e-9:
        return _to_4x4(R1)
    s_perp = s_perp / s_norm

    current_x = R1 @ np.array([1.0, 0.0, 0.0])
    R2 = _rotation_around_axis(n, current_x, s_perp)
    return _to_4x4(R2 @ R1)


def _rotation_around_axis(
    axis: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """3x3 rotation around ``axis`` that sends ``src`` to ``dst`` (both
    assumed unit-length and lying in the plane perpendicular to ``axis``).
    Uses the signed angle so the rotation goes the short way around.
    """
    cos_a = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    sin_a = float(np.dot(np.cross(src, dst), axis))
    return _rodrigues(axis, float(np.arctan2(sin_a, cos_a)))


def _normalize(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        raise ValueError("zero-length vector")
    return a / norm


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation sending unit-vector ``a`` to unit-vector ``b``."""
    cos_a = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cos_a > 1.0 - 1e-12:
        return np.eye(3)
    if cos_a < -1.0 + 1e-12:
        # 180-degree rotation; pick any axis perpendicular to ``a``.
        helper = (
            np.array([0.0, 1.0, 0.0])
            if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        )
        axis = np.cross(a, helper)
    else:
        axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)
    return _rodrigues(axis, float(np.arccos(cos_a)))


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    K = np.array([
        [0.0,     -axis[2], axis[1]],
        [axis[2],  0.0,    -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _to_4x4(R: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    return M


if __name__ == "__main__":
    # Quick visual sanity check via trimesh's built-in viewer.
    #   python -m ui.guidance_widget.arrow_builders            # straight
    #   python -m ui.guidance_widget.arrow_builders curved     # curved
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "straight"
    if which == "curved":
        create_curved_arrow().show()
    else:
        create_straight_arrow().show()
