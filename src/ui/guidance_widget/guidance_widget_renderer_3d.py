"""
3D guidance widget renderer using Generic3DRenderer.
Provides 3D visualization of probe orientation and coordinate axes.
"""

import os
import threading
import numpy as np
import trimesh
import time
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Optional, Union

from .arrow_builders import create_curved_arrow, create_straight_arrow
from ..generic_3d_renderer import (
    Generic3DRenderer,
    _resolve_meters_per_unit,
    create_translation_matrix,
    create_rotation_matrix,
    quaternion_to_matrix,
)


# Pose used to "hide" a node: shoves it far enough off-screen that it
# falls outside any reasonable view frustum. Cheaper than removing /
# re-adding the node every frame.
_HIDDEN_POSE = np.eye(4)
_HIDDEN_POSE[:3, 3] = [1e4, 1e4, 1e4]


# Arrow direction labels used in the public API and as node-name suffixes.
# Values are unit vectors in the IMU body frame (post probe_axis_fix):
#   +X = probe long axis toward the tip
#   ±Y, ±Z = perpendicular plane (lateral / back-of-probe)
# Translation arrows point in the perpendicular plane outward from the
# probe tip; rotation arrows curl around those same perpendicular axes
# at the probe tail. The screen-direction labels (left/right/up/down) map
# to the body-frame axes that visually correspond to those directions in
# the default camera view.
_TRANSLATION_DIRS = {
    "right": np.array([0.0,  1.0, 0.0]),
    "left":  np.array([0.0, -1.0, 0.0]),
    "up":    np.array([0.0, 0.0,  1.0]),
    "down":  np.array([0.0, 0.0, -1.0]),
}
_ROTATION_AXES = {
    "right": np.array([0.0, 0.0,  1.0]),
    "left":  np.array([0.0, 0.0, -1.0]),
    "up":    np.array([0.0,  1.0, 0.0]),
    "down":  np.array([0.0, -1.0, 0.0]),
}
# Roll (long-axis) curl arrows. Convention: viewer looks from probe tail
# toward tip (i.e. along +X in the IMU body frame). Under that view,
# right-hand rotation about +X sweeps CCW; right-hand rotation about -X
# sweeps CW. Map sign to label via the right-hand-rule axis.
_ROLL_AXES = {
    "ccw": np.array([ 1.0, 0.0, 0.0]),
    "cw":  np.array([-1.0, 0.0, 0.0]),
}


# Coordinate-axis base colours (RGB, 0-1). Single source of truth shared by the
# 3D axis materials (``_create_materials``) and the 2D on-frame legend
# (``_draw_guidance_legend``), so a legend swatch can never drift from the axis
# it labels. Tuple order is the on-screen / legend column order.
_AXIS_ORDER = ("x", "y", "z")
_AXIS_RGB = {
    "x": (1.0, 0.2, 0.2),   # red   — probe long axis
    "y": (0.2, 1.0, 0.2),   # green — perpendicular plane (lateral)
    "z": (0.2, 0.2, 1.0),   # blue  — perpendicular plane (back-of-probe)
}
# Opacity of the semi-transparent "predicted" axes (BLEND alpha).
_PREDICTED_AXIS_ALPHA = 0.3


# Legend label font, tried in order. Labels render as real anti-aliased
# TrueType via PIL instead of the technical-looking cv2 Hershey stroke font —
# the target users are clinicians, not engineers. The primary is Quicksand, a
# rounded geometric sans chosen for a friendly, non-technical look (Bold reads
# best against the dark panel at small sizes). The fallbacks mirror the
# vigil-system guidance *display* font stack (Inter, "Segoe UI", Arial,
# sans-serif — see display/vigil_display_utils/Style.py & GuidanceWidget.py), so
# where Quicksand is unavailable the legend still matches the operator UI: Inter
# (the house font), else the Arial substitute (Liberation Sans), else the
# generic sans (DejaVu). If nothing resolves, the legend falls back to the cv2
# Hershey font so rendering never breaks. Override with the GUIDANCE_LEGEND_FONT
# env var (absolute font path).
_LEGEND_FONT_CANDIDATES = (
    os.environ.get("GUIDANCE_LEGEND_FONT", ""),
    # Quicksand — friendly rounded primary.
    "/usr/share/fonts/truetype/quicksand/Quicksand-Bold.ttf",
    "/usr/share/fonts/truetype/quicksand/Quicksand-Medium.ttf",
    "Quicksand-Bold.ttf",
    # Inter (vigil house font). Per-user install dirs first (no sudo needed),
    # then the system paths a `fonts-inter` apt install would use.
    os.path.expanduser("~/.local/share/fonts/inter/Inter-Regular.otf"),
    os.path.expanduser("~/.fonts/inter/Inter-Regular.otf"),
    "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
    "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
    "/usr/share/fonts/truetype/inter/InterVariable.ttf",
    "Inter-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # ~Arial
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "LiberationSans-Regular.ttf",
    "Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",               # macOS
    "C:\\Windows\\Fonts\\arial.ttf",                      # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # generic sans fallback
    "DejaVuSans.ttf",
)
# (font_path, size) -> PIL ImageFont | None. Module-level so it is shared and
# survives across renderer instances; loading a TrueType face is not free.
_FONT_CACHE: dict = {}


def _load_legend_font(size_px: int):
    """Return a PIL ImageFont near ``size_px``, or None if none resolve."""
    size = max(8, int(size_px))
    for cand in _LEGEND_FONT_CANDIDATES:
        if not cand:
            continue
        key = (cand, size)
        if key not in _FONT_CACHE:
            try:
                from PIL import ImageFont
                _FONT_CACHE[key] = ImageFont.truetype(cand, size)
            except Exception:
                _FONT_CACHE[key] = None
        if _FONT_CACHE[key] is not None:
            return _FONT_CACHE[key]
    return None


def _alpha_blit_rgba(frame_bgra: np.ndarray, sprite_rgba: np.ndarray,
                     x: int, y: int) -> None:
    """Alpha-composite an RGBA sprite onto a BGRA frame at (x, y)."""
    fh, fw = frame_bgra.shape[:2]
    sh, sw = sprite_rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + sw), min(fh, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    sprite = sprite_rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sprite[..., 3:4].astype(np.float32) / 255.0
    src_bgr = sprite[..., 2::-1].astype(np.float32)   # RGB -> BGR
    roi = frame_bgra[y0:y1, x0:x1]
    roi[..., :3] = (
        src_bgr * a + roi[..., :3].astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)
    if frame_bgra.shape[2] == 4:
        roi[..., 3] = np.maximum(roi[..., 3], sprite[..., 3])


def _blit_pil_text(frame_bgra: np.ndarray, text: str, x: int, y: int,
                   font, color_bgr: tuple) -> None:
    """Draw ``text`` (TrueType, anti-aliased) with its visual top-left at
    (x, y) onto the BGRA frame."""
    from PIL import Image, ImageDraw
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top
    if tw <= 0 or th <= 0:
        return
    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(img).text(
        (-left, -top), text, font=font,
        fill=(color_bgr[2], color_bgr[1], color_bgr[0], 255),
    )
    _alpha_blit_rgba(frame_bgra, np.asarray(img), int(x), int(y))


class GuidanceWidgetRenderer3D:
    """
    3D guidance widget renderer with probe mesh and coordinate axes
    visualization. All scene coordinates are in meters.
    """

    def __init__(self, probe_path: str,
                 background_color=(0.1, 0.1, 0.15, 1.0),
                 wireframe=False,
                 window_size=(640, 480),
                 render_mode='fullscreen',
                 rotation_amplification=3.0,
                 legend_image_path: str = "",
                 probe_units: Union[str, float] = 'mm',
                 probe_long_axis_length_m: Optional[float] = 0.12,
                 axes_length_m: float = 0.12,
                 y_axis_length_m: Optional[float] = None,
                 x_axis_length_m: Optional[float] = None,
                 x_axis_from_tail: bool = False,
                 axes_radius_m: float = 0.003,
                 camera_distance_m: Optional[float] = None,
                 camera_zoom: float = 0.68,
                 camera_azimuth_deg: float = 135.0,
                 camera_elevation_deg: float = 45.0,
                 probe_origin_screen_y_from_top: float = 0.5,
                 probe_origin_m: tuple = (0.0, 0.0, 0.0),
                 probe_pivot_local_m: Optional[tuple] = None,
                 probe_pivot_axis: str = 'y',
                 probe_pivot_sign: int = 1,
                 probe_axis_fix_euler_deg: Optional[tuple] = (180.0, 0.0, 90.0),
                 reference_body_path: Optional[str] = None,
                 reference_body_units: Union[str, float] = 'cm',
                 reference_body_scale_ratio: float = 0.75,
                 auto_fit_live_geometry: bool = True,
                 live_geometry_rotation_euler_deg: Optional[tuple] = None,
                 contact_marker_radius_m: float = 0.006,
                 contact_marker_color: tuple = (1.0, 0.25, 0.2, 1.0),
                 place_probe_at_world_pos: bool = False,
                 translation_arrow_length_m: float = 0.09,
                 translation_arrow_offset_m: float = 0.025,
                 rotation_arrow_radius_m: float = 0.0375,
                 rotation_arrow_offset_m: float = 0.025,
                 rotation_arrow_angle_deg: float = 200.0,
                 alignment_color_max_deg: float = 8.0,
                 body_mesh_color: tuple = (0.85, 0.78, 0.72, 1.0),
                 body_mesh_metallic: float = 0.1,
                 body_mesh_roughness: float = 0.85,
                 ghost_probe_color: tuple = (0.6, 0.85, 1.0),
                 ghost_probe_alpha: float = 0.35,
                 debug_print: bool = False):
        """
        Initialize 3D guidance widget renderer.

        Args:
            probe_path: Path to the probe mesh file (STL or OBJ).
            background_color: Background color as RGBA tuple.
            wireframe: Whether to render in wireframe mode.
            window_size: Window dimensions (width, height) in pixels.
            render_mode: 'fullscreen' or 'overlay'.
            rotation_amplification: Amplification factor for rotation
                prediction.
            legend_image_path: Optional path to a legend image overlay.
            probe_units: Native unit of the probe mesh ('mm', 'cm', 'dm',
                'm', 'in', or a meters-per-unit float). Default 'mm'.
            probe_long_axis_length_m: Target extent (in meters) along
                ``probe_pivot_axis`` after units conversion. The probe
                sub-meshes are uniformly scaled to match this length, so
                widget geometry (axes, arrows, camera fit) stays
                identical across mesh swaps. Pass None to skip
                normalization and use the asset's native size.
            axes_length_m: Length of each coordinate-axis shaft in meters.
            y_axis_length_m: Optional override for the Y-axis shaft length
                in meters. Defaults to ``axes_length_m`` so all three
                axes appear the same length. Set explicitly if you want
                the green axis to stick out past the probe body.
            axes_radius_m: Shaft radius of each coordinate axis in meters.
            camera_distance_m: Camera distance from the scene origin in
                meters. If None, auto-fitted from the scene AABB.
            camera_zoom: Multiplicative zoom factor applied on top of the
                camera distance. >1 moves the camera closer (zoom in);
                <1 moves it farther (zoom out). Default 1.0.
            camera_azimuth_deg: Camera azimuth (rotation around +Y axis)
                in degrees.
            camera_elevation_deg: Camera elevation (pitch above the XZ
                plane) in degrees.
            probe_origin_m: Reserved translation for positional tracking;
                currently defaults to the origin.
            probe_pivot_local_m: Offset from the AABB-centered mesh origin
                to the physical tip, in mesh-local meters. If None, it is
                auto-computed from ``probe_pivot_axis`` / ``probe_pivot_sign``
                after the mesh loads. Rotations pivot around this point so
                the tip stays planted at ``probe_origin_m``, matching the
                sonographer's physical rotation around the skin-contact tip.
            probe_pivot_axis: Mesh-local axis ('x'|'y'|'z') the probe's
                long dimension runs along. Used only when
                ``probe_pivot_local_m`` is None. Default 'y'.
            probe_pivot_sign: +1 if the tip sits at the max end of
                ``probe_pivot_axis``, -1 for the min end. Default -1.
            probe_axis_fix_euler_deg: Optional (rx, ry, rz) in degrees
                (scipy 'xyz' intrinsic). Pre-rotates mesh-local vertices
                into the Clarius IMU body frame so live quaternions
                rotate the right axes. Default ``(180, 0, 90)`` is the
                fix for the official HD3 mesh: mesh +Y is the long axis
                toward the tip and mesh +Z faces the imaging surface,
                while the Clarius IMU has +X along the long axis toward
                the tip and +Z out the back of the probe (per
                https://github.com/clariusdev/imu, axes-v3.png). The
                composed matrix sends mesh +Y → IMU +X (long axis) and
                mesh +Z → IMU -Z (button-out at +Z). Set None for a
                pre-aligned mesh.
        """
        self.probe_path = probe_path
        self.background_color = background_color
        self.wireframe = wireframe
        self.rotation_amplification = rotation_amplification
        self.window_size = window_size
        self.render_mode = render_mode
        self.legend_image_path = legend_image_path

        self.probe_units = probe_units
        self.probe_long_axis_length_m = (
            float(probe_long_axis_length_m)
            if probe_long_axis_length_m is not None else None
        )
        self.axes_length_m = axes_length_m
        self.y_axis_length_m = (
            float(y_axis_length_m)
            if y_axis_length_m is not None else float(axes_length_m)
        )
        # The X (red) axis is the probe long axis. ``x_axis_from_tail`` draws it
        # along -X (out the probe tail) instead of +X (into the body) and
        # ``x_axis_length_m`` lets it run long, so it stretches out past the
        # probe body instead of being occluded by it.
        self.x_axis_length_m = (
            float(x_axis_length_m)
            if x_axis_length_m is not None else float(axes_length_m)
        )
        self.x_axis_from_tail = bool(x_axis_from_tail)
        self.axes_radius_m = axes_radius_m
        self.camera_distance_m = camera_distance_m
        self.camera_zoom = float(camera_zoom) if camera_zoom else 1.0
        self.camera_azimuth_deg = float(camera_azimuth_deg)
        self.camera_elevation_deg = float(camera_elevation_deg)
        # Captured so ``reset_camera`` can return to the constructor pose
        # after live nudging.
        self._default_camera_zoom = self.camera_zoom
        self._default_camera_azimuth_deg = float(camera_azimuth_deg)
        self._default_camera_elevation_deg = float(camera_elevation_deg)
        self.probe_origin_screen_y_from_top = float(probe_origin_screen_y_from_top)
        self.probe_origin_m = np.asarray(probe_origin_m, dtype=float)
        self.probe_pivot_axis = probe_pivot_axis
        self.probe_pivot_sign = int(probe_pivot_sign)
        self.probe_pivot_local_m = (
            np.asarray(probe_pivot_local_m, dtype=float)
            if probe_pivot_local_m is not None else None
        )
        if probe_axis_fix_euler_deg is None:
            self.probe_axis_fix_matrix = np.eye(3)
        else:
            self.probe_axis_fix_matrix = R.from_euler(
                'xyz', probe_axis_fix_euler_deg, degrees=True
            ).as_matrix()
        # Flip the rendered model 180° around the IMU long axis (+X) so the
        # probe button faces upward in the canonical calibration pose. Applied
        # only to the probe — current/predicted axes still use R3 alone.
        self.probe_axis_fix_matrix = (
            R.from_euler('x', 180, degrees=True).as_matrix()
            @ self.probe_axis_fix_matrix
        )

        self.reference_body_path = reference_body_path
        self.reference_body_units = reference_body_units
        self.reference_body_scale_ratio = float(reference_body_scale_ratio)

        # --- Live patient geometry (Stevens point cloud + SMPL mesh) ---
        # Inputs arrive on the ROS subscription thread via the set_* methods
        # below; all pyrender scene mutation happens on the render thread in
        # ``_apply_live_geometry`` (called from render_widget). The lock only
        # guards the cached payloads + dirty flags.
        self.auto_fit_live_geometry = bool(auto_fit_live_geometry)
        self.place_probe_at_world_pos = bool(place_probe_at_world_pos)
        self.contact_marker_radius_m = float(contact_marker_radius_m)
        self.contact_marker_color = tuple(contact_marker_color)
        if live_geometry_rotation_euler_deg is None:
            self._live_geometry_rotation = np.eye(3)
        else:
            self._live_geometry_rotation = R.from_euler(
                'xyz', live_geometry_rotation_euler_deg, degrees=True
            ).as_matrix()
        self._live_lock = threading.Lock()
        self._live_point_cloud = None      # (points Nx3 world m, colors or None)
        self._live_body = None             # (vertices Nx3 world m, faces Fx3)
        self._probe_world_pos = None       # (3,) probe tip in world m, or None
        self._pc_dirty = False
        self._body_dirty = False
        self._probe_dirty = False
        # world->scene 4x4 (auto-fit from the SMPL mesh extents so the body
        # sits where the static reference used to). None until first body.
        self._world_to_scene = None
        # latest SMPL mesh in *scene* coords, kept for closest-point contact.
        self._live_body_scene_mesh = None
        # whether the live SMPL body node is currently shown (vs static OBJ).
        self._live_body_active = False
        self._contact_marker_loaded = False

        self.translation_arrow_length_m = float(translation_arrow_length_m)
        self.translation_arrow_offset_m = float(translation_arrow_offset_m)
        self.rotation_arrow_radius_m = float(rotation_arrow_radius_m)
        self.rotation_arrow_offset_m = float(rotation_arrow_offset_m)
        self.rotation_arrow_angle_deg = float(rotation_arrow_angle_deg)
        # Geodesic angle (deg) at or above which arrows render fully
        # orange; at 0° they render fully green. Linear ramp between.
        self.alignment_color_max_deg = float(alignment_color_max_deg)
        # --- Configurable visualization look (body mesh + ghost target probe) ---
        # body_mesh_color accepts RGB or RGBA (0-1); normalized to RGBA here.
        # metallic/roughness are PBR knobs for the SMPL body surface.
        # ghost_probe_color is RGB; ghost_probe_alpha sets the ghost transparency.
        _bc = tuple(float(c) for c in body_mesh_color)
        self.body_mesh_color = _bc if len(_bc) == 4 else (_bc[0], _bc[1], _bc[2], 1.0)
        self.body_mesh_metallic = float(body_mesh_metallic)
        self.body_mesh_roughness = float(body_mesh_roughness)
        self.ghost_probe_color = tuple(float(c) for c in ghost_probe_color)[:3]
        self.ghost_probe_alpha = float(ghost_probe_alpha)
        # When on, every prediction-driven arrow update prints a one-line
        # delta summary. Toggled at runtime via the 'd' hotkey.
        self.debug_print = bool(debug_print)

        # Toggleable scene layers. ``target_probe`` overlays a
        # semi-transparent ghost probe at the predicted axes; off by
        # default since the ghost is busy and most useful when actively
        # correcting alignment. All names not listed here are rejected by
        # ``set_layer_visible``. ``current_axes`` and ``predicted_axes``
        # share a default so the 'x' hotkey flips both in lockstep from
        # the very first press.
        self._layers_visible = {
            "current_axes": True,
            "predicted_axes": True,
            "guidance_arrows": True,
            "target_probe": False,
            "reference_body": True,
            "world_pcl": True,
            "contact_marker": True,
        }
        # Stored at load time so the layer toggle can swap between the
        # original screen-corner pose and ``_HIDDEN_POSE``.
        self._reference_body_loaded = False
        self._reference_body_pose = np.eye(4)
        # Body orientation locked in world space so subsequent camera
        # nudges leave the orientation alone — the body then visually
        # rotates with the (also world-anchored) probe. Captured first
        # at construction (right after ``_setup_camera`` runs), and
        # re-captured on every ``calibrate`` call so the body re-aligns
        # with the operator's freshly chosen calibration view.
        # ``_default_body_orientation_world`` is the construction-time
        # snapshot that ``reset_calibration`` reverts to.
        self._body_orientation_world: Optional[np.ndarray] = None
        self._default_body_orientation_world: Optional[np.ndarray] = None
        # World-space distance below the probe tip at which the body
        # sits. Captured lazily on the first pose computation (when the
        # construction camera is in place) and frozen so subsequent
        # camera nudges / zooms leave the body planted under the tip.
        self._body_offset_below_m: Optional[float] = None
        # Per-direction visibility for each arrow group. Multiple
        # rotation arrows can be lit at once when a prediction has both
        # a Y- and Z-axis component (e.g. "rotate up AND right");
        # translation stays single-active for now since translation
        # prediction is not implemented yet.
        self._translation_arrow_visible = {k: False for k in _TRANSLATION_DIRS}
        self._rotation_arrow_visible = {k: False for k in _ROTATION_AXES}
        self._roll_arrow_visible = {k: False for k in _ROLL_AXES}
        # Minimum |Euler angle| (degrees) for a predicted rotation
        # component to light its arrow. Below this, the prediction is
        # treated as noise and the arrow stays hidden.
        self.rotation_arrow_threshold_deg = 1.0
        # --- Translation guidance (USGR probe-pose T_rel) tunables ---
        # Minimum |translation component| (metres, after amplification) for a
        # translation arrow to light. Below this the component is noise.
        self.translation_arrow_threshold_m = 0.005
        # Visual gain applied to the predicted translation before thresholding
        # / lighting arrows (analogous to ``rotation_amplification``).
        self.translation_amplification = 1.0
        # Maps the USGR probe-frame translation axes onto this renderer's IMU
        # body-frame axes (see ``_TRANSLATION_DIRS``). Identity is the starting
        # point; sign/permutation must be calibrated empirically on real frames
        # so "move right/up" arrows point the physically correct way.
        self.usgr_to_body = np.eye(3, dtype=float)
        # Gain applied to the body-frame translation when spatially offsetting
        # the predicted axes + target-probe ghost away from the current pose so
        # the operator sees WHERE to move (not just how to rotate). Raise it to
        # exaggerate small (cm-scale) displacements for visibility, like
        # ``rotation_amplification`` does for rotation.
        self.guidance_translation_amplification = 1.0
        # Last calibrated current-orientation R3 (set each render); used to map
        # the body-frame target translation into render-world.
        self._last_R3 = np.eye(3, dtype=float)
        # Set in _setup_guidance_arrows after the probe geometry is known.
        self._tail_imu = np.zeros(3, dtype=float)

        self.renderer = None
        self.probe_mesh = None
        self.probe_node_specs: list = []
        self.axes = None
        self.materials = None
        self.camera_pose = None
        self.camera_pos = None
        self.camera_fov = np.pi / 4

        # 3x3 left-multiplied onto R_imu so a body-frame rotation of the
        # probe maps to the same body-frame rotation of the rendered
        # probe — independent of how IMU-world axes happen to align with
        # render-world axes. Set by calibrate(); identity until the user
        # calibrates.
        self.calibration_matrix = np.eye(3)

        # Optional fixed body axes (scene/world frame) for the calibration target.
        # When the live SMPL body is placed by an explicit world->scene transform
        # (cardiac face-up mode), the patient's head/anterior directions are fixed
        # in the scene, not camera-relative; calibrate() must target THOSE so the
        # operator's canonical hold (tip superior, button up) maps to the body.
        # None -> fall back to the camera-relative reference-body convention.
        self.calibration_head_dir_world = None   # tip-points-to (patient superior)
        self.calibration_up_world = None         # button-up (patient anterior)

        # The legend is now drawn programmatically at the frame's native
        # resolution (see ``_draw_guidance_legend``) instead of being resampled
        # from a baked PNG — the down-then-up scaling of that PNG was what made
        # it look voxelized. A non-empty ``legend_image_path`` still acts purely
        # as the on/off switch so existing callers / env vars keep working; the
        # file itself is no longer read.
        self._legend_enabled = bool(self.legend_image_path)

        # Per-stage wall-clock for the last render_widget call, in ms.
        # Populated only when timing_enabled is True.
        self.timing_enabled = False
        self.last_timings: dict = {}

        self._initialize_renderer()
    
    def _initialize_renderer(self):
        """Initialize the Generic3DRenderer and load assets."""
        try:
            # Initialize renderer
            self.renderer = Generic3DRenderer(
                width=self.window_size[0],
                height=self.window_size[1],
                background_color=self.background_color,
                ambient_light=np.array([0.3, 0.3, 0.3])  # Lower ambient for better contrast
            )
            
            if not self.renderer.initialize(camera_fov=self.camera_fov):
                raise RuntimeError("Failed to initialize Generic3DRenderer")

            # Load probe mesh
            self._load_probe_mesh()

            # Create coordinate axes
            self._create_coordinate_axes()

            # Create materials
            self._create_materials()

            # Add meshes to scene
            self._setup_scene()

            # Build guidance arrows (depends on probe extents and pivot).
            self._setup_guidance_arrows()

            # Setup camera
            self._setup_camera()

            # Add reference body in a screen corner (after camera so we
            # can use the view basis to place it).
            self._setup_reference_body()
            
        except Exception as e:
            if self.renderer:
                self.renderer.destroy()
            raise RuntimeError(f"Failed to initialize 3D renderer: {e}")
    
    def _load_probe_mesh(self):
        """Load the probe mesh (STL or OBJ) and convert it to meters.

        OBJ files with multiple ``usemtl`` groups are loaded as a
        ``trimesh.Scene`` and split into one pyrender mesh per material so
        the per-part colors declared in the companion ``.mtl`` file render
        through. Single-mesh formats (STL, single-material OBJ) flow
        through the same code path with a list of length one. ``self.probe_mesh``
        remains a combined ``Trimesh`` so the existing extents- and pivot-
        based logic keeps working unchanged.
        """
        import pyrender

        fallback_radius_m = 0.022
        fallback_height_m = 0.15

        sub_meshes = self._load_probe_submeshes()
        if not sub_meshes:
            print(f"Probe mesh not loaded: {self.probe_path}, using capsule")
            capsule = trimesh.creation.capsule(
                radius=fallback_radius_m,
                height=fallback_height_m,
            )
            self.probe_mesh = capsule
            self.probe_node_specs = [("probe", capsule, None)]
        else:
            # Recenter and scale every sub-mesh together so the assembly
            # stays aligned and lands at the AABB center in meters.
            all_vertices = np.concatenate(
                [np.asarray(m.vertices, dtype=float) for m in sub_meshes],
                axis=0,
            )
            center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2.0
            meters_per_unit = _resolve_meters_per_unit(self.probe_units)

            for m in sub_meshes:
                m.vertices = m.vertices - center
                if meters_per_unit != 1.0:
                    m.apply_scale(meters_per_unit)

            # Normalize the probe to a canonical long-axis length so the
            # widget geometry (axes lengths, arrow arc radii, camera
            # auto-fit) stays identical when swapping probe meshes.
            # Without this, ``tail_norm`` and ``probe_extent`` shift with
            # every asset, propagating into arrow sizes and the camera
            # distance.
            if self.probe_long_axis_length_m is not None:
                axis_idx = {'x': 0, 'y': 1, 'z': 2}[self.probe_pivot_axis]
                all_v = np.concatenate(
                    [np.asarray(m.vertices, dtype=float) for m in sub_meshes],
                    axis=0,
                )
                current_long = float(
                    all_v[:, axis_idx].max() - all_v[:, axis_idx].min()
                )
                if current_long > 1e-9:
                    norm_scale = self.probe_long_axis_length_m / current_long
                    if abs(norm_scale - 1.0) > 1e-6:
                        for m in sub_meshes:
                            m.apply_scale(norm_scale)
                        print(
                            f"📏 Normalized probe long-axis "
                            f"({self.probe_pivot_axis}) to "
                            f"{self.probe_long_axis_length_m:.3f} m "
                            f"(scale ×{norm_scale:.4f})"
                        )

            self.probe_node_specs = [
                (f"probe.{i}", m, pyrender.Mesh.from_trimesh(m))
                for i, m in enumerate(sub_meshes)
            ]

            self.probe_mesh = trimesh.util.concatenate(sub_meshes)
            diag = float(np.linalg.norm(self.probe_mesh.extents))
            if not (0.02 <= diag <= 0.40):
                print(
                    f"⚠️  Probe mesh bbox diagonal {diag:.3f} m is outside "
                    f"[0.02, 0.40] m — asset may be mislabeled "
                    f"(probe_units={self.probe_units})."
                )

        if self.probe_pivot_local_m is None:
            if (self.probe_mesh is not None
                    and hasattr(self.probe_mesh, 'extents')
                    and hasattr(self.probe_mesh, 'vertices')):
                axis_idx = {'x': 0, 'y': 1, 'z': 2}[self.probe_pivot_axis]
                verts = np.asarray(self.probe_mesh.vertices, dtype=float)
                axis_coords = verts[:, axis_idx]
                if self.probe_pivot_sign < 0:
                    extreme = float(axis_coords.min())
                    tol = 0.02 * float(self.probe_mesh.extents[axis_idx])
                    mask = axis_coords <= extreme + tol
                else:
                    extreme = float(axis_coords.max())
                    tol = 0.02 * float(self.probe_mesh.extents[axis_idx])
                    mask = axis_coords >= extreme - tol
                # Centroid of the tip face — robust to side protrusions
                # (e.g. battery pack) that shift the AABB center off the
                # probe's centerline.
                self.probe_pivot_local_m = verts[mask].mean(axis=0)
                print(
                    f"probe_pivot_local_m auto-resolved to "
                    f"{self.probe_pivot_local_m} "
                    f"(centroid of {int(mask.sum())} vertices near "
                    f"{self.probe_pivot_axis}"
                    f"{'+' if self.probe_pivot_sign > 0 else '-'})"
                )
            else:
                self.probe_pivot_local_m = np.zeros(3, dtype=float)

    def _load_probe_submeshes(self) -> list:
        """Return a list of Trimesh sub-meshes (one per material), or [].

        Loads the probe asset preserving per-material structure for OBJ
        files; single-mesh formats produce a single-element list.
        """
        if not os.path.exists(self.probe_path):
            return []

        print(f"Loading probe mesh: {self.probe_path}")
        try:
            loaded = trimesh.load(self.probe_path)
        except Exception as e:
            print(f"Failed to load probe mesh: {e}")
            return []

        if isinstance(loaded, trimesh.Scene):
            # ``dump`` bakes scene-graph transforms onto each Trimesh and
            # preserves ``visual.material`` from the .mtl, which pyrender
            # converts to MetallicRoughnessMaterial in Mesh.from_trimesh.
            sub_meshes = list(loaded.dump(concatenate=False))
        else:
            sub_meshes = [loaded]

        return [
            m for m in sub_meshes
            if hasattr(m, 'vertices') and len(m.vertices) > 0
        ]

    def _create_coordinate_axes(self):
        """Create coordinate axes cylinders (sizes in meters)."""
        self.axes = self._create_axes_cylinders(
            length=self.axes_length_m,
            y_length=self.y_axis_length_m,
            radius=self.axes_radius_m,
            x_length=self.x_axis_length_m,
            x_from_tail=self.x_axis_from_tail,
        )
        # Create predicted axes (semi-transparent version)
        self.predicted_axes = self._create_axes_cylinders(
            length=self.axes_length_m,
            y_length=self.y_axis_length_m,
            radius=self.axes_radius_m,
            x_length=self.x_axis_length_m,
            x_from_tail=self.x_axis_from_tail,
        )

    def _create_axes_cylinders(self, length: float = 1.0,
                               y_length: Optional[float] = None,
                               radius: float = 0.02,
                               x_length: Optional[float] = None,
                               x_from_tail: bool = False) -> dict:
        """Create coordinate axes as trimesh cylinder objects."""
        axes = {}
        y_len = float(y_length) if y_length is not None else length
        x_len = float(x_length) if x_length is not None else length

        # X-axis (red) - the probe long axis. By default it extends from the
        # origin along +X (toward the imaging tip). With ``x_from_tail`` it starts
        # at the probe tail (-X end) and extends further out along -X, so the shaft
        # stretches clear of the probe body instead of being occluded by it.
        x_shaft = trimesh.creation.cylinder(radius=radius, height=x_len)
        x_shaft.apply_transform(create_rotation_matrix('y', np.pi/2))
        if x_from_tail:
            tail = float(self.probe_long_axis_length_m or 0.0)
            x_shaft.apply_transform(
                create_translation_matrix(-(tail + x_len/2), 0, 0))
        else:
            x_shaft.apply_transform(create_translation_matrix(x_len/2, 0, 0))
        axes['x'] = x_shaft

        # Y-axis (green) - cylinder extending from origin along +Y. Often
        # rendered longer than X/Z so it pokes out past the probe body.
        y_shaft = trimesh.creation.cylinder(radius=radius, height=y_len)
        y_shaft.apply_transform(create_rotation_matrix('x', -np.pi/2))
        y_shaft.apply_transform(create_translation_matrix(0, y_len/2, 0))
        axes['y'] = y_shaft

        # Z-axis (blue) - cylinder extending from origin along +Z
        z_shaft = trimesh.creation.cylinder(radius=radius, height=length)
        z_shaft.apply_transform(create_translation_matrix(0, 0, length/2))
        axes['z'] = z_shaft

        return axes
    
    def _create_materials(self):
        """Create materials for axes and the capsule fallback probe."""
        import pyrender

        self.materials = {
            # Used only when the probe asset can't be loaded and we fall
            # back to a generated capsule. Real probe meshes carry their
            # own per-submesh materials from the companion .mtl.
            'probe_fallback': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.98, 0.98, 0.98, 1.0],
                metallicFactor=0.5,
                roughnessFactor=0.5
            ),
            # Solid current-pose axes (colours from the shared _AXIS_RGB).
            'x': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*_AXIS_RGB['x'], 1.0],
                metallicFactor=0.3,
                roughnessFactor=0.6
            ),
            'y': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*_AXIS_RGB['y'], 1.0],
                metallicFactor=0.3,
                roughnessFactor=0.6
            ),
            'z': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*_AXIS_RGB['z'], 1.0],
                metallicFactor=0.3,
                roughnessFactor=0.6
            ),
            # Semi-transparent versions for predicted axes (same hues, low alpha).
            'x_predicted': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*_AXIS_RGB['x'], _PREDICTED_AXIS_ALPHA],
                metallicFactor=0.3,
                roughnessFactor=0.6,
                alphaMode='BLEND'
            ),
            'y_predicted': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*_AXIS_RGB['y'], _PREDICTED_AXIS_ALPHA],
                metallicFactor=0.3,
                roughnessFactor=0.6,
                alphaMode='BLEND'
            ),
            'z_predicted': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*_AXIS_RGB['z'], _PREDICTED_AXIS_ALPHA],
                metallicFactor=0.3,
                roughnessFactor=0.6,
                alphaMode='BLEND'
            ),
            # Guidance arrows: low-roughness + slight metallic so each
            # surface picks up its own specular highlight, helping the
            # eye separate overlapping arrows that share the same base
            # color.
            'arrow_translation': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 1.0, 0.0, 1.0],
                metallicFactor=0.25,
                roughnessFactor=0.35,
            ),
            'arrow_rotation': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 1.0, 0.0, 1.0],
                metallicFactor=0.25,
                roughnessFactor=0.35,
            ),
            # Single semi-transparent material for the ghost target probe.
            # Per-submesh .mtl colors are intentionally dropped — a uniform
            # tint reads more clearly as "ghost / target" than a partly
            # see-through replica of the real probe.
            'target_probe': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*self.ghost_probe_color, self.ghost_probe_alpha],
                metallicFactor=0.1,
                roughnessFactor=0.6,
                alphaMode='BLEND',
            ),
        }
    
    def _setup_scene(self):
        """Add meshes to the scene."""
        import pyrender

        # Add probe sub-nodes, one per material from the .mtl. Each entry
        # in ``probe_node_specs`` is (node_name, trimesh, pyrender_mesh).
        # ``pyrender_mesh`` is None for the capsule fallback path so we
        # apply the fallback material there instead.
        for name, sub_tri, pr_mesh in self.probe_node_specs:
            if pr_mesh is None:
                self.renderer.add_node(
                    name, sub_tri,
                    material=self.materials['probe_fallback'],
                )
            else:
                self.renderer.add_node(name, pr_mesh)

        # Ghost target probe: same geometry as the live probe, but with a
        # uniform semi-transparent material so it reads as a "where you
        # should be" overlay. Hidden until ``target_probe`` is toggled on
        # and a predicted rotation is available. Names mirror probe
        # submeshes with a ``target_`` prefix so we can iterate them in
        # lockstep.
        self.target_probe_node_names: list = []
        for name, sub_tri, _pr_mesh in self.probe_node_specs:
            target_name = f"target_{name}"
            self.renderer.add_node(
                target_name,
                pyrender.Mesh.from_trimesh(
                    sub_tri, material=self.materials['target_probe']
                ),
                pose=_HIDDEN_POSE,
            )
            self.target_probe_node_names.append(target_name)

        self.renderer.add_node("x_axis", pyrender.Mesh.from_trimesh(self.axes['x'], material=self.materials['x']))
        self.renderer.add_node("y_axis", pyrender.Mesh.from_trimesh(self.axes['y'], material=self.materials['y']))
        self.renderer.add_node("z_axis", pyrender.Mesh.from_trimesh(self.axes['z'], material=self.materials['z']))

        # Add predicted axes (semi-transparent)
        self.renderer.add_node("predicted_x_axis", pyrender.Mesh.from_trimesh(self.axes['x'], material=self.materials['x_predicted']))
        self.renderer.add_node("predicted_y_axis", pyrender.Mesh.from_trimesh(self.axes['y'], material=self.materials['y_predicted']))
        self.renderer.add_node("predicted_z_axis", pyrender.Mesh.from_trimesh(self.axes['z'], material=self.materials['z_predicted']))

    def _setup_guidance_arrows(self):
        """Create the 4 translation + 4 rotation guidance arrow nodes.

        All arrows are built once with their geometry baked into the IMU
        body frame; per-frame rendering only updates each node's pose to
        ``[R3 | probe_origin_m]`` (or hides it). Translation arrows sit
        outside the tip pointing outward; rotation arrows orbit the tail
        with the curl plane perpendicular to their axis.
        """
        import pyrender

        # Tail position in IMU body frame. With the probe rigged so the
        # mesh-local pivot lands on the IMU long-axis end (+X by default),
        # the tail end vertex sits at ``-2 * pivot`` in mesh-local, which
        # the axis-fix matrix maps into the IMU body frame.
        if self.probe_pivot_local_m is not None:
            self._tail_imu = self.probe_axis_fix_matrix @ (
                -2.0 * np.asarray(self.probe_pivot_local_m, dtype=float)
            )
        else:
            self._tail_imu = np.zeros(3, dtype=float)

        # Translation arrows: tail at ``offset * dir`` from probe tip,
        # pointing further along ``dir`` for ``length``. Tip is at the
        # IMU body-frame origin. ``flat_normal`` is the IMU long axis
        # (+X), so all four cardinal arrows lie flat in the YZ plane
        # together — matching the look of a flat arrow rosette.
        for direction, vec in _TRANSLATION_DIRS.items():
            arrow_mesh = create_straight_arrow(
                length=self.translation_arrow_length_m,
                shaft_width=self.axes_radius_m * 8.0,
                shaft_thickness=self.axes_radius_m * 0.7,
                head_length=self.translation_arrow_length_m * 0.30,
                head_width=self.axes_radius_m * 20.0,
                direction=tuple(vec),
                flat_normal=(1.0, 0.0, 0.0),
            )
            arrow_mesh.apply_translation(vec * (self.translation_arrow_offset_m * 2.0))
            self.renderer.add_node(
                f"trans_arrow_{direction}",
                pyrender.Mesh.from_trimesh(
                    arrow_mesh, material=self.materials['arrow_translation']
                ),
                pose=_HIDDEN_POSE,
            )

        # Rotation arrows: arcs that bend along the rotation trajectory.
        # The cross-section ('tall' orientation) keeps the ribbon's wide
        # face along the bend axis, matching the straight-arrow look at
        # the tail (wide along the cross axis, thin along the long axis).
        # Bend axis = long_axis × rotation_axis, so the ribbon's tangent
        # at the start equals +rotation_axis — the stem stretches along
        # the rotation axis, then curls toward the long axis.
        #
        # Each tunable below is independent of the others — change one
        # without disturbing the rest. ``arc_angle`` is the only derived
        # quantity (length / radius); fix length and radius and the angle
        # follows.
        tail_norm = float(np.linalg.norm(self._tail_imu))
        long_axis_imu = (
            self._tail_imu / tail_norm if tail_norm > 1e-9
            else np.array([-1.0, 0.0, 0.0])
        )

        # --- Rotation arrow tunables (all independent) ---
        rot_arc_radius_imu = max(tail_norm * 0.7, 1e-3)   # curvature
        rot_arc_length_imu = tail_norm * 0.55             # length along curve
        rot_ribbon_width_imu = self.axes_radius_m * 16.0  # shaft width
        rot_thickness_imu = self.axes_radius_m * 0.7      # out-of-plane depth
        rot_head_length_imu = rot_arc_length_imu * 0.30   # head arc length
        rot_head_width_imu = self.axes_radius_m * 40.0    # head base width
        # Where the arc's geometric center sits in IMU coords. The
        # ribbon's start point is ``arc_center + arc_radius * long_axis``,
        # so this default plants the start on the tail regardless of
        # ``rot_arc_radius_imu``. Edit this line alone to translate the
        # whole arrow along the long axis without changing its shape.
        rot_arc_center_imu = (tail_norm - rot_arc_radius_imu) * long_axis_imu
        # Sideways offset along each direction's rotation axis (applied
        # per-arrow inside the loop so each arrow lands on its own side).
        rot_side_offset_m = 2.0 * self.rotation_arrow_offset_m
        # --- end tunables ---

        rot_arc_angle_deg = float(
            np.rad2deg(rot_arc_length_imu / rot_arc_radius_imu)
        )

        for direction, axis in _ROTATION_AXES.items():
            bend_axis = np.cross(long_axis_imu, axis)
            n = float(np.linalg.norm(bend_axis))
            if n < 1e-9:
                # Long axis parallel to rotation axis (won't happen for
                # ±Y, ±Z rotations with a -X long axis — keep the guard).
                continue
            bend_axis = bend_axis / n

            arrow_mesh = create_curved_arrow(
                arc_radius=rot_arc_radius_imu,
                arc_angle_deg=rot_arc_angle_deg,
                ribbon_width=rot_ribbon_width_imu,
                thickness=rot_thickness_imu,
                head_length=rot_head_length_imu,
                head_width=rot_head_width_imu,
                plane_normal=tuple(bend_axis),
                start_dir=tuple(long_axis_imu),
                orientation='tall',
            )
            arrow_mesh.apply_translation(
                np.asarray(axis, dtype=float) * rot_side_offset_m
                + rot_arc_center_imu
            )
            self.renderer.add_node(
                f"rot_arrow_{direction}",
                pyrender.Mesh.from_trimesh(
                    arrow_mesh, material=self.materials['arrow_rotation']
                ),
                pose=_HIDDEN_POSE,
            )

        # Roll arrows: 270°-arc curls orbiting the long axis at the tail.
        # The arc plane is perpendicular to the long axis, so the ring
        # circles around the probe handle — visualizing the twist motion
        # the sonographer would apply to fix axial roll. ``orientation=
        # 'flat'`` keeps the wide ribbon face co-planar with the YZ-style
        # arc plane, so the curl reads as a flat halo around the handle
        # rather than a tape strip standing on its edge.
        roll_arc_radius_imu = max(tail_norm * 0.315, 1e-3)
        roll_arc_angle_deg = 270.0
        roll_ribbon_width_imu = self.axes_radius_m * 3.5
        roll_thickness_imu = self.axes_radius_m * 0.7
        roll_head_length_imu = roll_arc_radius_imu * 0.40
        roll_head_width_imu = self.axes_radius_m * 14.0
        # Place the arc center 1.5× the tail distance along the long
        # axis, so the curl floats just above the probe handle (further
        # from the tip) without crowding the directional rotation
        # arrows. ``_tail_imu`` already points along the long axis from
        # tip to tail in the IMU body frame. +Y is perpendicular to
        # either ±X plane normal, so a single ``start_dir`` works for
        # both roll directions.
        roll_arc_center_imu = 1.3 * self._tail_imu
        roll_start_dir = np.array([0.0, 1.0, 0.0])

        for direction, axis in _ROLL_AXES.items():
            arrow_mesh = create_curved_arrow(
                arc_radius=roll_arc_radius_imu,
                arc_angle_deg=roll_arc_angle_deg,
                ribbon_width=roll_ribbon_width_imu,
                thickness=roll_thickness_imu,
                head_length=roll_head_length_imu,
                head_width=roll_head_width_imu,
                plane_normal=tuple(axis),
                start_dir=tuple(roll_start_dir),
                orientation='flat',
            )
            arrow_mesh.apply_translation(roll_arc_center_imu)
            self.renderer.add_node(
                f"roll_arrow_{direction}",
                pyrender.Mesh.from_trimesh(
                    arrow_mesh, material=self.materials['arrow_rotation']
                ),
                pose=_HIDDEN_POSE,
            )

    def set_layer_visible(self, name: str, visible: bool) -> None:
        """Toggle a renderer layer on or off in real time.

        Args:
            name: One of ``"current_axes"``, ``"predicted_axes"``,
                ``"guidance_arrows"``, ``"target_probe"``,
                ``"reference_body"``.
            visible: True to show, False to hide on the next frame.
        """
        if name not in self._layers_visible:
            raise ValueError(
                f"Unknown layer '{name}'. Valid: "
                f"{sorted(self._layers_visible.keys())}"
            )
        self._layers_visible[name] = bool(visible)
        # Reference body has a static pose (camera-corner anchored) and is
        # not touched by render_widget, so apply visibility immediately by
        # swapping its node pose. Other layers are honored each frame.
        if (name == "reference_body"
                and self._reference_body_loaded
                and self.renderer is not None):
            self.renderer.set_pose(
                "reference_body",
                self._reference_body_pose if visible else _HIDDEN_POSE,
            )
        # Live-geometry layers are (re)applied on the render thread; mark
        # them dirty so the toggle takes effect on the next frame.
        if name in ("reference_body", "world_pcl", "contact_marker"):
            with self._live_lock:
                self._pc_dirty = True
                self._body_dirty = True
                self._probe_dirty = True

    def is_layer_visible(self, name: str) -> bool:
        return bool(self._layers_visible.get(name, False))

    def toggle_layer(self, name: str) -> bool:
        """Flip a layer's visibility. Returns the new state."""
        self.set_layer_visible(name, not self.is_layer_visible(name))
        return self.is_layer_visible(name)

    def toggle_debug_print(self) -> bool:
        """Flip the debug-print flag. Returns the new state."""
        self.debug_print = not self.debug_print
        print(f"🐛 debug_print = {self.debug_print}")
        return self.debug_print

    # ------------------------------------------------------------------
    # Live patient geometry API (thread-safe; cache only, no scene mutation)
    # ------------------------------------------------------------------
    def set_world_point_cloud(self, points, colors=None) -> None:
        """Cache the latest Stevens VGGT point cloud (world meters).

        Args:
            points: (N, 3) array of point positions in the world/RealSense
                frame, or None to clear the cloud.
            colors: Optional (N, 3)/(N, 4) per-point colours (uint8 0-255 or
                float 0-1).
        Safe to call from the ROS subscription thread.
        """
        with self._live_lock:
            if points is None:
                self._live_point_cloud = None
            else:
                self._live_point_cloud = (
                    np.asarray(points, dtype=np.float32).reshape(-1, 3),
                    None if colors is None else np.asarray(colors),
                )
            self._pc_dirty = True

    def set_reference_body_mesh(self, vertices, faces) -> None:
        """Cache the latest live SMPL body mesh (world meters).

        Pass ``vertices=None`` to drop the live body and fall back to the
        static reference OBJ. Safe to call from the ROS subscription thread.
        """
        with self._live_lock:
            if vertices is None or faces is None:
                self._live_body = None
            else:
                self._live_body = (
                    np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
                    np.asarray(faces),
                )
            self._body_dirty = True

    def set_probe_world_position(self, xyz) -> None:
        """Cache the probe tip position in world meters (or None to clear)."""
        with self._live_lock:
            self._probe_world_pos = (
                None if xyz is None
                else np.asarray(xyz, dtype=float).reshape(3)
            )
            self._probe_dirty = True

    def set_world_to_scene_transform(self, transform) -> None:
        """Override the world->scene 4x4 placement transform.

        When set, ``auto_fit_live_geometry`` is disabled so this transform is
        used verbatim for the point cloud, body mesh, and probe position.
        """
        with self._live_lock:
            self._world_to_scene = (
                None if transform is None
                else np.asarray(transform, dtype=float).reshape(4, 4)
            )
            self.auto_fit_live_geometry = False
            self._pc_dirty = self._body_dirty = self._probe_dirty = True

    # ------------------------------------------------------------------
    # Live patient geometry — render-thread application
    # ------------------------------------------------------------------
    def _compute_world_to_scene(self, vertices: np.ndarray) -> np.ndarray:
        """Auto-fit transform mapping world-frame geometry into the scene.

        Scales the SMPL body to roughly the size the static reference body
        used and translates its center to the same below-the-probe anchor,
        so the live body drops in where the static one was. An optional
        ``_live_geometry_rotation`` is applied for orientation tuning.
        """
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = float(np.max(maxs - mins))
        if extent <= 0.0:
            extent = 1.0
        if self.probe_mesh is not None and hasattr(self.probe_mesh, 'extents'):
            probe_size = float(np.max(self.probe_mesh.extents))
        else:
            probe_size = 0.04
        target_size = probe_size * self.reference_body_scale_ratio * 3.0
        scale = target_size / extent
        anchor = self._compute_center_bottom_pose()[:3, 3]
        sr = scale * self._live_geometry_rotation
        transform = np.eye(4)
        transform[:3, :3] = sr
        transform[:3, 3] = anchor - sr @ center
        return transform

    def _world_to_scene_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the cached world->scene transform to (N,3) points."""
        T = self._world_to_scene
        if T is None:
            return np.asarray(pts, dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        return (T[:3, :3] @ pts.T).T + T[:3, 3]

    def _apply_live_geometry(self) -> None:
        """Push cached live geometry into the scene (render thread only)."""
        if self.renderer is None:
            return
        with self._live_lock:
            pc_dirty, body_dirty, probe_dirty = (
                self._pc_dirty, self._body_dirty, self._probe_dirty
            )
            if not (pc_dirty or body_dirty or probe_dirty):
                return
            point_cloud = self._live_point_cloud
            body = self._live_body
            probe_world = (
                None if self._probe_world_pos is None
                else self._probe_world_pos.copy()
            )
            self._pc_dirty = self._body_dirty = self._probe_dirty = False

        # --- SMPL body mesh ---
        if body_dirty:
            if body is not None:
                verts, faces = body
                # Compute the placement transform once (from the first body)
                # and freeze it, so later poses show the patient *moving*
                # within a stable frame instead of rescaling every frame.
                if self._world_to_scene is None and self.auto_fit_live_geometry:
                    self._world_to_scene = self._compute_world_to_scene(verts)
                verts_scene = self._world_to_scene_points(verts)
                self._live_body_scene_mesh = trimesh.Trimesh(
                    vertices=verts_scene, faces=faces, process=False
                )
                show_body = self._layers_visible["reference_body"]
                self.renderer.update_mesh(
                    "live_body", verts_scene, faces,
                    material=self._skin_material(),
                    pose=(np.eye(4) if show_body else _HIDDEN_POSE),
                )
                # Hide the static OBJ while the live body is shown.
                if self._reference_body_loaded:
                    self.renderer.set_pose("reference_body", _HIDDEN_POSE)
                self._live_body_active = True
            else:
                # No live body — drop it and restore the static reference.
                if self._live_body_active:
                    self.renderer.remove_node("live_body", quiet=True)
                    self._live_body_scene_mesh = None
                    self._live_body_active = False
                self._apply_reference_body_pose()

        # --- world point cloud ---
        if pc_dirty:
            show_pc = self._layers_visible["world_pcl"]
            if point_cloud is not None and show_pc:
                pts, cols = point_cloud
                self.renderer.set_point_cloud(
                    "world_pcl", self._world_to_scene_points(pts), colors=cols,
                )
            else:
                self.renderer.remove_node("world_pcl", quiet=True)

        # --- contact marker (probe tip -> closest point on body) ---
        if body_dirty or probe_dirty or pc_dirty:
            self._update_contact_marker(probe_world)

    def _skin_material(self):
        import pyrender
        return pyrender.MetallicRoughnessMaterial(
            baseColorFactor=list(self.body_mesh_color),
            metallicFactor=self.body_mesh_metallic,
            roughnessFactor=self.body_mesh_roughness,
        )

    def _update_contact_marker(self, probe_world) -> None:
        """Place a marker at the body-surface point nearest the probe tip."""
        show = self._layers_visible["contact_marker"]
        if (not show or probe_world is None
                or self._live_body_scene_mesh is None):
            if self._contact_marker_loaded:
                self.renderer.set_pose("contact_marker", _HIDDEN_POSE)
            return

        probe_scene = self._world_to_scene_points(
            probe_world.reshape(1, 3)
        )[0]
        try:
            from trimesh.proximity import closest_point
            pts, _dist, _tri = closest_point(
                self._live_body_scene_mesh, probe_scene.reshape(1, 3)
            )
            contact_pt = pts[0]
        except Exception as e:
            if self.debug_print:
                print(f"contact closest-point failed: {e}")
            return

        if not self._contact_marker_loaded:
            import pyrender
            sphere = trimesh.creation.icosphere(
                subdivisions=2, radius=self.contact_marker_radius_m
            )
            marker_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=list(self.contact_marker_color),
                metallicFactor=0.0,
                roughnessFactor=0.6,
            )
            self.renderer.add_node(
                "contact_marker", sphere, pose=_HIDDEN_POSE,
                material=marker_material, quiet=True,
            )
            self._contact_marker_loaded = True

        pose = np.eye(4)
        pose[:3, 3] = contact_pt
        self.renderer.set_pose("contact_marker", pose)

        # Optionally move the probe mesh to the body (position only; the IMU
        # still drives orientation). Off by default until the IMU<->world
        # orientation calibration is dialed in.
        if self.place_probe_at_world_pos:
            self.probe_origin_m = probe_scene.astype(float)

    def set_active_translation_arrow(self, direction: str) -> None:
        """Pick which translation guidance arrow renders.

        Lights ``direction`` and clears any other translation arrow.

        Args:
            direction: One of ``"left"``, ``"right"``, ``"up"``, ``"down"``.
        """
        if direction not in _TRANSLATION_DIRS:
            raise ValueError(
                f"Unknown translation direction '{direction}'. Valid: "
                f"{sorted(_TRANSLATION_DIRS.keys())}"
            )
        for key in self._translation_arrow_visible:
            self._translation_arrow_visible[key] = (key == direction)

    def set_active_rotation_arrow(self, direction: str) -> None:
        """Pick which rotation guidance arrow renders.

        Lights ``direction`` and clears any other rotation arrow. Use
        :meth:`update_rotation_arrows_from_prediction` instead when the
        arrows should be driven by a model output rather than by manual
        selection — that path can light multiple arrows simultaneously.

        Args:
            direction: One of ``"left"``, ``"right"``, ``"up"``, ``"down"``.
        """
        if direction not in _ROTATION_AXES:
            raise ValueError(
                f"Unknown rotation direction '{direction}'. Valid: "
                f"{sorted(_ROTATION_AXES.keys())}"
            )
        for key in self._rotation_arrow_visible:
            self._rotation_arrow_visible[key] = (key == direction)

    def clear_rotation_arrows(self) -> None:
        """Hide every rotation arrow."""
        for key in self._rotation_arrow_visible:
            self._rotation_arrow_visible[key] = False

    def clear_translation_arrows(self) -> None:
        """Hide every translation arrow."""
        for key in self._translation_arrow_visible:
            self._translation_arrow_visible[key] = False

    def clear_roll_arrows(self) -> None:
        """Hide every roll (long-axis curl) arrow."""
        for key in self._roll_arrow_visible:
            self._roll_arrow_visible[key] = False

    def set_active_roll_arrow(self, direction: str) -> None:
        """Pick which roll guidance arrow renders.

        Lights ``direction`` and clears the other roll arrow.

        Args:
            direction: One of ``"cw"``, ``"ccw"`` (clockwise / counter-
                clockwise as viewed from the probe tail looking toward
                the tip — i.e. along +X in the IMU body frame).
        """
        if direction not in _ROLL_AXES:
            raise ValueError(
                f"Unknown roll direction '{direction}'. Valid: "
                f"{sorted(_ROLL_AXES.keys())}"
            )
        for key in self._roll_arrow_visible:
            self._roll_arrow_visible[key] = (key == direction)

    def update_rotation_arrows_from_prediction(
        self,
        rotation_prediction: Optional[np.ndarray],
        threshold_deg: Optional[float] = None,
    ) -> None:
        """Drive rotation + roll arrow visibility from a predicted rotation.

        Decomposes the body-frame correction (the inverse of
        ``rotation_prediction``, matching the convention used by
        :meth:`amplify_rotation_prediction` where the target pose is
        ``current @ inv(R_pred)``) into ``xyz`` Euler angles and lights
        the corresponding arrow for each component whose magnitude meets
        ``threshold_deg``. Multiple arrows can be lit simultaneously
        (e.g. "rotate up AND right AND roll CCW").

        Axis → arrow mapping (matches each arrow's bend axis, i.e. the
        axis its ribbon actually curls around — see ``_ROTATION_AXES``
        and the ``bend_axis = cross(long_axis_imu, axis)`` setup):
            rx (long-axis roll) → ``roll_arrow_visible``  ("cw"/"ccw")
            ry (around green Y) → ``rotation_arrow_visible`` ("right"/"left")
            rz (around blue Z)  → ``rotation_arrow_visible`` ("down"/"up")

        Args:
            rotation_prediction: 3x3 (or 4x4) predicted rotation matrix.
                Pass ``None`` to clear all arrows.
            threshold_deg: Override for ``rotation_arrow_threshold_deg``.
        """
        self.clear_rotation_arrows()
        self.clear_roll_arrows()
        if rotation_prediction is None:
            return

        thresh = (
            self.rotation_arrow_threshold_deg
            if threshold_deg is None else float(threshold_deg)
        )

        R_pred = (
            rotation_prediction[:3, :3]
            if rotation_prediction.shape == (4, 4) else rotation_prediction
        )
        R_correction = R_pred.T
        rx, ry, rz = R.from_matrix(R_correction).as_euler('xyz', degrees=True)

        if abs(rx) >= thresh:
            self._roll_arrow_visible["ccw" if rx > 0 else "cw"] = True
        if abs(ry) >= thresh:
            self._rotation_arrow_visible["right" if ry > 0 else "left"] = True
        if abs(rz) >= thresh:
            self._rotation_arrow_visible["down" if rz > 0 else "up"] = True

        if self.debug_print:
            # trans-1/trans-2 are placeholders until translation prediction
            # is implemented; rot-1=ry (right/left, curls around +Y),
            # rot-2=rz (down/up, curls around +Z), roll=rx (long-axis
            # twist). All in degrees.
            print(
                f"[delta] rot-1={ry:+6.1f}° rot-2={rz:+6.1f}° "
                f"trans-1=    -- trans-2=    -- roll={rx:+6.1f}°"
            )

    def translation_prediction_to_body(self, translation_prediction: np.ndarray) -> np.ndarray:
        """Map a USGR-frame T_rel (metres) into this renderer's IMU body frame.

        Applies the ``usgr_to_body`` axis sign/permutation then the visual
        ``translation_amplification`` gain. Returns a length-3 array
        ``[t_x, t_y, t_z]`` in body-frame metres.
        """
        t = np.asarray(translation_prediction, dtype=float).reshape(3)
        return (self.usgr_to_body @ t) * self.translation_amplification

    def update_translation_arrows_from_prediction(
        self,
        translation_prediction: Optional[np.ndarray],
        threshold_m: Optional[float] = None,
    ) -> None:
        """Drive translation arrow visibility from a predicted T_rel.

        ``translation_prediction`` is the body-centric translation (metres,
        USGR probe frame) from the current pose to the target standard view.
        It is mapped into the renderer body frame (:meth:`translation_prediction_to_body`)
        and decomposed onto the perpendicular-plane axes used by the
        translation arrows (``_TRANSLATION_DIRS``): the Y component lights
        ``right``/``left`` and the Z component lights ``up``/``down``. Both can
        light at once (e.g. "move right AND up"). The long-axis (X / in-out)
        component has no dedicated arrow and is ignored here.

        Args:
            translation_prediction: length-3 translation, or ``None`` to clear.
            threshold_m: override for ``translation_arrow_threshold_m``.
        """
        self.clear_translation_arrows()
        if translation_prediction is None:
            return

        thresh = (
            self.translation_arrow_threshold_m
            if threshold_m is None else float(threshold_m)
        )
        t_body = self.translation_prediction_to_body(translation_prediction)
        ty, tz = float(t_body[1]), float(t_body[2])

        if abs(ty) >= thresh:
            self._translation_arrow_visible["right" if ty > 0 else "left"] = True
        if abs(tz) >= thresh:
            self._translation_arrow_visible["up" if tz > 0 else "down"] = True

        if self.debug_print:
            print(
                f"[delta] trans-y={ty:+.4f}m trans-z={tz:+.4f}m "
                f"(thresh={thresh:.4f}m)"
            )

    def _alignment_color(self, angle_rad: float) -> list:
        """Map a geodesic rotation angle to an RGBA arrow color.

        Returns green at 0 rad and red at ``alignment_color_max_deg``,
        linearly interpolated in RGB. Alpha is fixed at 1.0.
        """
        max_rad = max(np.deg2rad(self.alignment_color_max_deg), 1e-6)
        t = float(np.clip(angle_rad / max_rad, 0.0, 1.0))
        green = np.array([0.0, 1.0, 0.0])
        red = np.array([1.0, 0.0, 0.0])
        rgb = (1.0 - t) * green + t * red
        return [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]

    def _update_arrow_alignment_color(
        self, rotation_prediction: Optional[np.ndarray]
    ) -> None:
        """Recolor every guidance arrow based on the predicted misalignment.

        The geodesic distance from identity (``arccos((tr(R)-1)/2)``) is
        the natural angular distance between the current and predicted
        axes. ``pyrender.Mesh.from_trimesh`` deep-copies the supplied
        material onto each primitive, so the per-primitive material is
        the only one the shader sees — mutating the dict templates does
        nothing on screen.
        """
        if rotation_prediction is None or self.renderer is None:
            return
        R3 = (
            rotation_prediction[:3, :3]
            if rotation_prediction.shape == (4, 4) else rotation_prediction
        )
        cos_angle = (float(np.trace(R3)) - 1.0) / 2.0
        angle_rad = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        color = self._alignment_color(angle_rad)
        arrow_node_names = (
            [f"trans_arrow_{d}" for d in _TRANSLATION_DIRS]
            + [f"rot_arrow_{d}" for d in _ROTATION_AXES]
            + [f"roll_arrow_{d}" for d in _ROLL_AXES]
        )
        for name in arrow_node_names:
            mesh = self.renderer.meshes.get(name)
            if mesh is None:
                continue
            for primitive in mesh.primitives:
                primitive.material.baseColorFactor = color
        if self.debug_print:
            max_rad = max(np.deg2rad(self.alignment_color_max_deg), 1e-6)
            print(
                f"[color] geodesic={np.rad2deg(angle_rad):5.1f}° "
                f"max={self.alignment_color_max_deg:.1f}° "
                f"t={angle_rad / max_rad:.2f} "
                f"rgba=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f},{color[3]:.2f})"
            )

    def _compute_scene_diag_m(self) -> float:
        """Return the scene AABB diagonal length in meters.

        Covers the probe mesh plus the coordinate axes. Falls back to a
        probe-only estimate if extents aren't available.
        """
        axes_extent = max(self.axes_length_m, self.y_axis_length_m)
        probe_reach = 0.0
        if self.probe_mesh is not None and hasattr(self.probe_mesh, 'extents'):
            probe_extent = float(np.max(self.probe_mesh.extents))
            pivot_offset = (
                float(np.linalg.norm(self.probe_pivot_local_m))
                if self.probe_pivot_local_m is not None else 0.0
            )
            probe_reach = probe_extent / 2.0 + pivot_offset
        bbox_side = max(axes_extent, probe_reach)
        return float(np.sqrt(3.0) * bbox_side)

    def _setup_camera(self):
        """Setup camera position and orientation (all distances in meters)."""
        if self.camera_distance_m is not None:
            distance = float(self.camera_distance_m)
        else:
            distance = max(0.35, 2.0 * self._compute_scene_diag_m())
        zoom = self.camera_zoom if self.camera_zoom > 1e-6 else 1.0
        distance = distance / zoom

        # Place the camera on a sphere of radius `distance` around the origin,
        # parameterized by azimuth (around +Y) and elevation (above XZ plane).
        az = np.deg2rad(self.camera_azimuth_deg)
        el = np.deg2rad(self.camera_elevation_deg)
        camera_position = np.array([
            distance * np.cos(el) * np.sin(az),
            distance * np.sin(el),
            distance * np.cos(el) * np.cos(az),
        ])

        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        forward = target - camera_position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up_corrected = np.cross(right, forward)

        # Translate the camera along its own up-axis so the world origin
        # renders at ``probe_origin_screen_y_from_top`` (fraction from top).
        # Pure translation — keeps ``forward``/``right``/``up_corrected``
        # intact, so the look direction is unchanged.
        half_h = distance * np.tan(self.camera_fov / 2.0)
        y_from_top = self.probe_origin_screen_y_from_top
        camera_position = camera_position + up_corrected * (
            (2.0 * y_from_top - 1.0) * half_h
        )

        self.camera_pose = np.eye(4)
        self.camera_pose[:3, 0] = right
        self.camera_pose[:3, 1] = up_corrected
        self.camera_pose[:3, 2] = -forward
        self.camera_pose[:3, 3] = camera_position
        self.camera_pos = camera_position

        self.renderer.set_camera_pose(self.camera_pose)

        # Update lighting direction to point from camera to object
        self._setup_camera_aligned_lighting()

    def refit_camera(self):
        """Recompute camera framing from the current scene AABB."""
        self._setup_camera()

    def _refresh_camera(self) -> None:
        """Re-apply the camera pose and re-anchor camera-relative props.

        ``_setup_camera`` rebuilds ``camera_pose`` from the current
        azimuth/elevation/distance and re-aims the lights. The reference
        body's screen-corner placement is camera-relative, so its pose
        has to be recomputed against the new view.
        """
        self._setup_camera()
        self._apply_reference_body_pose()

    def _capture_body_world_orientation(self) -> np.ndarray:
        """Build a 3x3 body-axes-in-world matrix from the current view.

        Body-local axes: +Y head, +Z front (face), +X body-left.
        The body lies on its back, so head points along the projected
        camera-right (``_world_head_direction``) and face points world up.
        ``+X = +Y × +Z`` keeps the basis right-handed.
        """
        head_dir = self._world_head_direction()
        world_up = np.array([0.0, 1.0, 0.0])
        body_y = head_dir
        body_z = world_up
        body_x = np.cross(body_y, body_z)
        n = float(np.linalg.norm(body_x))
        if n > 1e-9:
            body_x = body_x / n
        return np.column_stack([body_x, body_y, body_z])

    def _apply_reference_body_pose(self) -> None:
        """Recompute and push the reference body's pose to the renderer.

        Called whenever an input that feeds ``_compute_center_bottom_pose``
        changes — camera nudges, calibration, or calibration reset. Skips
        when the body is hidden so the hidden-pose stays in place; the
        new pose still lives in ``_reference_body_pose`` and gets applied
        the next time the layer is turned on.
        """
        if not self._reference_body_loaded or self.renderer is None:
            return
        self._reference_body_pose = self._compute_center_bottom_pose()
        if self._layers_visible["reference_body"]:
            self.renderer.set_pose(
                "reference_body", self._reference_body_pose
            )

    def nudge_camera_azimuth(self, delta_deg: float) -> None:
        """Rotate the camera around +Y by ``delta_deg``."""
        self.camera_azimuth_deg = float(self.camera_azimuth_deg + delta_deg)
        self._refresh_camera()
        print(
            f"📷 camera azimuth={self.camera_azimuth_deg:+.1f}° "
            f"elevation={self.camera_elevation_deg:+.1f}°"
        )

    def nudge_camera_elevation(self, delta_deg: float) -> None:
        """Pitch the camera by ``delta_deg``, clamped to avoid gimbal flip."""
        self.camera_elevation_deg = float(np.clip(
            self.camera_elevation_deg + delta_deg, -89.0, 89.0
        ))
        self._refresh_camera()
        print(
            f"📷 camera azimuth={self.camera_azimuth_deg:+.1f}° "
            f"elevation={self.camera_elevation_deg:+.1f}°"
        )

    def nudge_camera_zoom(self, factor: float) -> None:
        """Multiply the camera zoom by ``factor``.

        ``factor > 1`` moves the camera closer (zoom in); ``< 1`` farther
        (zoom out). Clamped to a sane range so a stuck-key combo can't
        push the camera through the origin or off into infinity.
        """
        if factor <= 0:
            return
        self.camera_zoom = float(np.clip(
            self.camera_zoom * float(factor), 0.1, 10.0
        ))
        self._refresh_camera()
        print(f"🔍 camera zoom={self.camera_zoom:.2f}x")

    def reset_camera(self) -> None:
        """Restore the default azimuth/elevation/zoom set at construction."""
        self.camera_azimuth_deg = self._default_camera_azimuth_deg
        self.camera_elevation_deg = self._default_camera_elevation_deg
        self.camera_zoom = self._default_camera_zoom
        self._refresh_camera()
        print(
            f"📷 camera reset (azimuth={self.camera_azimuth_deg:+.1f}° "
            f"elevation={self.camera_elevation_deg:+.1f}° "
            f"zoom={self.camera_zoom:.2f}x)"
        )

    def _world_head_direction(self) -> np.ndarray:
        """Unit vector in the world XZ plane that the body reference's head
        (and the probe's tip after calibration) points toward.

        Picked as the camera-right direction projected onto the ground
        plane. That keeps the reference scene laid flat on Y=0 — so when
        the camera elevates, both the body and the calibrated probe show
        their top faces — while still matching screen-right at the current
        camera azimuth so the probe and body line up visually.
        """
        if self.camera_pose is None:
            return np.array([1.0, 0.0, 0.0])
        right_world = self.camera_pose[:3, 0]
        world_up = np.array([0.0, 1.0, 0.0])
        head_dir = right_world - world_up * float(np.dot(right_world, world_up))
        n = float(np.linalg.norm(head_dir))
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return head_dir / n

    def _compute_calibration_target_world(self) -> np.ndarray:
        """3x3 R_target that places the probe in the calibration view.

        Maps the IMU body frame to world axes — not camera axes — so the
        calibrated probe lies flat on the world ground plane. The Clarius
        HD3 IMU body frame (per clariusdev/imu axes-v3.png) is +X along
        the long axis toward the imaging tip, +Y the cross axis, +Z out
        the back of the probe (away from the imaging face). At
        calibration the operator holds the probe with tip pointing
        superior (screen-right per the body reference) and button up, so
        we send IMU +X → +head_dir (tip at screen-right) and IMU +Z →
        world +Y (button-up). IMU +Y is the cross product (Z × X) to
        keep the basis right-handed.
        """
        if self.calibration_head_dir_world is not None:
            # Live face-up body: target the patient's fixed scene axes so the
            # canonical hold (tip toward the head, button up) maps onto the body,
            # not onto a camera-relative direction.
            head_dir = np.asarray(self.calibration_head_dir_world, float)
            head_dir = head_dir / (np.linalg.norm(head_dir) + 1e-9)
            world_up = np.asarray(self.calibration_up_world, float)
            world_up = world_up / (np.linalg.norm(world_up) + 1e-9)
        else:
            head_dir = self._world_head_direction()
            world_up = np.array([0.0, 1.0, 0.0])
        probe_x_world = head_dir
        probe_y_world = np.cross(world_up, head_dir)
        n = float(np.linalg.norm(probe_y_world))
        if n > 1e-9:
            probe_y_world = probe_y_world / n
        return np.column_stack([probe_x_world, probe_y_world, world_up])

    def set_calibration_body_axes(self, head_dir_world, up_world) -> None:
        """Fix the calibration target to the patient's scene axes (or None to
        restore the camera-relative reference-body convention).

        head_dir_world: unit-ish direction the probe tip points at calibration
        (patient superior). up_world: the button-up direction (patient anterior).
        """
        if head_dir_world is None or up_world is None:
            self.calibration_head_dir_world = None
            self.calibration_up_world = None
        else:
            self.calibration_head_dir_world = np.asarray(head_dir_world, float).reshape(3)
            self.calibration_up_world = np.asarray(up_world, float).reshape(3)

    def calibrate(self, qw: float = 1.0, qx: float = 0.0,
                  qy: float = 0.0, qz: float = 0.0):
        """Reset the rendering pose so the probe's current IMU reading maps
        to the canonical calibration view: tip pointing toward the body's
        superior direction (screen-right per the body reference) with the
        button side facing screen-up.

        Call while the operator holds the probe parallel to the patient's
        long axis, button up, tip pointing superior.
        """
        R_imu = quaternion_to_matrix(qw, qx, qy, qz)[:3, :3]
        R_target = self._compute_calibration_target_world()
        # Left-multiply form: R3 = calibration_matrix @ R_imu_now.
        # Derivation: with R_imu_now = R_imu_calib @ dR_body (rotation in
        # the probe's body frame), we want the rendered probe to have
        # R_target @ dR_body — i.e., the same body-frame rotation applied
        # to the calibration pose. That gives R3 = R_target @ R_imu_calib.T
        # @ R_imu_now, regardless of how IMU-world axes align with render-
        # world axes. The previous right-multiply form leaked the IMU-
        # world basis into render rotations, swapping/flipping axes when
        # the two world frames disagreed (e.g., IMU +Z up vs render +Y up).
        self.calibration_matrix = R_target @ R_imu.T

        # Re-freeze the reference body's orientation against the world axes
        # implied by the calibration view (head along the projected
        # camera-right, face up along world +Y). The body had a
        # construction-time freeze; this overwrites it with the
        # operator's chosen calibration view.
        self._body_orientation_world = self._capture_body_world_orientation()
        self._apply_reference_body_pose()

        print(
            f"✅ Probe calibrated (q=[{qw:.3f}, {qx:.3f}, "
            f"{qy:.3f}, {qz:.3f}])."
        )

    def reset_calibration(self):
        """Drop any prior calibration; render raw IMU rotation again."""
        self.calibration_matrix = np.eye(3)
        if self._default_body_orientation_world is not None:
            self._body_orientation_world = (
                self._default_body_orientation_world.copy()
            )
            self._apply_reference_body_pose()
        print("↩️  Probe calibration reset.")

    def _setup_reference_body(self):
        """Load the reference body mesh and pin it to a screen corner.

        The body is rendered as a static reference (does not rotate with
        the IMU) and is sized as a fraction of the probe's largest extent
        so it stays small relative to the transducer.
        """
        if not self.reference_body_path:
            return
        if not os.path.exists(self.reference_body_path):
            print(
                f"⚠️  Reference body model not found at "
                f"{self.reference_body_path}"
            )
            return

        try:
            body = trimesh.load(self.reference_body_path, force='mesh')
        except Exception as e:
            print(f"⚠️  Failed to load reference body: {e}")
            return

        if not hasattr(body, 'vertices') or len(body.vertices) == 0:
            print("⚠️  Reference body mesh has no vertices; skipping.")
            return

        meters_per_unit = _resolve_meters_per_unit(self.reference_body_units)
        if meters_per_unit != 1.0:
            body.apply_scale(meters_per_unit)

        body_center = (body.bounds[0] + body.bounds[1]) / 2.0
        body.vertices = body.vertices - body_center

        if self.probe_mesh is not None and hasattr(self.probe_mesh, 'extents'):
            probe_size = float(np.max(self.probe_mesh.extents))
            target_size = probe_size * self.reference_body_scale_ratio * 3.0
            body_size = float(np.max(body.extents))
            if body_size > 0 and target_size > 0:
                body.apply_scale(target_size / body_size)

        # Freeze the body's world-space orientation against the initial
        # camera so future camera nudges leave it alone. ``calibrate``
        # overwrites this; ``reset_calibration`` reverts to this default.
        body_orientation = self._capture_body_world_orientation()
        self._body_orientation_world = body_orientation
        self._default_body_orientation_world = body_orientation.copy()
        pose = self._compute_center_bottom_pose()

        # Reuse the single configurable body material so the static reference
        # body and the live SMPL body share one look (config.ini [Cardiac]
        # body_mesh_color / metallic / roughness).
        body_material = self._skin_material()
        self.renderer.add_node(
            "reference_body", body,
            pose=pose,
            material=body_material,
        )
        self._reference_body_pose = pose
        self._reference_body_loaded = True

    def _compute_center_bottom_pose(self) -> np.ndarray:
        """4x4 pose that places the body reference directly below the probe tip.

        Position is world-anchored: a fixed offset along world -Y from
        the probe tip (``probe_origin_m``), captured once on the first
        pose computation and frozen so subsequent camera nudges and
        zooms leave the body planted under the tip. The frozen offset
        magnitude is derived from the construction-time camera framing
        so the body sits at a sensible distance below the tip.
        Orientation is world-anchored once ``calibrate`` has run — the
        body's axes are frozen to the world directions captured at that
        moment, so subsequent camera nudges rotate the body in step with
        the (also world-anchored) probe. Before calibration, orientation
        falls back to camera-relative so the legend reads consistently
        in screen space from launch. Body-local axes for the reference
        asset: +Y head, +Z front (face), +X body-left.
        """
        if self.camera_pose is None or self.camera_pos is None:
            return np.eye(4)

        # Lazy capture: convert the prior screen-bottom gap (~0.78 of the
        # construction-view half-height) into a world-space distance and
        # freeze. After that, camera moves don't change the offset.
        offset_below = self._body_offset_below_m
        if offset_below is None:
            distance = float(np.linalg.norm(self.camera_pos))
            half_h = distance * np.tan(self.camera_fov / 2.0)
            offset_below = 0.78 * half_h
            self._body_offset_below_m = offset_below

        offset = self.probe_origin_m.astype(float) + np.array(
            [0.0, -offset_below, 0.0]
        )

        pose = np.eye(4)
        if self._body_orientation_world is not None:
            pose[:3, :3] = self._body_orientation_world
        else:
            right_world = self.camera_pose[:3, 0]
            up_world = self.camera_pose[:3, 1]
            toward_viewer = self.camera_pose[:3, 2]
            pose[:3, 0] = toward_viewer   # body +X (body-left) → out of screen
            pose[:3, 1] = right_world     # body +Y (head)      → screen-right
            pose[:3, 2] = up_world        # body +Z (front)     → screen-up
        pose[:3, 3] = offset
        return pose
    
    def _setup_camera_aligned_lighting(self):
        """Setup directional light that points from camera to the object at origin."""
        import pyrender
        
        # Remove existing lights
        for light_name in list(self.renderer.light_nodes.keys()):
            if light_name in self.renderer.light_nodes:
                self.renderer.scene.remove_node(self.renderer.light_nodes[light_name])
                del self.renderer.light_nodes[light_name]
        
        # Calculate direction from camera to object (origin)
        camera_position = self.camera_pos
        target = np.array([0.0, 0.0, 0.0])
        light_direction = target - camera_position
        light_direction /= np.linalg.norm(light_direction)
        
        # Create a directional light that points from camera to object
        main_light = pyrender.DirectionalLight(
            color=np.array([1.0, 1.0, 1.0]),
            intensity=2.0
        )
        
        # Create pose matrix for the light
        # The light should be positioned at the camera and point toward the object
        light_pose = np.eye(4)
        light_pose[:3, 3] = camera_position
        
        # Set the light's orientation to point from camera to object
        # The directional light's direction is along -Z in its local frame
        # So we need to orient -Z to point in the light_direction
        forward = light_direction
        
        # Create an arbitrary up vector
        if abs(forward[1]) < 0.9:
            up = np.array([0.0, 1.0, 0.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up_corrected = np.cross(right, forward)
        
        light_pose[:3, 0] = right
        light_pose[:3, 1] = up_corrected
        light_pose[:3, 2] = -forward  # -Z points in the direction of light
        
        main_light_node = self.renderer.scene.add(main_light, pose=light_pose)
        self.renderer.light_nodes['camera_aligned'] = main_light_node
        
        # Add a softer fill light from the opposite direction for better visualization
        fill_light = pyrender.DirectionalLight(
            color=np.array([0.8, 0.9, 1.0]),
            intensity=0.3
        )
        fill_light_pose = np.eye(4)
        fill_light_pose[:3, 3] = -camera_position * 0.5
        fill_light_pose[:3, 0] = -right
        fill_light_pose[:3, 1] = up_corrected
        fill_light_pose[:3, 2] = forward  # Opposite direction
        
        fill_light_node = self.renderer.scene.add(fill_light, pose=fill_light_pose)
        self.renderer.light_nodes['fill'] = fill_light_node
    
    def _set_axes_pose(self, names, pose: np.ndarray, visible: bool) -> None:
        """Apply ``pose`` to each named axis node, or hide all if not visible."""
        target = pose if visible else _HIDDEN_POSE
        for name in names:
            self.renderer.set_pose(name, target)

    def _update_guidance_arrow_poses(self, R3: np.ndarray) -> None:
        """Show every visible translation/rotation arrow; hide the rest.

        Arrow geometry is baked in IMU body-frame coordinates (tip at
        origin, tail offset along the long axis), so the per-frame pose
        is just ``[R3 | probe_origin_m]`` for each arrow flagged
        visible. Visibility is controlled by
        ``_translation_arrow_visible`` / ``_rotation_arrow_visible``;
        prediction-driven updates can light multiple rotation arrows
        simultaneously.
        """
        layer_on = self._layers_visible["guidance_arrows"]
        active_pose = np.eye(4)
        active_pose[:3, :3] = R3
        active_pose[:3, 3] = self.probe_origin_m

        for direction in _TRANSLATION_DIRS.keys():
            on = layer_on and self._translation_arrow_visible[direction]
            self.renderer.set_pose(
                f"trans_arrow_{direction}",
                active_pose if on else _HIDDEN_POSE,
            )

        for direction in _ROTATION_AXES.keys():
            on = layer_on and self._rotation_arrow_visible[direction]
            self.renderer.set_pose(
                f"rot_arrow_{direction}",
                active_pose if on else _HIDDEN_POSE,
            )

        for direction in _ROLL_AXES.keys():
            on = layer_on and self._roll_arrow_visible[direction]
            self.renderer.set_pose(
                f"roll_arrow_{direction}",
                active_pose if on else _HIDDEN_POSE,
            )

    def amplify_rotation_prediction(self, rotation_matrix, current_imu_rotation):
        """
        Amplify rotation prediction and combine with current IMU rotation.
        
        Args:
            rotation_matrix: 3x3 rotation matrix (raw prediction correction)
            current_imu_rotation: 4x4 transformation matrix (current IMU pose)
            
        Returns:
            3x3 rotation matrix (amplified combined rotation)
        """
        if rotation_matrix is None:
            return None
            
        # Extract 3x3 rotation from 4x4 current_imu_rotation
        current_rotation_3x3 = current_imu_rotation[:3, :3] if current_imu_rotation.shape == (4, 4) else current_imu_rotation
        
        # Ensure rotation_matrix is 3x3
        rotation_3x3 = rotation_matrix[:3, :3] if rotation_matrix.shape == (4, 4) else rotation_matrix
        
        # Extract rotation angles and amplify
        r = R.from_matrix(rotation_3x3)
        angles = r.as_euler('xyz', degrees=True)
        exaggerated_angles = angles * self.rotation_amplification
        exaggerated_matrix = R.from_euler('xyz', exaggerated_angles, degrees=True).as_matrix()
        
        # Apply inverse transformation
        exaggerated_inverse = np.linalg.inv(exaggerated_matrix)
        
        # Combine with current IMU rotation (3x3 @ 3x3)
        return current_rotation_3x3 @ exaggerated_inverse
    
    
    def render_widget(self, input_frame_bgr: Optional[np.ndarray] = None,
                     qw: float = 1.0, qx: float = 0.0, qy: float = 0.0, qz: float = 0.0,
                     predicted_rotation_matrix: Optional[np.ndarray] = None,
                     predicted_translation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render the 3D guidance widget.

        Args:
            input_frame_bgr: Input frame (ignored for fullscreen mode)
            qw, qx, qy, qz: Quaternion components for probe orientation
            predicted_rotation_matrix: Optional 3x3 predicted rotation correction matrix (raw prediction)
            predicted_translation: Optional length-3 body-centric translation (metres,
                USGR probe frame) to the target standard view. Drives the translation
                guidance arrows (and offsets the target-probe ghost). Pass ``None`` to
                clear translation guidance.

        Returns:
            np.ndarray: Rendered frame with 3D guidance widget
        """
        if self.renderer is None:
            raise RuntimeError("Renderer not initialized")

        timing_on = self.timing_enabled
        if timing_on:
            t0 = time.perf_counter()

        # Apply any pending live patient geometry (Stevens point cloud +
        # SMPL body + contact marker). No-op unless a setter marked it dirty.
        self._apply_live_geometry()

        # Convert quaternion to rotation matrix (returns 4x4)
        rotation_matrix = quaternion_to_matrix(qw, qx, qy, qz)

        # Apply amplification to predicted rotation if provided (returns 3x3)
        amplified_predicted_rotation = None
        if predicted_rotation_matrix is not None:
            amplified_predicted_rotation = self.amplify_rotation_prediction(
                predicted_rotation_matrix, rotation_matrix)
            # Drive rotation arrow visibility from the (raw) prediction
            # so the arrow set updates every frame in lockstep with the
            # ghost axes, regardless of whether ``predicted_axes`` is
            # toggled on. Translation arrows are not auto-updated yet —
            # translation prediction is not implemented.
            self.update_rotation_arrows_from_prediction(
                predicted_rotation_matrix
            )
            # Recolor every arrow by the predicted misalignment magnitude:
            # green when the current axes already match the predicted axes,
            # ramping toward orange as the geodesic angle grows.
            self._update_arrow_alignment_color(predicted_rotation_matrix)

        # Drive translation arrows from the predicted T_rel, when provided.
        # Mirrors the rotation convention: ``None`` means "don't touch" (so
        # manual set_active_translation_arrow callers like probe_render.py keep
        # control); a vector re-derives which arrows light this frame.
        if predicted_translation is not None:
            self.update_translation_arrows_from_prediction(predicted_translation)

        if timing_on:
            t_pred = time.perf_counter()

        # Apply rotation to probe and current axes (needs 4x4). The probe
        # is pivoted around ``probe_pivot_local_m`` (the tip, in mesh-local
        # coords) so the tip stays planted at ``probe_origin_m`` while the
        # body swings around it — matching a sonographer's physical pivot.
        # ``probe_axis_fix_matrix`` pre-rotates mesh vertices into the IMU
        # frame so the live quaternion means what the IMU thinks it means.
        # Axes represent the IMU frame directly and receive R3 alone.
        if rotation_matrix is not None:
            # Left-multiply by ``calibration_matrix`` so the calibration
            # pose maps to the canonical render orientation and live
            # rotations follow the probe's body axes; identity until the
            # operator calls ``calibrate()``.
            R3 = self.calibration_matrix @ rotation_matrix[:3, :3]
            # Cache for the target-probe ghost translation offset (maps the
            # body-frame T_rel into the current render-world frame).
            self._last_R3 = R3
            R_probe = R3 @ self.probe_axis_fix_matrix
            probe_pose = np.eye(4)
            probe_pose[:3, :3] = R_probe
            probe_pose[:3, 3] = self.probe_origin_m - R_probe @ self.probe_pivot_local_m
            for name, _sub_tri, _pr_mesh in self.probe_node_specs:
                self.renderer.set_pose(name, probe_pose)

            # Axes originate at the probe tip (``probe_origin_m``), same as the
            # probe mesh and guidance arrows, so they ride with the probe when it
            # is off the scene origin (e.g. the ``fixed_body`` render anchor).
            # In the legacy ``fixed_probe`` framing the tip sits at the origin, so
            # this is a no-op there.
            axes_pose = np.eye(4)
            axes_pose[:3, :3] = R3
            axes_pose[:3, 3] = self.probe_origin_m
            self._set_axes_pose(
                ("x_axis", "y_axis", "z_axis"),
                axes_pose, visible=self._layers_visible["current_axes"],
            )

            self._update_guidance_arrow_poses(R3)

        # Render-world displacement to the target view, shared by the predicted
        # axes and the ghost so both sit at the SAME target location. The
        # body-frame T_rel is mapped into render-world via the current
        # orientation (_last_R3). Zero when no translation is predicted this
        # frame. Frame/scale (usgr_to_body, guidance_translation_amplification)
        # need empirical calibration.
        target_translation_offset = np.zeros(3)
        if predicted_translation is not None:
            t_body = (
                self.usgr_to_body
                @ np.asarray(predicted_translation, dtype=float).reshape(3)
            ) * self.guidance_translation_amplification
            target_translation_offset = self._last_R3 @ t_body

        # Predicted axes follow the prediction when one is available AND
        # the layer is on; otherwise they're hidden off-screen. They originate
        # from the current-axes origin (the probe tip, ``probe_origin_m``) PLUS
        # the predicted translation, so a needed move shows the predicted frame
        # shifted away from the current.
        predicted_layer_on = self._layers_visible["predicted_axes"]
        if predicted_layer_on and amplified_predicted_rotation is not None:
            predicted_pose = np.eye(4)
            predicted_pose[:3, :3] = (
                self.calibration_matrix @ amplified_predicted_rotation
            )
            predicted_pose[:3, 3] = self.probe_origin_m + target_translation_offset
            self._set_axes_pose(
                ("predicted_x_axis", "predicted_y_axis", "predicted_z_axis"),
                predicted_pose, visible=True,
            )
        else:
            self._set_axes_pose(
                ("predicted_x_axis", "predicted_y_axis", "predicted_z_axis"),
                _HIDDEN_POSE, visible=False,
            )

        # Target probe: same pivot/origin as the live probe, but rotated
        # to the (amplified) predicted pose so it shows the operator
        # "where the probe should be". Hidden when the layer is off or
        # when there's no prediction this frame.
        target_layer_on = self._layers_visible["target_probe"]
        if target_layer_on and amplified_predicted_rotation is not None:
            R_target = (
                self.calibration_matrix @ amplified_predicted_rotation
                @ self.probe_axis_fix_matrix
            )
            target_probe_pose = np.eye(4)
            target_probe_pose[:3, :3] = R_target
            # Tip placed at the live tip PLUS the predicted translation, using the
            # same render-world offset as the predicted axes so both agree on the
            # target location.
            target_probe_pose[:3, 3] = (
                self.probe_origin_m - R_target @ self.probe_pivot_local_m
                + target_translation_offset
            )
            for name in self.target_probe_node_names:
                self.renderer.set_pose(name, target_probe_pose)
        else:
            for name in self.target_probe_node_names:
                self.renderer.set_pose(name, _HIDDEN_POSE)

        if timing_on:
            t_pose = time.perf_counter()

        # Render the scene. pyrender.OffscreenRenderer.render() does the
        # GL submission AND a blocking glReadPixels back to CPU, so by the
        # time it returns the GPU is fully synced — the timestamp below
        # captures rasterization + readback together.
        if self.wireframe:
            import pyrender
            color, _ = self.renderer.render(flags=pyrender.RenderFlags.ALL_WIREFRAME)
        else:
            color, _ = self.renderer.render()

        if timing_on:
            t_gl = time.perf_counter()

        # make a copy of the color
        color = color.copy()
        rendered_frame_bgra = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)

        # Overlay legend only when the predicted-axes layer is on. The
        # legend's bottom row keys the translucent predicted-axis colors,
        # so it has no referent once those axes are hidden — and the 'x'
        # hotkey toggles current + predicted axes together, so dropping
        # the legend with them keeps the widget uncluttered.
        if self._legend_enabled and self._layers_visible["predicted_axes"]:
            self._draw_guidance_legend(rendered_frame_bgra)
        output_frame = rendered_frame_bgra

        # Handle different render modes
        if self.render_mode == 'overlay' and input_frame_bgr is not None:
            result = self._overlay_on_frame(output_frame, input_frame_bgr)
        else:
            result = output_frame

        if timing_on:
            t_post = time.perf_counter()
            self.last_timings = {
                "pred_update": (t_pred - t0)    * 1e3,
                "pose_update": (t_pose - t_pred) * 1e3,
                "gl_render":   (t_gl   - t_pose) * 1e3,
                "output_post": (t_post - t_gl)   * 1e3,
            }

        return result

    @staticmethod
    def _axis_bgra(axis: str) -> tuple:
        """Return the BGRA (uint8, opaque) legend colour for an axis."""
        r, g, b = _AXIS_RGB[axis]
        return (int(b * 255), int(g * 255), int(r * 255), 255)

    @staticmethod
    def _draw_dashed_bar(frame: np.ndarray, x: int, y: int, w: int, h: int,
                         color: tuple) -> None:
        """Draw a horizontal dashed bar (filled dashes + gaps) in ``frame``.

        Used for the "Predicted" legend row: a dashed swatch reads as
        "predicted / not-yet-there" far more legibly than the old faded
        translucent block, while staying crisp (it's a static key, so the
        dash-shimmer that rules dashes out for the live 3D axes never applies).

        Fixed at three dashes with an airy gap so the "predicted" cue stays
        unambiguous rather than reading as a near-solid bar.
        """
        n_dashes = 3
        gap_ratio = 0.975   # gap as a fraction of dash length (>0.5 = airier)
        dash = max(2, int(round(w / (n_dashes + (n_dashes - 1) * gap_ratio))))
        gap = max(2, int(round(dash * gap_ratio)))
        cx = x
        for _ in range(n_dashes):
            seg = min(dash, x + w - cx)
            if seg <= 0:
                break
            cv2.rectangle(frame, (cx, y), (cx + seg, y + h), color, -1,
                          cv2.LINE_AA)
            cx += dash + gap

    def _draw_guidance_legend(self, frame_bgra: np.ndarray) -> None:
        """Draw the axis legend directly onto the BGRA widget frame.

        Vector-drawn at the frame's native resolution (no baked-PNG resample),
        with swatch colours pulled from the same ``_AXIS_RGB`` constants that
        drive the 3D axis materials — so the key can never drift from the axes.

        Two labelled rows, one swatch per axis colour:
          * "Current" — solid bars  → the opaque current-pose axes.
          * "Target"  — dashed bars → the translucent predicted-pose axes.
        Labels use a friendly TrueType font (see ``_load_legend_font``) so the
        legend reads as a clinical UI, not an engineering plot. All geometry is
        proportional to frame size so it stays consistent across the
        256/512/640 window sizes the widget is instantiated at.
        """
        H, W = frame_bgra.shape[:2]
        has_alpha = frame_bgra.shape[2] == 4

        # --- geometry (all proportional to frame size) ---
        pad = max(6, int(round(H * 0.022)))       # inner panel padding
        margin = max(8, int(round(H * 0.03)))     # gap from the frame corner
        row_h = max(12, int(round(H * 0.06)))     # per data-row height
        swatch_w = max(24, int(round(W * 0.06)))  # swatch length (1.2x)
        swatch_h = max(4, int(round(H * 0.017)))  # swatch thickness
        col_gap = max(4, int(round(W * 0.012)))   # gap between swatches
        label_gap = max(6, int(round(W * 0.018)))  # gap label->grid

        # Friendly TrueType label font; fall back to cv2 Hershey if none load.
        pil_font = _load_legend_font(max(11, int(round(H * 0.045))))
        cv_scale = max(0.34, H * 0.0011)
        cv_thick = max(1, int(round(H * 0.0028)))
        text_col = (238, 238, 238)   # BGR

        def _text_size(text: str) -> tuple:
            if pil_font is not None:
                left, top, right, bottom = pil_font.getbbox(text)
                return right - left, bottom - top
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, cv_scale, cv_thick)
            return tw, th

        def _draw_label(text: str, x: int, cy: int) -> None:
            _tw, th = _text_size(text)
            if pil_font is not None:
                _blit_pil_text(frame_bgra, text, x, cy - th // 2,
                               pil_font, text_col)
            else:
                cv2.putText(frame_bgra, text, (x, cy + th // 2),
                            cv2.FONT_HERSHEY_DUPLEX, cv_scale,
                            (*text_col, 255), cv_thick, cv2.LINE_AA)

        labels = ("Current", "Target")
        label_w = max(_text_size(t)[0] for t in labels)

        n = len(_AXIS_ORDER)
        grid_w = n * swatch_w + (n - 1) * col_gap
        panel_w = pad * 2 + label_w + label_gap + grid_w
        panel_h = pad * 2 + 2 * row_h
        x0, y0 = margin, margin
        x1 = min(x0 + panel_w, W)
        y1 = min(y0 + panel_h, H)

        # --- scrim: manual alpha blend so it works whether or not the frame's
        # own alpha channel is honoured downstream (ROS publish vs Qt) ---
        roi = frame_bgra[y0:y1, x0:x1]
        scrim = np.array([28, 26, 24], dtype=np.float32)  # near-black BGR
        a = 0.58
        roi[..., :3] = (
            roi[..., :3] * (1.0 - a) + scrim * a
        ).astype(np.uint8)
        if has_alpha:
            roi[..., 3] = 255

        grid_x0 = x0 + pad + label_w + label_gap
        col_x = [grid_x0 + i * (swatch_w + col_gap) for i in range(n)]

        # --- two data rows: solid (current) then dashed (target) ---
        row_top = y0 + pad
        for r, label in enumerate(labels):
            cy = row_top + r * row_h + row_h // 2
            _draw_label(label, x0 + pad, cy)
            dashed = (r == 1)
            for i, ax in enumerate(_AXIS_ORDER):
                color = self._axis_bgra(ax)
                sx, sy = col_x[i], cy - swatch_h // 2
                if dashed:
                    self._draw_dashed_bar(frame_bgra, sx, sy, swatch_w,
                                          swatch_h, color)
                else:
                    cv2.rectangle(frame_bgra, (sx, sy),
                                  (sx + swatch_w, sy + swatch_h), color, -1,
                                  cv2.LINE_AA)

    def _overlay_on_frame(self, widget_frame: np.ndarray, input_frame: np.ndarray) -> np.ndarray:
        """Overlay the 3D widget on an input frame."""
        # Simple overlay in top-left corner (can be enhanced)
        h, w = input_frame.shape[:2]
        widget_h, widget_w = widget_frame.shape[:2]
        
        # Resize widget if needed
        overlay_size = min(h, w) // 4
        if widget_w != overlay_size or widget_h != overlay_size:
            import cv2
            widget_frame = cv2.resize(widget_frame, (overlay_size, overlay_size))
        
        # Overlay on input frame
        result = input_frame.copy()
        margin = 10
        y1, y2 = margin, margin + overlay_size
        x1, x2 = margin, margin + overlay_size
        
        if y2 <= h and x2 <= w:
            result[y1:y2, x1:x2] = widget_frame
        
        return result
    
    def set_render_mode(self, mode: str):
        """Set the rendering mode."""
        if mode in ['fullscreen', 'overlay']:
            self.render_mode = mode
        else:
            raise ValueError(f"Invalid render mode: {mode}")
    
    def get_render_mode(self) -> str:
        """Get the current rendering mode."""
        return self.render_mode
    
    def set_wireframe(self, wireframe: bool):
        """Enable or disable wireframe rendering."""
        self.wireframe = wireframe
    
    def get_wireframe(self) -> bool:
        """Get current wireframe state."""
        return self.wireframe
    
    def destroy(self):
        """Clean up 3D renderer resources."""
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
