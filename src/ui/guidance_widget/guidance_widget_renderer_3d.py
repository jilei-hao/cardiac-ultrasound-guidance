"""
3D guidance widget renderer using Generic3DRenderer.
Provides 3D visualization of probe orientation and coordinate axes.
"""

import os
import numpy as np
import trimesh
import time
import cv2
from ..image_helpers import overlay_color_legend
from scipy.spatial.transform import Rotation as R
from typing import Optional, Union

from .guidance_widget_renderer import GuidanceWidgetRenderer
from ..generic_3d_renderer import (
    Generic3DRenderer,
    _resolve_meters_per_unit,
    create_translation_matrix,
    create_rotation_matrix,
    quaternion_to_matrix,
)


class GuidanceWidgetRenderer3D(GuidanceWidgetRenderer):
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
                 axes_length_m: float = 0.12,
                 y_axis_length_m: Optional[float] = 0.20,
                 axes_radius_m: float = 0.003,
                 camera_distance_m: Optional[float] = None,
                 camera_azimuth_deg: float = 135.0,
                 camera_elevation_deg: float = 22.0,
                 probe_origin_m: tuple = (0.0, 0.0, 0.0),
                 probe_pivot_local_m: Optional[tuple] = None,
                 probe_pivot_axis: str = 'y',
                 probe_pivot_sign: int = 1,
                 probe_axis_fix_euler_deg: Optional[tuple] = (180.0, 0.0, 0.0),
                 reference_body_path: Optional[str] = None,
                 reference_body_units: Union[str, float] = 'cm',
                 reference_body_scale_ratio: float = 0.75,
                 reference_body_corner: str = 'bottom-left',
                 reference_body_margin: float = 0.78,
                 reference_body_h_margin: Optional[float] = 0.5,
                 **kwargs):
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
            axes_length_m: Length of each coordinate-axis shaft in meters.
            y_axis_length_m: Optional override for the Y-axis shaft length
                in meters. Defaults longer than ``axes_length_m`` so the
                green axis sticks out past the probe body and stays
                visible. Pass None to use ``axes_length_m`` for Y too.
            axes_radius_m: Shaft radius of each coordinate axis in meters.
            camera_distance_m: Camera distance from the scene origin in
                meters. If None, auto-fitted from the scene AABB.
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
                into the Clarius IMU frame before the live quaternion is
                applied, so the probe renders in the correct orientation
                when the IMU reports identity. Leave as None for an
                IMU-aligned mesh (see scripts/scale_probe_stl.py
                --align-imu); set empirically otherwise, per the static
                pose verification in the Clarius motion docs
                (https://github.com/clariusdev/motion).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(probe_path, **kwargs)

        self.background_color = background_color
        self.wireframe = wireframe
        self.rotation_amplification = rotation_amplification
        self.window_size = window_size
        self.render_mode = render_mode
        self.legend_image_path = legend_image_path

        self.probe_units = probe_units
        self.axes_length_m = axes_length_m
        self.y_axis_length_m = (
            float(y_axis_length_m)
            if y_axis_length_m is not None else float(axes_length_m)
        )
        self.axes_radius_m = axes_radius_m
        self.camera_distance_m = camera_distance_m
        self.camera_azimuth_deg = camera_azimuth_deg
        self.camera_elevation_deg = camera_elevation_deg
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

        self.reference_body_path = reference_body_path
        self.reference_body_units = reference_body_units
        self.reference_body_scale_ratio = float(reference_body_scale_ratio)
        self.reference_body_corner = reference_body_corner
        self.reference_body_margin = float(reference_body_margin)
        self.reference_body_h_margin = (
            float(reference_body_h_margin)
            if reference_body_h_margin is not None
            else float(reference_body_margin)
        )

        self.renderer = None
        self.probe_mesh = None
        self.probe_node_specs: list = []
        self.axes = None
        self.materials = None
        self.camera_pose = None
        self.camera_pos = None
        self.camera_fov = np.pi / 4

        self.legend_image_bgra = cv2.imread(self.legend_image_path, cv2.IMREAD_UNCHANGED)

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
        )
        # Create predicted axes (semi-transparent version)
        self.predicted_axes = self._create_axes_cylinders(
            length=self.axes_length_m,
            y_length=self.y_axis_length_m,
            radius=self.axes_radius_m,
        )

    def _create_axes_cylinders(self, length: float = 1.0,
                               y_length: Optional[float] = None,
                               radius: float = 0.02) -> dict:
        """Create coordinate axes as trimesh cylinder objects."""
        axes = {}
        y_len = float(y_length) if y_length is not None else length

        # X-axis (red) - cylinder extending from origin along +X
        x_shaft = trimesh.creation.cylinder(radius=radius, height=length)
        x_shaft.apply_transform(create_rotation_matrix('y', np.pi/2))
        x_shaft.apply_transform(create_translation_matrix(length/2, 0, 0))
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
            'x': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[1.0, 0.2, 0.2, 1.0],  # Bright red
                metallicFactor=0.3,
                roughnessFactor=0.6
            ),
            'y': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.2, 1.0, 0.2, 1.0],  # Bright green
                metallicFactor=0.3,
                roughnessFactor=0.6
            ),
            'z': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.2, 0.2, 1.0, 1.0],  # Bright blue
                metallicFactor=0.3,
                roughnessFactor=0.6
            ),
            # Semi-transparent versions for predicted axes
            'x_predicted': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[1.0, 0.2, 0.2, 0.3],  # Bright red, 30% opacity
                metallicFactor=0.3,
                roughnessFactor=0.6,
                alphaMode='BLEND'
            ),
            'y_predicted': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.2, 1.0, 0.2, 0.3],  # Bright green, 30% opacity
                metallicFactor=0.3,
                roughnessFactor=0.6,
                alphaMode='BLEND'
            ),
            'z_predicted': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.2, 0.2, 1.0, 0.3],  # Bright blue, 30% opacity
                metallicFactor=0.3,
                roughnessFactor=0.6,
                alphaMode='BLEND'
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

        self.renderer.add_node("x_axis", pyrender.Mesh.from_trimesh(self.axes['x'], material=self.materials['x']))
        self.renderer.add_node("y_axis", pyrender.Mesh.from_trimesh(self.axes['y'], material=self.materials['y']))
        self.renderer.add_node("z_axis", pyrender.Mesh.from_trimesh(self.axes['z'], material=self.materials['z']))
        
        # Add predicted axes (semi-transparent)
        self.renderer.add_node("predicted_x_axis", pyrender.Mesh.from_trimesh(self.axes['x'], material=self.materials['x_predicted']))
        self.renderer.add_node("predicted_y_axis", pyrender.Mesh.from_trimesh(self.axes['y'], material=self.materials['y_predicted']))
        self.renderer.add_node("predicted_z_axis", pyrender.Mesh.from_trimesh(self.axes['z'], material=self.materials['z_predicted']))
    
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
            target_size = probe_size * self.reference_body_scale_ratio
            body_size = float(np.max(body.extents))
            if body_size > 0 and target_size > 0:
                body.apply_scale(target_size / body_size)

        pose = self._compute_corner_pose(self.reference_body_corner)

        import pyrender
        body_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.85, 0.78, 0.72, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.85,
        )
        self.renderer.add_node(
            "reference_body", body,
            pose=pose,
            material=body_material,
        )

    def _compute_corner_pose(self, corner: str) -> np.ndarray:
        """4x4 pose that places a node at one corner of the camera view.

        Uses the camera basis vectors plus the vertical FOV to convert a
        normalized screen offset into a world-space translation at the
        same depth as the scene origin. The rotation lays the body
        horizontally with head pointing to screen-right and chest facing
        screen-up (i.e. lying on its back, viewed from the body's left
        side), regardless of camera angle. Body-local axes for the
        reference asset are +Y up (head), +Z front (face), +X body-left.
        """
        if self.camera_pose is None or self.camera_pos is None:
            return np.eye(4)

        distance = float(np.linalg.norm(self.camera_pos))
        aspect = self.window_size[0] / max(1, self.window_size[1])
        half_h = distance * np.tan(self.camera_fov / 2.0)
        half_w = half_h * aspect

        sign_x = +1 if 'right' in corner else -1
        sign_y = +1 if 'top' in corner else -1

        right_world = self.camera_pose[:3, 0]
        up_world = self.camera_pose[:3, 1]
        toward_viewer = self.camera_pose[:3, 2]

        offset = (
            right_world * (sign_x * half_w * self.reference_body_h_margin)
            + up_world * (sign_y * half_h * self.reference_body_margin)
        )

        pose = np.eye(4)
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
                     predicted_rotation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render the 3D guidance widget.
        
        Args:
            input_frame_bgr: Input frame (ignored for fullscreen mode)
            qw, qx, qy, qz: Quaternion components for probe orientation
            predicted_rotation_matrix: Optional 3x3 predicted rotation correction matrix (raw prediction)
            
        Returns:
            np.ndarray: Rendered frame with 3D guidance widget
        """
        if self.renderer is None:
            raise RuntimeError("Renderer not initialized")
        
        # Convert quaternion to rotation matrix (returns 4x4)
        rotation_matrix = quaternion_to_matrix(qw, qx, qy, qz)
        
        # Apply amplification to predicted rotation if provided (returns 3x3)
        amplified_predicted_rotation = None
        if predicted_rotation_matrix is not None:
            amplified_predicted_rotation = self.amplify_rotation_prediction(
                predicted_rotation_matrix, rotation_matrix)
        
        
        # Apply rotation to probe and current axes (needs 4x4). The probe
        # is pivoted around ``probe_pivot_local_m`` (the tip, in mesh-local
        # coords) so the tip stays planted at ``probe_origin_m`` while the
        # body swings around it — matching a sonographer's physical pivot.
        # ``probe_axis_fix_matrix`` pre-rotates mesh vertices into the IMU
        # frame so the live quaternion means what the IMU thinks it means.
        # Axes represent the IMU frame directly and receive R3 alone.
        if rotation_matrix is not None:
            R3 = rotation_matrix[:3, :3]
            R_probe = R3 @ self.probe_axis_fix_matrix
            probe_pose = np.eye(4)
            probe_pose[:3, :3] = R_probe
            probe_pose[:3, 3] = self.probe_origin_m - R_probe @ self.probe_pivot_local_m
            for name, _sub_tri, _pr_mesh in self.probe_node_specs:
                self.renderer.set_pose(name, probe_pose)
            self.renderer.set_pose("x_axis", rotation_matrix)
            self.renderer.set_pose("y_axis", rotation_matrix)
            self.renderer.set_pose("z_axis", rotation_matrix)
        
        # Apply amplified predicted rotation to predicted axes
        if amplified_predicted_rotation is not None:
            # Convert 3x3 rotation matrix to 4x4 transformation matrix
            predicted_pose = np.eye(4)
            predicted_pose[:3, :3] = amplified_predicted_rotation
            
            self.renderer.set_pose("predicted_x_axis", predicted_pose)
            self.renderer.set_pose("predicted_y_axis", predicted_pose)
            self.renderer.set_pose("predicted_z_axis", predicted_pose)
        else:
            # Hide predicted axes if no prediction available (move far away)
            hidden_pose = np.eye(4)
            hidden_pose[:3, 3] = [1000, 1000, 1000]  # Move off-screen
            
            self.renderer.set_pose("predicted_x_axis", hidden_pose)
            self.renderer.set_pose("predicted_y_axis", hidden_pose)
            self.renderer.set_pose("predicted_z_axis", hidden_pose)
        
        # Render the scene
        if self.wireframe:
            import pyrender
            color, _ = self.renderer.render(flags=pyrender.RenderFlags.ALL_WIREFRAME)
        else:
            color, _ = self.renderer.render()

        # make a copy of the color
        color = color.copy()
        rendered_frame_bgra = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)

        # Overlay legend if available
        if self.legend_image_bgra is not None:
            output_frame = overlay_color_legend(rendered_frame_bgra, self.legend_image_bgra, top=10, left=10, pct_width=0.5)
        else:
            output_frame = rendered_frame_bgra

        # Handle different render modes
        if self.render_mode == 'overlay' and input_frame_bgr is not None:
            return self._overlay_on_frame(output_frame, input_frame_bgr)
        else:
            return output_frame
    
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
