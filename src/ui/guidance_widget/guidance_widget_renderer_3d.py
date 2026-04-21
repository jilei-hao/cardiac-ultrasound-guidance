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
                 axes_radius_m: float = 0.003,
                 camera_distance_m: Optional[float] = None,
                 camera_azimuth_deg: float = 135.0,
                 camera_elevation_deg: float = 22.0,
                 probe_origin_m: tuple = (0.0, 0.0, 0.0),
                 **kwargs):
        """
        Initialize 3D guidance widget renderer.

        Args:
            probe_path: Path to the probe STL file.
            background_color: Background color as RGBA tuple.
            wireframe: Whether to render in wireframe mode.
            window_size: Window dimensions (width, height) in pixels.
            render_mode: 'fullscreen' or 'overlay'.
            rotation_amplification: Amplification factor for rotation
                prediction.
            legend_image_path: Optional path to a legend image overlay.
            probe_units: Native unit of the probe STL ('mm', 'cm', 'dm',
                'm', 'in', or a meters-per-unit float). Default 'mm'.
            axes_length_m: Length of each coordinate-axis shaft in meters.
            axes_radius_m: Shaft radius of each coordinate axis in meters.
            camera_distance_m: Camera distance from the scene origin in
                meters. If None, auto-fitted from the scene AABB.
            camera_azimuth_deg: Camera azimuth (rotation around +Y axis)
                in degrees.
            camera_elevation_deg: Camera elevation (pitch above the XZ
                plane) in degrees.
            probe_origin_m: Reserved translation for positional tracking;
                currently defaults to the origin.
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
        self.axes_radius_m = axes_radius_m
        self.camera_distance_m = camera_distance_m
        self.camera_azimuth_deg = camera_azimuth_deg
        self.camera_elevation_deg = camera_elevation_deg
        self.probe_origin_m = np.asarray(probe_origin_m, dtype=float)

        self.renderer = None
        self.probe_mesh = None
        self.axes = None
        self.materials = None
        self.camera_pose = None
        self.camera_pos = None

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
            
            if not self.renderer.initialize(camera_fov=np.pi / 4):
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
            
        except Exception as e:
            if self.renderer:
                self.renderer.destroy()
            raise RuntimeError(f"Failed to initialize 3D renderer: {e}")
    
    def _load_probe_mesh(self):
        """Load probe mesh from STL file and convert to meters."""
        fallback_radius_m = 0.022
        fallback_height_m = 0.15

        if os.path.exists(self.probe_path):
            print(f"Loading probe STL: {self.probe_path}")
            self.probe_mesh = self.renderer.load_mesh_from_file(
                self.probe_path,
                source_units=self.probe_units,
                center=True,
            )
            if self.probe_mesh is not None:
                diag = float(np.linalg.norm(self.probe_mesh.extents))
                if not (0.02 <= diag <= 0.40):
                    print(
                        f"⚠️  Probe mesh bbox diagonal {diag:.3f} m is outside "
                        f"[0.02, 0.40] m — asset may be mislabeled "
                        f"(probe_units={self.probe_units})."
                    )
            else:
                print("Failed to load probe STL, using capsule fallback")
                self.probe_mesh = trimesh.creation.capsule(
                    radius=fallback_radius_m,
                    height=fallback_height_m,
                )
        else:
            print(f"Probe STL not found: {self.probe_path}, using capsule")
            self.probe_mesh = trimesh.creation.capsule(
                radius=fallback_radius_m,
                height=fallback_height_m,
            )
    
    def _create_coordinate_axes(self):
        """Create coordinate axes cylinders (sizes in meters)."""
        self.axes = self._create_axes_cylinders(
            length=self.axes_length_m,
            radius=self.axes_radius_m,
        )
        # Create predicted axes (semi-transparent version)
        self.predicted_axes = self._create_axes_cylinders(
            length=self.axes_length_m,
            radius=self.axes_radius_m,
        )
    
    def _create_axes_cylinders(self, length: float = 1.0, radius: float = 0.02) -> dict:
        """Create coordinate axes as trimesh cylinder objects."""
        axes = {}
        
        # X-axis (red) - cylinder extending from origin along +X
        x_shaft = trimesh.creation.cylinder(radius=radius, height=length)
        x_shaft.apply_transform(create_rotation_matrix('y', np.pi/2))
        x_shaft.apply_transform(create_translation_matrix(length/2, 0, 0))
        axes['x'] = x_shaft
        
        # Y-axis (green) - cylinder extending from origin along +Y
        y_shaft = trimesh.creation.cylinder(radius=radius, height=length)
        y_shaft.apply_transform(create_rotation_matrix('x', -np.pi/2))
        y_shaft.apply_transform(create_translation_matrix(0, length/2, 0))
        axes['y'] = y_shaft
        
        # Z-axis (blue) - cylinder extending from origin along +Z
        z_shaft = trimesh.creation.cylinder(radius=radius, height=length)
        z_shaft.apply_transform(create_translation_matrix(0, 0, length/2))
        axes['z'] = z_shaft
        
        return axes
    
    def _create_materials(self):
        """Create materials for probe and axes."""
        import pyrender
        
        self.materials = {
            'probe': pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.98, 0.98, 0.98, 1.0],  # White plastic
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
        
        # Add probe and current axes
        self.renderer.add_node("probe", self.probe_mesh, material=self.materials['probe'])
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
        axes_extent = self.axes_length_m
        probe_extent = 0.0
        if self.probe_mesh is not None and hasattr(self.probe_mesh, 'extents'):
            probe_extent = float(np.max(self.probe_mesh.extents))
        bbox_side = max(axes_extent, probe_extent)
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
        # also picks up ``probe_origin_m`` so positional tracking can later
        # place it anywhere in the scene; axes stay anchored at the origin.
        if rotation_matrix is not None:
            probe_pose = rotation_matrix.copy()
            probe_pose[:3, 3] = self.probe_origin_m
            self.renderer.set_pose("probe", probe_pose)
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
