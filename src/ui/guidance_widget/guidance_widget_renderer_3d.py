"""
3D guidance widget renderer using Generic3DRenderer.
Provides 3D visualization of probe orientation and coordinate axes.
"""

import os
import numpy as np
import trimesh
import time
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple

from .guidance_widget_renderer import GuidanceWidgetRenderer
from ..generic_3d_renderer import (
    Generic3DRenderer,
    create_translation_matrix,
    create_rotation_matrix,
    quaternion_to_matrix
)
from ..image_window.window_factory import create_image_window


class GuidanceWidgetRenderer3D(GuidanceWidgetRenderer):
    """
    3D guidance widget renderer with probe mesh and coordinate axes visualization.
    """
    
    def __init__(self, probe_path: str,
                 background_color=(0.1, 0.1, 0.15, 1.0),
                 scale_factor=2.0,
                 wireframe=False,
                 window_size=(640, 480),
                 render_mode='fullscreen',
                 rotation_amplification=3.0,
                 **kwargs):
        """
        Initialize 3D guidance widget renderer.
        
        Args:
            probe_path: Path to the probe STL file
            background_color: Background color as RGBA tuple
            scale_factor: Scaling factor for probe and axes size
            wireframe: Whether to render in wireframe mode
            window_size: Window dimensions (width, height)
            render_mode: Rendering mode ('fullscreen' or 'overlay')
            rotation_amplification: Amplification factor for rotation prediction
            **kwargs: Additional configuration parameters
        """
        super().__init__(probe_path, **kwargs)
        
        # Store 3D-specific parameters
        self.background_color = background_color
        self.scale_factor = scale_factor
        self.wireframe = wireframe
        self.rotation_amplification = rotation_amplification
        self.window_size = window_size
        self.render_mode = render_mode
        
        # Initialize renderer components
        self.renderer = None
        self.probe_mesh = None
        self.axes = None
        self.materials = None
        self.camera_pose = None
        self.camera_pos = None
        
        # Initialize the 3D renderer
        self._initialize_renderer()
    
    def _initialize_renderer(self):
        """Initialize the Generic3DRenderer and load assets."""
        try:
            # Initialize renderer
            self.renderer = Generic3DRenderer(
                width=self.window_size[0],
                height=self.window_size[1],
                background_color=self.background_color,
                ambient_light=np.array([0.2, 0.2, 0.2])  # Lower ambient for better contrast
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
        """Load probe mesh from STL file."""
        if os.path.exists(self.probe_path):
            print(f"Loading probe STL: {self.probe_path}")
            self.probe_mesh = self.renderer.load_mesh_from_file(
                self.probe_path, 
                scale=0.01 * self.scale_factor, 
                center=True
            )
            if self.probe_mesh is None:
                print("Failed to load probe STL, using capsule fallback")
                self.probe_mesh = trimesh.creation.capsule(
                    radius=0.2 * self.scale_factor, 
                    height=2.0 * self.scale_factor
                )
        else:
            print(f"Probe STL not found: {self.probe_path}, using capsule")
            self.probe_mesh = trimesh.creation.capsule(
                radius=0.2 * self.scale_factor, 
                height=2.0 * self.scale_factor
            )
    
    def _create_coordinate_axes(self):
        """Create coordinate axes cylinders."""
        self.axes = self._create_axes_cylinders(
            length=1.2 * self.scale_factor, 
            radius=0.02 * self.scale_factor
        )
        # Create predicted axes (semi-transparent version)
        self.predicted_axes = self._create_axes_cylinders(
            length=1.2 * self.scale_factor,
            radius=0.02 * self.scale_factor
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
                metallicFactor=0.1,
                roughnessFactor=0.7
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
    
    def _setup_camera(self):
        """Setup camera position and orientation."""
        # Adjust camera distance based on scale factor
        camera_distance = 5.0 * max(1.0, self.scale_factor * 0.8)
        self.camera_pose = np.eye(4)
        self.camera_pose[:3, 3] = [3.0 * camera_distance/5.0, 2.0 * camera_distance/5.0, camera_distance]

        target = np.array([0.0, 0.0, 0.0])
        self.camera_pos = self.camera_pose[:3, 3]
        up = np.array([0.0, 1.0, 0.0])

        forward = target - self.camera_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up_corrected = np.cross(right, forward)

        self.camera_pose[:3, 0] = right
        self.camera_pose[:3, 1] = up_corrected
        self.camera_pose[:3, 2] = -forward
        self.renderer.set_camera_pose(self.camera_pose)
    
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
    
    
    def render_widget(self, input_frame: Optional[np.ndarray] = None,
                     qw: float = 1.0, qx: float = 0.0, qy: float = 0.0, qz: float = 0.0,
                     predicted_rotation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render the 3D guidance widget.
        
        Args:
            input_frame: Input frame (ignored for fullscreen mode)
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
        
        
        # Apply rotation to probe and current axes (needs 4x4)
        if rotation_matrix is not None:
            self.renderer.set_pose("probe", rotation_matrix)
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
        
        # Handle different render modes
        if self.render_mode == 'overlay' and input_frame is not None:
            return self._overlay_on_frame(color, input_frame)
        else:
            return color
    
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
    
    def set_scale_factor(self, scale_factor: float):
        """Update the scale factor and reinitialize if needed."""
        if scale_factor != self.scale_factor:
            self.scale_factor = scale_factor
            # Would need to reinitialize meshes and camera for full effect
            self._setup_camera()  # At least update camera
    
    def destroy(self):
        """Clean up 3D renderer resources."""
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
