"""
Generic 3D Renderer using pyrender and trimesh for off-screen rendering.
Provides a clean interface for managing 3D scenes with mesh nodes.
"""

import numpy as np
import pyrender
import trimesh
from typing import Dict, Optional, Tuple, Union, Any
import os


UNIT_TO_METERS: Dict[str, float] = {
    'm': 1.0,
    'dm': 0.1,
    'cm': 0.01,
    'mm': 0.001,
    'in': 0.0254,
}


def _resolve_meters_per_unit(source_units: Union[str, float]) -> float:
    if isinstance(source_units, (int, float)):
        return float(source_units)
    try:
        return UNIT_TO_METERS[source_units]
    except KeyError as exc:
        raise ValueError(
            f"Unknown source_units '{source_units}'. "
            f"Expected one of {sorted(UNIT_TO_METERS)} or a meters-per-unit float."
        ) from exc


class Generic3DRenderer:
    """
    Generic 3D renderer using pyrender for off-screen rendering.
    
    This class provides a simple interface for creating and managing 3D scenes,
    adding mesh nodes, and updating their poses. It uses pyrender's OffscreenRenderer
    for efficient rendering without requiring a display.
    
    Example:
        renderer = Generic3DRenderer(width=512, height=512)
        renderer.initialize()
        
        # Add a mesh
        mesh = trimesh.creation.icosphere()
        node = renderer.add_node("sphere", mesh, pose=np.eye(4))
        
        # Update pose
        new_pose = np.eye(4)
        new_pose[:3, 3] = [1, 0, 0]  # Translate
        renderer.set_pose("sphere", new_pose)
        
        # Render
        color, depth = renderer.render()
    """
    
    def __init__(self, 
                 width: int = 512, 
                 height: int = 512,
                 background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
                 ambient_light: Optional[np.ndarray] = None):
        """
        Initialize the Generic3DRenderer.
        
        Args:
            width: Render buffer width in pixels
            height: Render buffer height in pixels
            background_color: RGBA background color (values 0-1)
            ambient_light: RGB ambient light color (values 0-1). If None, uses white.
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.ambient_light = ambient_light if ambient_light is not None else np.ones(3)
        
        # Core components
        self.scene: Optional[pyrender.Scene] = None
        self.camera: Optional[pyrender.Camera] = None
        self.camera_node: Optional[pyrender.Node] = None
        self.renderer: Optional[pyrender.OffscreenRenderer] = None
        
        # Node tracking
        self.nodes: Dict[str, pyrender.Node] = {}
        self.meshes: Dict[str, pyrender.Mesh] = {}
        
        # Light nodes
        self.light_nodes: Dict[str, pyrender.Node] = {}
        
        self.is_initialized = False
    
    def initialize(self,
                   camera_fov: float = np.pi / 3.0,
                   camera_pose: Optional[np.ndarray] = None,
                   znear: float = 0.01,
                   zfar: float = 10.0) -> bool:
        """
        Initialize the renderer, scene, camera, and lighting.

        Args:
            camera_fov: Camera field of view in radians (default: 60 degrees)
            camera_pose: 4x4 camera transformation matrix. If None, uses default position.
            znear: Near clip plane in meters (default 0.01).
            zfar: Far clip plane in meters (default 10.0).

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Create scene
            self.scene = pyrender.Scene(
                bg_color=self.background_color,
                ambient_light=self.ambient_light
            )

            # Create camera
            self.camera = pyrender.PerspectiveCamera(
                yfov=camera_fov,
                aspectRatio=self.width / self.height,
                znear=znear,
                zfar=zfar,
            )

            # Set camera pose (default: looking at origin from +Z, ~1 m back)
            if camera_pose is None:
                camera_pose = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
            
            self.camera_node = self.scene.add(self.camera, pose=camera_pose)
            
            # Add default lighting
            self._setup_default_lighting()
            
            # Create off-screen renderer
            self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize Generic3DRenderer: {e}")
            return False
    
    def _setup_default_lighting(self):
        """Set up realistic scene lighting for better surface detail visibility."""
        # Main directional light (like sun) - from upper front left
        main_light = pyrender.DirectionalLight(
            color=np.array([1.0, 0.95, 0.9]),  # Slightly warm white
            intensity=1.5  # Reduced intensity
        )
        # Position light from upper front left for good modeling
        main_light_pose = np.array([
            [0.7071, -0.5, 0.5, 0.0],
            [0.7071, 0.5, -0.5, 0.0],
            [0.0, 0.7071, 0.7071, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        main_light_node = self.scene.add(main_light, pose=main_light_pose)
        self.light_nodes['main_directional'] = main_light_node
        
        # Fill light - softer light from opposite side to reduce harsh shadows
        fill_light = pyrender.DirectionalLight(
            color=np.array([0.8, 0.9, 1.0]),  # Slightly cool white
            intensity=0.3  # Much softer fill light
        )
        fill_light_pose = np.array([
            [-0.7071, 0.5, -0.5, 0.0],
            [-0.7071, -0.5, 0.5, 0.0],
            [0.0, -0.7071, 0.7071, 0.7],
            [0.0, 0.0, 0.0, 1.0]
        ])
        fill_light_node = self.scene.add(fill_light, pose=fill_light_pose)
        self.light_nodes['fill_directional'] = fill_light_node
        
        # Key light - point light for highlights and surface detail
        key_light = pyrender.PointLight(
            color=np.array([1.0, 1.0, 1.0]), 
            intensity=2.0  # Much lower intensity
        )
        key_light_pose = np.array([
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.7],
            [0.0, 0.0, 1.0, 0.8],
            [0.0, 0.0, 0.0, 1.0]
        ])
        key_light_node = self.scene.add(key_light, pose=key_light_pose)
        self.light_nodes['key_point'] = key_light_node
    
    def add_node(self, 
                 name: str,
                 mesh: Union[trimesh.Trimesh, pyrender.Mesh],
                 pose: Optional[np.ndarray] = None,
                 material: Optional[pyrender.Material] = None) -> Optional[pyrender.Node]:
        """
        Add a mesh node to the scene.
        
        Args:
            name: Unique identifier for the node
            mesh: Trimesh or pyrender Mesh object
            pose: 4x4 transformation matrix. If None, uses identity.
            material: Optional material for the mesh. If None, uses default.
            
        Returns:
            pyrender.Node: The created scene node, or None if failed
        """
        if not self.is_initialized:
            print("Renderer not initialized. Call initialize() first.")
            return None
        
        # Remove existing node with same name
        if name in self.nodes:
            self.remove_node(name)
        
        try:
            # Convert trimesh to pyrender mesh if needed
            if isinstance(mesh, trimesh.Trimesh):
                if material is None:
                    # Create default material
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                        metallicFactor=0.2,
                        roughnessFactor=0.8
                    )
                pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            else:
                pr_mesh = mesh
            
            # Set default pose if not provided
            if pose is None:
                pose = np.eye(4)
            
            # Add to scene
            node = self.scene.add(pr_mesh, pose=pose)
            
            # Track the node and mesh
            self.nodes[name] = node
            self.meshes[name] = pr_mesh

            print(f"Added node '{name}' to scene")
            return node
            
        except Exception as e:
            print(f"Failed to add node '{name}': {e}")
            return None
    
    def set_pose(self, name: str, pose: np.ndarray) -> bool:
        """
        Update the pose of an existing node.
        
        Args:
            name: Name of the node to update
            pose: 4x4 transformation matrix
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name not in self.nodes:
            print(f"Node '{name}' not found in scene")
            return False
        
        try:
            self.scene.set_pose(self.nodes[name], pose)
            return True
        except Exception as e:
            print(f"Failed to set pose for '{name}': {e}")
            return False
    
    def get_pose(self, name: str) -> Optional[np.ndarray]:
        """
        Get the current pose of a node.
        
        Args:
            name: Name of the node
            
        Returns:
            np.ndarray: 4x4 transformation matrix, or None if node not found
        """
        if name not in self.nodes:
            print(f"Node '{name}' not found in scene")
            return None
        
        try:
            return self.scene.get_pose(self.nodes[name])
        except Exception as e:
            print(f"Failed to get pose for '{name}': {e}")
            return None
    
    def remove_node(self, name: str) -> bool:
        """
        Remove a node from the scene.
        
        Args:
            name: Name of the node to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name not in self.nodes:
            return True  # Already removed
        
        try:
            self.scene.remove_node(self.nodes[name])
            del self.nodes[name]
            del self.meshes[name]
            print(f"Removed node '{name}' from scene")
            return True
        except Exception as e:
            print(f"Failed to remove node '{name}': {e}")
            return False
    
    def set_camera_pose(self, pose: np.ndarray):
        """
        Update the camera pose.
        
        Args:
            pose: 4x4 camera transformation matrix
        """
        if self.camera_node is not None:
            self.scene.set_pose(self.camera_node, pose)
    
    def get_camera_pose(self) -> Optional[np.ndarray]:
        """
        Get the current camera pose.
        
        Returns:
            np.ndarray: 4x4 transformation matrix, or None if not initialized
        """
        if self.camera_node is not None:
            return self.scene.get_pose(self.camera_node)
        return None
    
    def render(self, flags: Optional[pyrender.RenderFlags] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render the scene and return color and depth buffers.
        
        Args:
            flags: Optional rendering flags for custom rendering modes
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (color_buffer, depth_buffer)
                - color_buffer: RGB image (height x width x 3), dtype uint8
                - depth_buffer: Depth map (height x width), dtype float32
        """
        if not self.is_initialized or self.renderer is None:
            print("Renderer not initialized")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8), \
                   np.zeros((self.height, self.width), dtype=np.float32)
        
        try:
            if flags is None:
                color, depth = self.renderer.render(self.scene)
            else:
                color, depth = self.renderer.render(self.scene, flags=flags)
            return color, depth
        except Exception as e:
            print(f"Render failed: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8), \
                   np.zeros((self.height, self.width), dtype=np.float32)
    
    def get_scene_info(self) -> Dict[str, Any]:
        """
        Get information about the current scene state.
        
        Returns:
            Dict containing scene information
        """
        return {
            'initialized': self.is_initialized,
            'width': self.width,
            'height': self.height,
            'num_nodes': len(self.nodes),
            'node_names': list(self.nodes.keys()),
            'num_lights': len(self.light_nodes),
            'camera_pose': self.get_camera_pose().tolist() if self.camera_node else None
        }
    
    def clear_scene(self):
        """Remove all mesh nodes from the scene (keeps camera and lights)."""
        node_names = list(self.nodes.keys())
        for name in node_names:
            self.remove_node(name)
        print("Scene cleared (camera and lights preserved)")
    
    def load_mesh_from_file(self,
                            filepath: str,
                            source_units: Union[str, float] = 'm',
                            extra_scale: float = 1.0,
                            center: bool = True,
                            recenter_mode: str = 'aabb') -> Optional[trimesh.Trimesh]:
        """
        Load a mesh from file (STL, OBJ, PLY, etc.) and convert it to meters.

        Args:
            filepath: Path to mesh file.
            source_units: Native unit of the mesh file. String in
                {'m','dm','cm','mm','in'} or a float meters-per-unit.
            extra_scale: Additional dimensionless scale applied after unit
                conversion (default 1.0). Use only for deliberate artistic
                overrides, not unit correction.
            center: Whether to translate the mesh so its chosen center lies
                at the origin.
            recenter_mode: 'aabb' (axis-aligned bounding-box midpoint,
                default) or 'mass' (density-weighted centroid). AABB is
                preferred for thin-shelled STLs where mass-based centroids
                can drift off the visible silhouette.

        Returns:
            trimesh.Trimesh: Loaded mesh with vertices in meters, or None
            if loading failed.
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None

        try:
            mesh = trimesh.load(filepath)

            if center:
                if recenter_mode == 'aabb':
                    mesh.vertices -= (mesh.bounds[0] + mesh.bounds[1]) / 2.0
                elif recenter_mode == 'mass':
                    mesh.vertices -= mesh.center_mass
                else:
                    raise ValueError(
                        f"Unknown recenter_mode '{recenter_mode}' "
                        "(expected 'aabb' or 'mass')"
                    )

            meters_per_unit = _resolve_meters_per_unit(source_units)
            total_scale = meters_per_unit * extra_scale
            if total_scale != 1.0:
                mesh.apply_scale(total_scale)

            print(
                f"Loaded mesh from {filepath} "
                f"({len(mesh.vertices)} vertices, "
                f"source_units={source_units}, "
                f"extents_m={mesh.extents})"
            )
            return mesh

        except Exception as e:
            print(f"Failed to load mesh from {filepath}: {e}")
            return None
    
    def destroy(self):
        """Clean up renderer resources."""
        try:
            if self.renderer is not None:
                self.renderer.delete()
                self.renderer = None
            
            # Clear references
            self.scene = None
            self.camera = None
            self.camera_node = None
            self.nodes.clear()
            self.meshes.clear()
            self.light_nodes.clear()
            
            self.is_initialized = False
            print("Generic3DRenderer destroyed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.destroy()


# Utility functions for common transformations

def create_translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Create a 4x4 translation matrix."""
    mat = np.eye(4)
    mat[:3, 3] = [x, y, z]
    return mat


def create_rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix around a principal axis.
    
    Args:
        axis: 'x', 'y', or 'z'
        angle: Rotation angle in radians
        
    Returns:
        4x4 rotation matrix
    """
    mat = np.eye(4)
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis.lower() == 'x':
        mat[:3, :3] = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis.lower() == 'y':
        mat[:3, :3] = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        mat[:3, :3] = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
    
    return mat


def create_scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    """Create a 4x4 scale matrix."""
    mat = np.eye(4)
    mat[0, 0] = sx
    mat[1, 1] = sy
    mat[2, 2] = sz
    return mat


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Convert quaternion to 4x4 transformation matrix.
    
    Args:
        qw, qx, qy, qz: Quaternion components (w, x, y, z)
        
    Returns:
        4x4 transformation matrix with rotation only
    """
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm == 0:
        return np.eye(4)
    
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Create rotation matrix
    mat = np.eye(4)
    mat[:3, :3] = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return mat
