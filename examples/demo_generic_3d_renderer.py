"""
Demo of Generic3DRenderer with various meshes and transformations.
Shows integration with AbstractImageWindow for display.
"""

import sys
import os
import platform

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import trimesh
from src.guidance.rendering.generic_3d_renderer import (
    Generic3DRenderer,
    create_translation_matrix,
    create_rotation_matrix,
    quaternion_to_matrix
)
from src.ui.image_window.window_factory import create_image_window
import time


def _create_window(title: str, width: int, height: int, **kwargs):
    """Create and initialize a platform-specific image window."""
    system = platform.system().lower()
    backend = 'opencv' if system == 'darwin' else None
    window = create_image_window(title=title, width=width, height=height, backend=backend, **kwargs)
    if not window.initialize():
        raise RuntimeError(f"Failed to initialize window '{title}'")
    return window


def _register_quit_handlers(window):
    """Register standard quit handlers and return mutable state."""
    state = {'quit': False}

    def request_quit():
        state['quit'] = True

    window.register_key_handler('q', request_quit)
    window.register_key_handler('Escape', request_quit)
    return state


def _unregister_quit_handlers(window):
    """Remove standard quit handlers."""
    window.unregister_key_handler('q')
    window.unregister_key_handler('Escape')


def demo_basic_rendering():
    """Basic rendering demo with a single mesh."""
    print("\n=== Basic Rendering Demo ===")
    
    # Create renderer
    renderer = Generic3DRenderer(width=512, height=512)
    if not renderer.initialize():
        print("❌ Failed to initialize renderer")
        renderer.destroy()
        return
    
    # Create a simple icosphere
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    renderer.add_node("sphere", mesh)
    
    # Render and display
    color, depth = renderer.render()
    
    # Display in window
    window = _create_window(title="3D Renderer - Basic", width=512, height=512)
    quit_state = _register_quit_handlers(window)
    window.display_frame(color)
    
    print("Press 'q' to continue to next demo...")
    while window.update() and not quit_state['quit']:
        time.sleep(0.01)
    
    _unregister_quit_handlers(window)
    window.destroy()
    renderer.destroy()


def demo_multiple_meshes():
    """Demo with multiple meshes at different positions."""
    print("\n=== Multiple Meshes Demo ===")
    
    renderer = Generic3DRenderer(width=640, height=480)
    if not renderer.initialize():
        print("❌ Failed to initialize renderer")
        renderer.destroy()
        return
    
    # Add sphere
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    sphere_pose = create_translation_matrix(-1.5, 0, 0)
    renderer.add_node("sphere", sphere, pose=sphere_pose)
    
    # Add cube
    cube = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    cube_pose = create_translation_matrix(0, 0, 0)
    renderer.add_node("cube", cube, pose=cube_pose)
    
    # Add cylinder
    cylinder = trimesh.creation.cylinder(radius=0.3, height=1.0)
    cylinder_pose = create_translation_matrix(1.5, 0, 0)
    renderer.add_node("cylinder", cylinder, pose=cylinder_pose)
    
    # Render
    color, depth = renderer.render()
    
    # Display
    window = _create_window(title="3D Renderer - Multiple Meshes", width=640, height=480)
    quit_state = _register_quit_handlers(window)
    window.display_frame(color)
    
    print("Press 'q' to continue to next demo...")
    while window.update() and not quit_state['quit']:
        time.sleep(0.01)
    
    _unregister_quit_handlers(window)
    window.destroy()
    renderer.destroy()


def demo_animation():
    """Animated rotation demo."""
    print("\n=== Animation Demo ===")
    
    renderer = Generic3DRenderer(width=512, height=512)
    if not renderer.initialize():
        print("❌ Failed to initialize renderer")
        renderer.destroy()
        return
    
    # Create a cube
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    renderer.add_node("cube", cube)
    
    # Display window
    window = _create_window(title="3D Renderer - Animation", width=512, height=512)
    quit_state = _register_quit_handlers(window)
    
    print("Animating rotation... Press 'q' to continue to next demo")
    
    angle = 0.0
    running = True
    
    while running and not quit_state['quit']:
        # Update pose
        rot_y = create_rotation_matrix('y', angle)
        rot_x = create_rotation_matrix('x', angle * 0.5)
        pose = rot_y @ rot_x
        renderer.set_pose("cube", pose)
        
        # Render and display
        color, depth = renderer.render()
        window.display_frame(color)
        if not window.update():
            break
        if quit_state['quit']:
            break
        
        # Update angle
        angle += 0.05
        time.sleep(0.016)  # ~60 FPS
    
    _unregister_quit_handlers(window)
    window.destroy()
    renderer.destroy()


def demo_quaternion_pose():
    """Demo using quaternion-based pose updates."""
    print("\n=== Quaternion Pose Demo ===")
    
    renderer = Generic3DRenderer(width=512, height=512)
    if not renderer.initialize():
        print("❌ Failed to initialize renderer")
        renderer.destroy()
        return
    
    # Create mesh
    mesh = trimesh.creation.capsule(radius=0.3, height=1.5)
    renderer.add_node("capsule", mesh)
    
    # Display window
    window = _create_window(title="3D Renderer - Quaternion", width=512, height=512)
    quit_state = _register_quit_handlers(window)
    
    print("Rotating using quaternions... Press 'q' to continue to next demo")
    
    angle = 0.0
    running = True
    
    while running and not quit_state['quit']:
        # Create quaternion rotation (around Y axis)
        qw = np.cos(angle / 2)
        qx = 0.0
        qy = np.sin(angle / 2)
        qz = 0.0
        
        # Convert to matrix
        pose = quaternion_to_matrix(qw, qx, qy, qz)
        renderer.set_pose("capsule", pose)
        
        # Render and display
        color, depth = renderer.render()
        window.display_frame(color)
        if not window.update():
            break
        if quit_state['quit']:
            break
        
        angle += 0.03
        time.sleep(0.016)
    
    _unregister_quit_handlers(window)
    window.destroy()
    renderer.destroy()


def demo_camera_movement():
    """Demo with moving camera."""
    print("\n=== Camera Movement Demo ===")
    
    renderer = Generic3DRenderer(width=640, height=480)
    if not renderer.initialize():
        print("❌ Failed to initialize renderer")
        renderer.destroy()
        return
    
    # Add some objects
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    renderer.add_node("sphere", sphere, pose=create_translation_matrix(0, 0, 0))
    
    # Add coordinate axes (simple colored sticks)
    # X-axis (red)
    x_axis = trimesh.creation.cylinder(radius=0.05, height=2.0)
    x_pose = create_rotation_matrix('z', -np.pi/2) @ create_translation_matrix(1, 0, 0)
    renderer.add_node("x_axis", x_axis, pose=x_pose)
    
    # Y-axis (green)
    y_axis = trimesh.creation.cylinder(radius=0.05, height=2.0)
    y_pose = create_translation_matrix(0, 1, 0)
    renderer.add_node("y_axis", y_axis, pose=y_pose)
    
    # Z-axis (blue)
    z_axis = trimesh.creation.cylinder(radius=0.05, height=2.0)
    z_pose = create_rotation_matrix('x', np.pi/2) @ create_translation_matrix(0, 0, 1)
    renderer.add_node("z_axis", z_axis, pose=z_pose)
    
    # Display window
    window = _create_window(title="3D Renderer - Camera Movement", width=640, height=480)
    quit_state = _register_quit_handlers(window)
    
    print("Moving camera in a circle... Press 'q' to exit")
    
    angle = 0.0
    running = True
    
    while running and not quit_state['quit']:
        # Camera orbits around the origin
        radius = 5.0
        cam_x = radius * np.cos(angle)
        cam_z = radius * np.sin(angle)
        cam_y = 2.0
        
        # Camera looks at origin
        camera_pos = np.array([cam_x, cam_y, cam_z])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        
        # Create view matrix (simple look-at)
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_corrected = np.cross(right, forward)
        
        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up_corrected
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = camera_pos
        
        renderer.set_camera_pose(camera_pose)
        
        # Render and display
        color, depth = renderer.render()
        window.display_frame(color)
        if not window.update():
            break
        if quit_state['quit']:
            break
        
        angle += 0.02
        time.sleep(0.016)
    
    _unregister_quit_handlers(window)
    window.destroy()
    renderer.destroy()


def demo_stl_loading():
    """Demo loading and displaying STL file (if available)."""
    print("\n=== STL Loading Demo ===")
    
    # Try to find the probe STL file
    stl_path = os.path.join(os.path.dirname(__file__), '..', 'research', 'probe.stl')
    
    if not os.path.exists(stl_path):
        print(f"⚠️  STL file not found at {stl_path}")
        print("Skipping STL loading demo")
        return
    
    renderer = Generic3DRenderer(width=512, height=512)
    if not renderer.initialize():
        print("❌ Failed to initialize renderer")
        renderer.destroy()
        return
    
    # Load mesh from STL
    mesh = renderer.load_mesh_from_file(stl_path, scale=0.01, center=True)
    
    if mesh is None:
        print("Failed to load STL file")
        renderer.destroy()
        return
    
    renderer.add_node("probe", mesh)
    
    # Display window
    window = _create_window(title="3D Renderer - STL Model", width=512, height=512)
    quit_state = _register_quit_handlers(window)
    
    print("Rotating STL model... Press 'q' to exit")
    
    angle = 0.0
    running = True
    
    while running and not quit_state['quit']:
        # Rotate the probe
        rot = create_rotation_matrix('y', angle)
        renderer.set_pose("probe", rot)
        
        # Render and display
        color, depth = renderer.render()
        window.display_frame(color)
        if not window.update():
            break
        if quit_state['quit']:
            break
        
        angle += 0.03
        time.sleep(0.016)
    
    _unregister_quit_handlers(window)
    window.destroy()
    renderer.destroy()


def main():
    """Run all demos."""
    print("=" * 50)
    print("Generic3DRenderer Demo")
    print("=" * 50)
    
    try:
        demo_basic_rendering()
        demo_multiple_meshes()
        demo_animation()
        demo_quaternion_pose()
        demo_camera_movement()
        demo_stl_loading()
        
        print("\n✅ All demos completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
