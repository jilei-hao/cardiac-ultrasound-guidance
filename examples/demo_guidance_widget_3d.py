"""
Example showing how to use Generic3DRenderer for a 3D guidance widget.
Demonstrates rendering probe mesh with orientation from IMU data.
"""

import sys
import os
import platform
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import trimesh
from src.ui.generic_3d_renderer import (
    Generic3DRenderer,
    create_translation_matrix,
    create_rotation_matrix,
    quaternion_to_matrix
)
from src.ui.image_window.window_factory import create_image_window
from src.data_access.simulator.frame_data_simulator import FrameDataSimulator
import time


def _select_backend() -> str | None:
    """Select window backend based on platform."""
    return 'opencv' if platform.system().lower() == 'darwin' else None


def _create_window(title: str, width: int, height: int, **kwargs):
    """Create and initialize an image window with platform-aware backend."""
    backend = _select_backend()
    window = create_image_window(title=title, width=width, height=height, backend=backend, **kwargs)
    if not window.initialize():
        raise RuntimeError(f"Failed to initialize window '{title}'")
    return window


def _unregister_keys(window, keys):
    for key in keys:
        window.unregister_key_handler(key)


def create_coordinate_axes(length: float = 1.0, radius: float = 0.02) -> dict:
    """
    Create coordinate axes as trimesh cylinder objects originating from center.
    X-axis (red) points along probe head direction (positive X)
    Y-axis (green) points sideways 
    Z-axis (blue) points up/down
    
    Returns:
        dict: {'x': trimesh, 'y': trimesh, 'z': trimesh}
    """
    axes = {}
    
    # X-axis (red) - cylinder extending from origin along +X
    x_shaft = trimesh.creation.cylinder(radius=radius, height=length)
    
    # Position shaft to extend from origin to +X
    x_shaft.apply_transform(create_rotation_matrix('y', np.pi/2))
    x_shaft.apply_transform(create_translation_matrix(length/2, 0, 0))
    
    axes['x'] = x_shaft
    
    # Y-axis (green) - cylinder extending from origin along +Y
    y_shaft = trimesh.creation.cylinder(radius=radius, height=length)
    
    # Position shaft to extend from origin to +Y
    y_shaft.apply_transform(create_rotation_matrix('x', -np.pi/2))
    y_shaft.apply_transform(create_translation_matrix(0, length/2, 0))
    
    axes['y'] = y_shaft
    
    # Z-axis (blue) - cylinder extending from origin along +Z
    z_shaft = trimesh.creation.cylinder(radius=radius, height=length)
    
    # Position shaft to extend from origin to +Z (already aligned)
    z_shaft.apply_transform(create_translation_matrix(0, 0, length/2))
    
    axes['z'] = z_shaft
    
    return axes


def simulate_imu_quaternions(num_samples: int = 100):
    """
    Simulate IMU quaternion data with smooth rotation.
    
    Returns:
        np.ndarray: (num_samples, 4) array of [qw, qx, qy, qz]
    """
    quaternions = []
    
    for i in range(num_samples):
        # Smooth rotation around Y axis
        angle = (i / num_samples) * 2 * np.pi
        qw = np.cos(angle / 2)
        qx = 0.1 * np.sin(angle / 2)  # Small X component
        qy = np.sin(angle / 2)
        qz = 0.05 * np.sin(angle / 2)  # Small Z component
        
        # Normalize
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        quaternions.append([qw/norm, qx/norm, qy/norm, qz/norm])
    
    return np.array(quaternions)


def demo_guidance_widget(data_folder: str = None, scale_factor: float = 1.0, wireframe: bool = False):
    """
    Demo of a 3D guidance widget similar to guidance_widget_renderer.py
    Shows probe orientation using real or simulated IMU data.
    
    Args:
        data_folder: Path to folder containing frame.imu.csv file. If None, uses simulated data.
        scale_factor: Scaling factor for probe and axes size (default: 1.0)
        wireframe: If True, render in wireframe mode (default: False)
    """
    print("\n=== 3D Guidance Widget Demo ===")

    renderer = Generic3DRenderer(
        width=640,
        height=480,
        background_color=(0.1, 0.1, 0.15, 1.0),
        ambient_light=np.array([0.2, 0.2, 0.2])  # Much lower ambient light for better contrast
    )
    if not renderer.initialize(camera_fov=np.pi / 4):
        print("❌ Failed to initialize renderer")
        renderer.destroy()
        return

    # Load transducer mesh from assets with scaling
    stl_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'transducer.stl')
    if os.path.exists(stl_path):
        print(f"Loading transducer STL file with scale factor {scale_factor}...")
        probe_mesh = renderer.load_mesh_from_file(stl_path, scale=0.01 * scale_factor, center=True)
        if probe_mesh is None:
            print("Failed to load transducer STL, using capsule instead")
            probe_mesh = trimesh.creation.capsule(radius=0.2 * scale_factor, height=2.0 * scale_factor)
    else:
        print(f"Transducer STL not found at {stl_path}, using capsule mesh")
        probe_mesh = trimesh.creation.capsule(radius=0.2 * scale_factor, height=2.0 * scale_factor)

    # Create coordinate axes that originate from probe center with scaling
    # Simple cylinder shafts - X (red) extends along probe head direction
    axes = create_coordinate_axes(length=1.2 * scale_factor, radius=0.02 * scale_factor)
    print(f"Created coordinate axes with scale factor {scale_factor} - X (red) extends to probe head, Y (green) sideways, Z (blue) up/down")

    import pyrender

    # Create realistic materials with better surface detail visibility
    # Wireframe is controlled by render flags instead of material properties
    materials = {
        'probe': pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.98, 0.98, 0.98, 1.0],  # White/off-white plastic color
            metallicFactor=0.1,  # No metallic properties for plastic appearance
            roughnessFactor=0.7  # Higher roughness for matte plastic finish
        ),
        'x': pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 0.2, 0.2, 1.0],  # Brighter red for better visibility
            metallicFactor=0.3, 
            roughnessFactor=0.6
        ),
        'y': pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.2, 1.0, 0.2, 1.0],  # Brighter green for better visibility
            metallicFactor=0.3, 
            roughnessFactor=0.6
        ),
        'z': pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.2, 0.2, 1.0, 1.0],  # Brighter blue for better visibility
            metallicFactor=0.3, 
            roughnessFactor=0.6
        ),
    }
    
    if wireframe:
        print("🔲 Using wireframe rendering mode (RenderFlags.ALL_WIREFRAME)")

    # Add probe and axes with appropriate materials
    renderer.add_node("probe", probe_mesh, material=materials['probe'])
    renderer.add_node("x_axis", pyrender.Mesh.from_trimesh(axes['x'], material=materials['x']))
    renderer.add_node("y_axis", pyrender.Mesh.from_trimesh(axes['y'], material=materials['y']))
    renderer.add_node("z_axis", pyrender.Mesh.from_trimesh(axes['z'], material=materials['z']))

    # Adjust camera distance based on scale factor
    camera_distance = 5.0 * max(1.0, scale_factor * 0.8)  # Adjust distance but not linearly
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [3.0 * camera_distance/5.0, 2.0 * camera_distance/5.0, camera_distance]

    target = np.array([0.0, 0.0, 0.0])
    camera_pos = camera_pose[:3, 3]
    up = np.array([0.0, 1.0, 0.0])

    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_corrected = np.cross(right, forward)

    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up_corrected
    camera_pose[:3, 2] = -forward
    renderer.set_camera_pose(camera_pose)

    # Initialize data source
    simulator = None
    quaternions = None
    use_simulator = data_folder is not None
    
    if use_simulator:
        print(f"Loading IMU data from: {data_folder}")
        simulator = FrameDataSimulator(frames_folder=data_folder)
        try:
            simulator.start()
        except Exception as e:
            print(f"❌ Failed to start simulator: {e}")
            print("Falling back to simulated quaternions...")
            use_simulator = False
            quaternions = simulate_imu_quaternions(num_samples=200)
    else:
        print("Generating simulated IMU quaternions...")
        quaternions = simulate_imu_quaternions(num_samples=200)

    window = None
    registered_keys: list[str] = []

    try:
        window = _create_window(title="3D Guidance Widget", width=640, height=480)

        print("\n=== Controls ===")
        print("Press 'q' or Escape to quit")
        print("Press 'r' to reset rotation")
        print("Press space to pause/resume")
        if use_simulator:
            print(f"\nStreaming real IMU data from: {data_folder}")
        else:
            print("\nRendering probe with simulated IMU data...")

        state = {'quit': False, 'reset': False, 'toggle_pause': False}

        def request_quit():
            state['quit'] = True

        def request_reset():
            state['reset'] = True

        def toggle_pause():
            state['toggle_pause'] = True

        window.register_key_handler('q', request_quit)
        window.register_key_handler('Escape', request_quit)
        window.register_key_handler('r', request_reset)
        window.register_key_handler('space', toggle_pause)
        registered_keys = ['q', 'Escape', 'r', 'space']

        frame_idx = 0
        paused = False

        while window.update() and not state['quit']:
            if state['reset']:
                if use_simulator and simulator:
                    simulator.reset()
                else:
                    frame_idx = 0
                state['reset'] = False
                print("Reset rotation")

            if state['toggle_pause']:
                paused = not paused
                state['toggle_pause'] = False
                print("Paused" if paused else "Resumed")

            if not paused:
                rotation_matrix = None
                if use_simulator and simulator:
                    # Get frame data from simulator
                    frame_data = simulator.get_frame()
                    if frame_data.imu is not None:
                        # Extract quaternion from IMU data
                        imu = frame_data.imu
                        qw, qx, qy, qz = imu.qw, imu.qx, imu.qy, imu.qz
                        rotation_matrix = quaternion_to_matrix(qw, qx, qy, qz)
                else:
                    # Use simulated quaternions
                    if frame_idx > 0 and frame_idx % len(quaternions) == 0:
                        print("🔄 Reached end of simulated sequence, restarting from beginning")
                    qw, qx, qy, qz = quaternions[frame_idx % len(quaternions)]
                    rotation_matrix = quaternion_to_matrix(qw, qx, qy, qz)
                    frame_idx += 1
                
                # Apply the same rotation to both probe and axes
                if rotation_matrix is not None:
                    renderer.set_pose("probe", rotation_matrix)
                    renderer.set_pose("x_axis", rotation_matrix)
                    renderer.set_pose("y_axis", rotation_matrix)
                    renderer.set_pose("z_axis", rotation_matrix)

            # Render with appropriate flags
            if wireframe:
                import pyrender
                color, _ = renderer.render(flags=pyrender.RenderFlags.ALL_WIREFRAME)
            else:
                # Use standard rendering for now (shadows have NumPy compatibility issues)
                color, _ = renderer.render()
            window.display_frame(color)

            time.sleep(0.03)

        print("✅ Demo completed")

    finally:
        if simulator:
            simulator.stop()
        if window is not None:
            _unregister_keys(window, registered_keys)
            window.destroy()
        renderer.destroy()


def demo_rotation_amplification():
    """Demo showing rotation amplification similar to the original widget."""
    # print("\n=== Rotation Amplification Demo ===")

    # renderer = Generic3DRenderer(width=640, height=480)
    # if not renderer.initialize():
    #     print("❌ Failed to initialize renderer")
    #     renderer.destroy()
    #     return

    # window = None
    # registered_keys: list[str] = []

    # try:
    #     # Create arrow mesh to visualize orientation
    #     arrow_tip = trimesh.creation.cone(radius=0.2, height=0.5)
    #     shaft = trimesh.creation.cylinder(radius=0.05, height=1.0)
    #     shaft.apply_transform(create_translation_matrix(0, 0.5, 0))
    #     arrow_tip.apply_transform(create_translation_matrix(0, 1.0, 0))
    #     arrow_mesh = trimesh.util.concatenate([shaft, arrow_tip])
    #     renderer.add_node("arrow", arrow_mesh)

    #     # Add reference axes with colors
    #     axes = create_coordinate_axes(length=2.0, radius=0.03)
    #     import pyrender
    #     materials = {
    #         'x': pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0, 0.0, 0.0, 1.0], metallicFactor=0.1, roughnessFactor=0.9),
    #         'y': pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 1.0, 0.0, 1.0], metallicFactor=0.1, roughnessFactor=0.9),
    #         'z': pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 1.0, 1.0], metallicFactor=0.1, roughnessFactor=0.9),
    #     }
    #     renderer.add_node("x_axis", pyrender.Mesh.from_trimesh(axes['x'], material=materials['x']))
    #     renderer.add_node("y_axis", pyrender.Mesh.from_trimesh(axes['y'], material=materials['y']))
    #     renderer.add_node("z_axis", pyrender.Mesh.from_trimesh(axes['z'], material=materials['z']))

    #     window = _create_window(title="Rotation Amplification", width=640, height=480)

    #     state = {'quit': False}

    #     def request_quit():
    #         state['quit'] = True

    #     window.register_key_handler('q', request_quit)
    #     window.register_key_handler('Escape', request_quit)
    #     registered_keys = ['q', 'Escape']

    #     angle = 0.0
    #     amplification_factor = 3.0

    #     while window.update() and not state['quit']:
    #         small_angle = 0.1 * np.sin(angle)
    #         amplified_angle = small_angle * amplification_factor

    #         pose = create_rotation_matrix('x', amplified_angle) @ create_rotation_matrix('y', angle * 0.5)
    #         renderer.set_pose("arrow", pose)

    #         color, _ = renderer.render()
    #         window.display_frame(color)

    #         angle += 0.02
    #         time.sleep(0.016)

    #     print("✅ Rotation amplification demo completed")

    # finally:
    #     if window is not None:
    #         if registered_keys:
    #             _unregister_keys(window, registered_keys)
    #         window.destroy()
    #     renderer.destroy()


def main():
    """Run guidance widget examples."""
    parser = argparse.ArgumentParser(description='3D Guidance Widget Demo')
    parser.add_argument('--data-folder', '-d', type=str, 
                       help='Path to folder containing frame.imu.csv file for real IMU data')
    parser.add_argument('--skip-amplification', action='store_true',
                       help='Skip the rotation amplification demo')
    parser.add_argument('--scale', '-s', type=float, default=2.0,
                       help='Scale factor for probe and axes size (default: 2.0, try 2.0-5.0 for larger)')
    parser.add_argument('--wireframe', '-w', action='store_true',
                       help='Render in wireframe mode (shows mesh structure)')

    args = parser.parse_args()
    
    print("=" * 50)
    print("3D Guidance Widget Examples")
    print("Using Generic3DRenderer")
    print("=" * 50)
    
    if args.data_folder:
        if not os.path.exists(args.data_folder):
            print(f"❌ Data folder not found: {args.data_folder}")
            return
        
        imu_file = os.path.join(args.data_folder, 'frame.imu.csv')
        if not os.path.exists(imu_file):
            print(f"❌ IMU file not found: {imu_file}")
            return
        
        print(f"📊 Using real IMU data from: {args.data_folder}")
    else:
        print("🔄 Using simulated IMU data")
    
    print(f"📏 Scale factor: {args.scale}x (probe and axes will be {'larger' if args.scale > 1 else 'smaller' if args.scale < 1 else 'normal size'})")
    if args.wireframe:
        print("🔲 Wireframe mode enabled")
    
    try:
        demo_guidance_widget(data_folder=args.data_folder, scale_factor=args.scale, wireframe=args.wireframe)
        
        if not args.skip_amplification:
            demo_rotation_amplification()
        
        print("\n✅ All examples completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
