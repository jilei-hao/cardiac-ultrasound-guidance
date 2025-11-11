#!/usr/bin/env python3
"""
Unified test for the guidance widget system with support for both 2D and 3D renderers.
This test allows switching between renderer types via command line options.

RECOMMENDED: Use conda vigil environment for testing:
    conda run -n vigil python tests/test_guidance.py [options...]
    
Or use the provided script:
    ./run_guidance_tests.sh simple
    ./run_guidance_tests.sh full --renderer 3d
"""

import sys
import os
import time
import argparse
import numpy as np
from dotenv import load_dotenv
import cv2
from PIL import Image

load_dotenv()  # Load environment variables from .env file

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Unified imports using the factory pattern
from inference.fourch_landmark.fourch_landmark_pipeline import run_fourch_core_prediction
from ui.image_window import create_dual_window_setup
from data_access.simulator import FrameDataSimulator
from inference.fourch_landmark.fourch_landmark_pipeline import configure_fourch_landmark_pipeline, run_fourch_core_prediction
from inference.rotation.rotation_predictor import create_rotation_predictor
from inference import inference_helper as ihp
from ui.guidance_widget import create_guidance_widget_renderer  # Unified factory
from inference.pose_classification import pose_classification
from inference.pose_classification.pose_classification import PoseClassifierResultPooler
from ui.fourch_landmark_renderer import FourChamberLandmarkRenderer
from common.add_clarius_lib_path import setup_clarius_environment
import ui.image_helpers as ih

# Conditionally import ClariusStreamer
try:
    # Setup Clarius environment first
    clarius_available, clarius_error = setup_clarius_environment()
    if clarius_available:
        import data_access.clarius.clarius_streamer as ClariusStreamer
    else:
        print(f"Warning: Clarius not available - {clarius_error}")
        ClariusStreamer = None
except Exception as e:
    print(f"Warning: Failed to setup Clarius environment - {e}")
    ClariusStreamer = None

class InferenceRunner:
    def __init__(self):
        self.__device = ihp.get_device()
        self.__landmark_pipeline = configure_fourch_landmark_pipeline(
            weight_path=os.getenv("4CH_WEIGHTS"),
            device=self.__device,
            landmarks_per_chamber={
                50: 12,   # Left Ventricle (LV)
                100: 12,  # Right Ventricle (RV)
                150: 12,  # Left Atrium (LA)
                200: 12,  # Right Atrium (RA)
            }
        )
        self.__rotation_predictor = create_rotation_predictor(
            config_path=os.getenv("ROTATION_PREDICTION_CONFIG"),
            checkpoint_path=os.getenv("ROTATION_PREDICTION_WEIGHTS"),
            device=self.__device
        )

        _model, _preprocess = pose_classification.load_pretrained_model(self.__device, weight_path=os.getenv("POSE_EST_IMG_WEIGHTS"))
        self.__pose_classifier_model = _model
        self.__pose_classifier_preprocess = _preprocess
        self.__pose_classifier_result_pooler = PoseClassifierResultPooler(pool_size=15, min_size=1)

    def run_inference(self, frame):
        # Run inference on the given frame using the landmark pipeline
        cla, landmarks_tensor = run_fourch_core_prediction(self.__landmark_pipeline, frame)
        rotation_prediction = self.__rotation_predictor.predict_rotation(frame)
        pose_classification_prediction = pose_classification.run_pose_classifier(
            self.__pose_classifier_model,
            self.__device,
            frame,
            landmarks_tensor,
            self.__pose_classifier_preprocess
        )
        self.__pose_classifier_result_pooler.add_result(pose_classification_prediction)
        pose_classification_prediction = self.__pose_classifier_result_pooler.get_pooled_result()

        return cla, landmarks_tensor, rotation_prediction, pose_classification_prediction


def test_guidance_unified(source, frames_folder=None, clarius_ip=None, clarius_port=None, renderer_type='auto', renderer_args=None):
    """Test unified guidance functionality with configurable renderer type and data source."""
    print(f"=== Guidance Test ({renderer_type.upper()}) ===")
    print(f"Source: {source}")
    if source == 'frames':
        print(f"Frames folder: {frames_folder}")
    elif source == 'clarius':
        print(f"Clarius IP: {clarius_ip}, Port: {clarius_port}")

    # Create dual window setup
    main_window, guidance_window = create_dual_window_setup(
        main_title="Main View",
        guidance_title=f"Guidance View ({renderer_type.upper()})",
        width=512,
        height=512,
        x_pos=100,
        y_pos=100
    )
    
    # Set up key handlers for both windows
    renderer_switch_mode = False
    current_renderer_type = renderer_type
    
    def handle_quit():
        print("Quit key pressed!")
        main_window.destroy()
        guidance_window.destroy()
    
    def handle_switch_2d():
        nonlocal renderer_switch_mode, current_renderer_type
        print("🔄 Switching to 2D renderer...")
        renderer_switch_mode = True
        current_renderer_type = '2d'
    
    def handle_switch_3d():
        nonlocal renderer_switch_mode, current_renderer_type
        print("🔄 Switching to 3D renderer...")
        renderer_switch_mode = True
        current_renderer_type = '3d'
    
    def handle_switch_auto():
        nonlocal renderer_switch_mode, current_renderer_type
        print("🔄 Switching to auto renderer...")
        renderer_switch_mode = True
        current_renderer_type = 'auto'
    
    # Register key handlers
    main_window.register_key_handler('q', handle_quit)
    guidance_window.register_key_handler('q', handle_quit)
    main_window.register_key_handler('2', handle_switch_2d)
    guidance_window.register_key_handler('2', handle_switch_2d)
    main_window.register_key_handler('3', handle_switch_3d)
    guidance_window.register_key_handler('3', handle_switch_3d)
    main_window.register_key_handler('a', handle_switch_auto)
    guidance_window.register_key_handler('a', handle_switch_auto)
    
    # Initialize both windows
    if not main_window.initialize():
        print("❌ Failed to initialize main window")
        return
    
    if not guidance_window.initialize():
        print("❌ Failed to initialize guidance window")
        main_window.destroy()
        return
    
    print("✅ Both windows initialized successfully")
    print("Main window info:", main_window.get_display_info())
    print("Guidance window info:", guidance_window.get_display_info())
    print("\n🎮 Controls:")
    print("  'q' - Quit")
    print("  '2' - Switch to 2D renderer")
    print("  '3' - Switch to 3D renderer")
    print("  'a' - Switch to auto renderer")
    
    # Initialize data source
    simulator = None
    if source == 'frames':
        # Create simulator with the provided frame folder
        simulator = FrameDataSimulator(frames_folder=frames_folder)
    elif source == 'clarius':
        # Check if Clarius is available before trying to use it
        if not clarius_available:
            print("❌ Clarius SDK is not available. Cannot use Clarius streaming.")
            main_window.destroy()
            guidance_window.destroy()
            return
            
        # Initialize Clarius streamer
        print(f"🔌 Connecting to Clarius probe at {clarius_ip}:{clarius_port}...")
        try:
            ClariusStreamer.configure(clarius_ip, clarius_port, "research")
            ClariusStreamer.set_print_debug_info(False)
            ClariusStreamer.start()
            print("✅ Connected to Clarius probe.")
        except Exception as e:
            print(f"❌ Failed to connect to Clarius probe: {e}")
            main_window.destroy()
            guidance_window.destroy()
            return

    inference_runner = InferenceRunner()

    # Create initial renderer
    def create_renderer(rtype, args=None):
        """Create renderer with specified type and arguments."""
        default_args = {
            'probe_path': os.getenv("PROBE_MESH_PATH"),
            'renderer_type': rtype,
            'background_color': (20, 20, 40),
            'render_mode': 'fullscreen',
            'window_size': (400, 400),
            'fallback_to_2d': True
        }
        
        # Renderer-specific configurations
        if rtype in ['2d', 'auto']:
            default_args.update({
                'predicted_axis_opacity': 0.6,
                'axis_thickness': 3,
                'axis_labels': ('x', 'y', 'z'),
                'lod_level': 0.1,
                'probe_scale': 0.7,
                'rotation_amplification': 5.0
            })
        
        if rtype in ['3d', 'auto']:
            default_args.update({
                'scale_factor': 2.0,
                'wireframe': False
            })
        
        # Override with custom arguments if provided
        if args:
            default_args.update(args)
        
        try:
            renderer = create_guidance_widget_renderer(**default_args)
            print(f"✅ Created {type(renderer).__name__}")
            return renderer
        except Exception as e:
            print(f"❌ Failed to create {rtype} renderer: {e}")
            return None

    probe_renderer = create_renderer(current_renderer_type, renderer_args)
    if probe_renderer is None:
        print("❌ Failed to create initial renderer")
        return

    landmark_renderer = FourChamberLandmarkRenderer(
        landmark_radius=8,
        min_opacity=0.6,
        max_opacity=1.0,
        uncertainty_threshold=6.0,
        reference_size=None,  # Landmarks are already in crop_frame coordinates
        color_legend_png_path=os.getenv("LANDMARK_COLOR_LEGEND_PNG_PATH")
    )
    
    try:
        if simulator:
            simulator.start()
        
        frame_count = 0
        print("\nStarting display loop...")
        
        while main_window.is_window_active() and guidance_window.is_window_active():
            # Check for renderer switch
            if renderer_switch_mode:
                print(f"🔄 Switching from {type(probe_renderer).__name__} to {current_renderer_type}...")
                old_renderer = probe_renderer
                probe_renderer = create_renderer(current_renderer_type, renderer_args)
                
                if probe_renderer is None:
                    print(f"❌ Failed to switch to {current_renderer_type}, keeping current renderer")
                    probe_renderer = old_renderer
                else:
                    print(f"✅ Successfully switched to {type(probe_renderer).__name__}")
                    old_renderer.destroy()
                    guidance_window.set_title(f"Guidance View ({current_renderer_type.upper()})")
                
                renderer_switch_mode = False
            
            # Get frame from data source
            if source == 'frames':
                frame_data = simulator.get_frame()
                if frame_data.image is None:
                    # End of sequence, reset
                    simulator.reset()
                    continue
            elif source == 'clarius':
                if not clarius_available:
                    print("❌ Clarius SDK not available")
                    break
                frame_data = ClariusStreamer.get_frame()
                if frame_data is None:
                    continue

            # Extract IMU data
            if frame_data.imu is not None:
                imu = frame_data.imu
                qw, qx, qy, qz = imu.qw, imu.qx, imu.qy, imu.qz
            else:
                # Default quaternion values (no rotation)
                qw, qx, qy, qz = 1.0, 1.0, 0.0, 0.0
            
            # Convert image to numpy array
            if source == 'frames':
                # Convert PIL image to numpy array (RGB format)
                orig_frame = np.array(frame_data.image)
                
                # Handle different image formats properly
                if len(orig_frame.shape) == 3 and orig_frame.shape[2] == 3:
                    # RGB format
                    orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
                elif len(orig_frame.shape) == 3 and orig_frame.shape[2] == 4:
                    # RGBA format
                    orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2GRAY)
                    orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2RGB)  # Convert to RGB for display
                elif len(orig_frame.shape) == 2:
                    # Already grayscale
                    orig_gray = orig_frame
                    orig_frame = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB)  # Convert to RGB for display
                else:
                    print(f"🛑 Unsupported image format: {orig_frame.shape}")
                    continue
            elif source == 'clarius':
                # Clarius frame is already numpy array in BGRA format
                orig_frame = np.array(frame_data.image)
                orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGRA2GRAY)
                orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGRA2RGB)  # Convert to RGB for display

            cla, landmarks_tensor, rotation_prediction, pose_classification_prediction_rgb = inference_runner.run_inference(orig_gray)

            # ---------------------------------------
            # Render ultrasound window
            # ---------------------------------------
            result_frame = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB)
            result_frame = landmark_renderer.overlay_landmarks(
                result_frame,
                landmarks_tensor,
                reference_size=None
            )

            result_frame = ih.add_rgb_border_to_result(result_frame, pose_classification_prediction_rgb)

            # --------------------------------------
            # Render guidance window (unified interface)
            # --------------------------------------
            try:
                # Use unified interface - works for both 2D and 3D
                if hasattr(probe_renderer, 'render_widget') and len(probe_renderer.render_widget.__code__.co_varnames) > 5:
                    # 2D renderer signature with background image and rotation matrix
                    widget_display = probe_renderer.render_widget(
                        np.zeros_like(orig_gray),
                        qw, qx, qy, qz,
                        predicted_rotation_matrix=rotation_prediction
                    )
                else:
                    # 3D renderer signature with quaternion only
                    widget_display = probe_renderer.render_widget(qw=qw, qx=qx, qy=qy, qz=qz)

                # Convert widget_display to numpy array
                if isinstance(widget_display, np.ndarray):
                    guidance_img = widget_display
                else:
                    # If it's a PIL Image, convert to numpy array
                    guidance_img = np.array(widget_display)

                # Ensure the image has the right format (RGB for display)
                if len(guidance_img.shape) == 3 and guidance_img.shape[2] == 4:
                    # Convert RGBA to RGB
                    guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_RGBA2RGB)
                elif len(guidance_img.shape) == 2:
                    # Convert grayscale to RGB
                    guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_GRAY2RGB)
                    
            except Exception as e:
                print(f"⚠️  Render error: {e}")
                # Create a fallback error display
                guidance_img = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(guidance_img, "Render Error", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # ==========================================
            main_frame = result_frame.copy()
            guidance_frame = guidance_img.copy()

            # Display frames in both windows
            if not main_window.display_frame_rgb(main_frame):
                print("❌ Failed to display main frame")
                break

            if not guidance_window.display_frame_rgb(guidance_frame):
                print("❌ Failed to display guidance frame")
                break
            
            # Update both windows
            if not main_window.update() or not guidance_window.update():
                print("🛑 One or both windows closed")
                break
            
            frame_count += 1
            
            # Limit to ~10 FPS for testing
            time.sleep(0.1)
            
            # Show progress every 50 frames
            if frame_count % 50 == 0:
                renderer_name = type(probe_renderer).__name__
                print(f"📊 Frame {frame_count} - Using {renderer_name}")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if simulator:
            simulator.stop()
        if source == 'clarius' and clarius_available:
            try:
                ClariusStreamer.stop()
                print("✅ Clarius connection closed.")
            except Exception as e:
                print(f"⚠️ Error closing Clarius connection: {e}")
        if probe_renderer:
            probe_renderer.destroy()
        main_window.destroy()
        guidance_window.destroy()
        print("✅ Test completed")


def main():
    """Run unified guidance tests with command line options."""
    parser = argparse.ArgumentParser(
        description="Unified Guidance Widget Test - supports both 2D and 3D renderers with frames or Clarius streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using frames folder
  python test_guidance.py frames /path/to/frames --renderer auto
  python test_guidance.py frames /path/to/frames --renderer 3d --scale 2.5 --wireframe
  
  # Using Clarius streaming
  python test_guidance.py clarius --clarius 192.168.1.1 5828 --renderer 2d
  python test_guidance.py clarius --clarius 10.0.0.1 5828 --renderer auto
  
During execution:
  Press '2' to switch to 2D renderer
  Press '3' to switch to 3D renderer  
  Press 'a' to switch to auto renderer
  Press 'q' to quit
        """
    )
    
    parser.add_argument('source', 
                       choices=['frames', 'clarius'],
                       help='Data source: frames folder or Clarius streaming')
    
    parser.add_argument('frames_folder', 
                       nargs='?',
                       help='Path to the frame_imu folder containing test data (required for frames source)')
    
    parser.add_argument('--clarius', '-c',
                       nargs=2,
                       metavar=('IP', 'PORT'),
                       help='Clarius probe IP address and port (required for clarius source)')
    
    parser.add_argument('--renderer', '-r', 
                       choices=['2d', '3d', 'auto'], 
                       default='auto',
                       help='Renderer type to use (default: auto)')
    
    # 2D-specific arguments
    parser.add_argument('--axis-thickness', type=int, default=3,
                       help='Thickness of coordinate axes (2D renderer)')
    
    parser.add_argument('--probe-scale', type=float, default=0.7,
                       help='Probe scale relative to axes (2D renderer)')
    
    parser.add_argument('--rotation-amplification', type=float, default=5.0,
                       help='Rotation amplification factor (2D renderer)')
    
    # 3D-specific arguments  
    parser.add_argument('--scale', type=float, default=2.0,
                       help='Scale factor for 3D objects (3D renderer)')
    
    parser.add_argument('--wireframe', action='store_true',
                       help='Enable wireframe mode (3D renderer)')
    
    # Common arguments
    parser.add_argument('--window-size', type=int, nargs=2, default=[400, 400],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Window size for guidance renderer')
    
    parser.add_argument('--background-color', type=int, nargs=3, default=[20, 20, 40],
                       metavar=('R', 'G', 'B'),
                       help='Background color RGB values')

    args = parser.parse_args()
    
    # Validate arguments based on source
    if args.source == 'frames':
        if not args.frames_folder:
            parser.error("frames_folder is required when using 'frames' source")
        if not os.path.exists(args.frames_folder):
            parser.error(f"Frames folder '{args.frames_folder}' does not exist")
        clarius_ip, clarius_port = None, None
    elif args.source == 'clarius':
        if not clarius_available:
            parser.error("Clarius SDK is not available. Cannot use 'clarius' source. Please install the Clarius SDK or use 'frames' source instead.")
        if not args.clarius:
            parser.error("--clarius IP PORT is required when using 'clarius' source")
        clarius_ip, clarius_port = args.clarius[0], int(args.clarius[1])
        args.frames_folder = None
    
    print("Unified Guidance Widget Test")
    print("=" * 50)
    print(f"Source: {args.source}")
    if args.source == 'frames':
        print(f"Frames folder: {args.frames_folder}")
    elif args.source == 'clarius':
        print(f"Clarius IP: {clarius_ip}, Port: {clarius_port}")
    print(f"Renderer type: {args.renderer}")
    print(f"Window size: {args.window_size}")
    print(f"Background: {args.background_color}")
    
    if args.renderer in ['2d', 'auto']:
        print(f"2D Settings:")
        print(f"  Axis thickness: {args.axis_thickness}")
        print(f"  Probe scale: {args.probe_scale}")
        print(f"  Rotation amplification: {args.rotation_amplification}")
    
    if args.renderer in ['3d', 'auto']:
        print(f"3D Settings:")
        print(f"  Scale factor: {args.scale}")
        print(f"  Wireframe: {args.wireframe}")
    
    print("=" * 50)
    
    # Construct renderer arguments from command line
    renderer_args = {
        'window_size': tuple(args.window_size),
        'background_color': tuple(args.background_color),
    }
    
    # Add renderer-specific arguments
    if args.renderer in ['2d', 'auto']:
        renderer_args.update({
            'axis_thickness': args.axis_thickness,
            'probe_scale': args.probe_scale,
            'rotation_amplification': args.rotation_amplification,
        })
    
    if args.renderer in ['3d', 'auto']:
        renderer_args.update({
            'scale_factor': args.scale,
            'wireframe': args.wireframe,
        })
    
    try:
        # Run the unified test
        test_guidance_unified(
            source=args.source,
            frames_folder=args.frames_folder,
            clarius_ip=clarius_ip,
            clarius_port=clarius_port,
            renderer_type=args.renderer,
            renderer_args=renderer_args
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Unified guidance test completed!")


if __name__ == "__main__":
    main()
