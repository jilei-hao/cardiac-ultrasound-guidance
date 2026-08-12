#!/usr/bin/env python3
"""
Live test for the 3D guidance widget renderer.

RECOMMENDED: Use conda vigil environment for testing:
    conda run -n vigil python tests/test_guidance.py [options...]
"""

import sys
import os
import time
import argparse
import numpy as np
from dotenv import load_dotenv
import cv2

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.image_window import create_dual_window_setup
from data_access.simulator import FrameDataSimulator
from guidance.rendering.guidance_widget import GuidanceWidgetRenderer3D
from guidance.rendering.fourch_landmark_renderer import FourChamberLandmarkRenderer
import guidance.rendering.image_helpers as ih
from guidance.inference.guidance_pipeline import GuidancePipeline, GuidancePipelineConfig

try:
    import data_access.clarius.clarius_streamer as ClariusStreamer
    clarius_available = True
    clarius_error = None
except Exception as e:
    print(f"Warning: Clarius not available - {e}")
    ClariusStreamer = None
    clarius_available = False
    clarius_error = str(e)


def run_guidance(source, frames_folder=None, clarius_ip=None, clarius_port=None,
                 window_size=(400, 400), background_color=(0, 0, 0, 1),
                 scale_factor=2.0, wireframe=False):
    """Run the live guidance test against the requested data source."""
    print("=== Guidance Test (3D) ===")
    print(f"Source: {source}")
    if source == 'frames':
        print(f"Frames folder: {frames_folder}")
    elif source == 'clarius':
        print(f"Clarius IP: {clarius_ip}, Port: {clarius_port}")

    guidance_config = GuidancePipelineConfig(
        fourch_landmark_weights=os.getenv("4CH_WEIGHTS"),
        rotation_prediction_config=os.getenv("ROTATION_PREDICTION_CONFIG"),
        rotation_prediction_weights=os.getenv("ROTATION_PREDICTION_WEIGHTS"),
        pose_classification_weights=os.getenv("POSE_EST_IMG_WEIGHTS"),
        lvef_weights=os.getenv("LVEF_WEIGHTS"),
        landmarks_per_chamber={50: 24, 100: 12, 150: 12, 200: 12},
    )
    guidance_pipeline = GuidancePipeline(guidance_config)

    main_window, guidance_window = create_dual_window_setup(
        main_title="Main View",
        guidance_title="Guidance View (3D)",
        width=512,
        height=512,
        x_pos=100,
        y_pos=100,
    )

    def handle_quit():
        print("Quit key pressed!")
        main_window.destroy()
        guidance_window.destroy()

    main_window.register_key_handler('q', handle_quit)
    guidance_window.register_key_handler('q', handle_quit)

    if not main_window.initialize():
        print("❌ Failed to initialize main window")
        return
    if not guidance_window.initialize():
        print("❌ Failed to initialize guidance window")
        main_window.destroy()
        return

    print("✅ Both windows initialized successfully")
    print("\n🎮 Controls:")
    print("  'q' - Quit")

    simulator = None
    if source == 'frames':
        simulator = FrameDataSimulator(frames_folder=frames_folder)
    elif source == 'clarius':
        if not clarius_available:
            print("❌ Clarius SDK is not available. Cannot use Clarius streaming.")
            main_window.destroy()
            guidance_window.destroy()
            return
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

    probe_renderer = GuidanceWidgetRenderer3D(
        probe_path=os.getenv("PROBE_MESH_PATH"),
        background_color=background_color,
        render_mode='fullscreen',
        window_size=window_size,
        legend_image_path=os.getenv("GUIDANCE_WIDGET_LEGEND_PNG_PATH"),
        wireframe=wireframe,
    )
    print(f"✅ Created {type(probe_renderer).__name__}")

    landmark_renderer = FourChamberLandmarkRenderer(
        landmark_radius=8,
        min_opacity=0.6,
        max_opacity=1.0,
        uncertainty_threshold=6.0,
        reference_size=None,
        color_legend_png_path=os.getenv("LANDMARK_COLOR_LEGEND_PNG_PATH"),
    )

    try:
        if simulator:
            simulator.start()

        frame_count = 0
        latest_lvef = None
        print("\nStarting display loop...")

        while main_window.is_window_active() and guidance_window.is_window_active():
            if source == 'frames':
                frame_data = simulator.get_frame()
                if frame_data.image is None:
                    simulator.reset()
                    continue
            elif source == 'clarius':
                if not clarius_available:
                    print("❌ Clarius SDK not available")
                    break
                frame_data = ClariusStreamer.get_frame()
                if frame_data is None:
                    continue

            if frame_data.imu is not None:
                imu = frame_data.imu
                qw, qx, qy, qz = imu.qw, imu.qx, imu.qy, imu.qz
            else:
                qw, qx, qy, qz = 1.0, 1.0, 0.0, 0.0

            if source == 'frames':
                orig_frame = np.array(frame_data.image)
                if len(orig_frame.shape) == 3 and orig_frame.shape[2] == 3:
                    orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
                elif len(orig_frame.shape) == 3 and orig_frame.shape[2] == 4:
                    orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2GRAY)
                    orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2RGB)
                elif len(orig_frame.shape) == 2:
                    orig_gray = orig_frame
                    orig_frame = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB)
                else:
                    print(f"🛑 Unsupported image format: {orig_frame.shape}")
                    continue
            elif source == 'clarius':
                orig_frame = np.array(frame_data.image)
                orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGRA2GRAY)
                orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGRA2RGB)

            inference_output = guidance_pipeline.run_inference(orig_gray)

            if latest_lvef is None:
                guidance_pipeline.run_lvef_async()
                latest_lvef = guidance_pipeline.get_latest_lvef()

            result_frame = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB)
            result_frame = landmark_renderer.overlay_landmarks(
                result_frame,
                inference_output.landmarks_tensor,
                reference_size=None,
            )
            result_frame = ih.add_rgb_border_to_result(
                result_frame, inference_output.pose_classification_prediction_rgb
            )
            if latest_lvef is not None:
                ih.draw_text_on_frame(result_frame, "LVEF: " + str(round(latest_lvef)) + "%")

            try:
                widget_display = probe_renderer.render_widget(qw=qw, qx=qx, qy=qy, qz=qz)
                guidance_img = (
                    widget_display if isinstance(widget_display, np.ndarray)
                    else np.array(widget_display)
                )
                if len(guidance_img.shape) == 3 and guidance_img.shape[2] == 4:
                    guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_RGBA2RGB)
                elif len(guidance_img.shape) == 2:
                    guidance_img = cv2.cvtColor(guidance_img, cv2.COLOR_GRAY2RGB)
            except Exception as e:
                print(f"⚠️  Render error: {e}")
                guidance_img = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(guidance_img, "Render Error", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if not main_window.display_frame_rgb(result_frame.copy()):
                print("❌ Failed to display main frame")
                break
            if not guidance_window.display_frame_rgb(guidance_img.copy()):
                print("❌ Failed to display guidance frame")
                break

            if not main_window.update() or not guidance_window.update():
                print("🛑 One or both windows closed")
                break

            frame_count += 1
            time.sleep(0.1)
            if frame_count % 50 == 0:
                print(f"📊 Frame {frame_count}")

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
    parser = argparse.ArgumentParser(
        description="Guidance Widget Test - 3D renderer with frames or Clarius streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_guidance.py frames /path/to/frames --scale 2.5 --wireframe
  python test_guidance.py clarius --clarius 192.168.1.1 5828
        """,
    )

    parser.add_argument('source', choices=['frames', 'clarius'],
                        help='Data source: frames folder or Clarius streaming')
    parser.add_argument('frames_folder', nargs='?',
                        help='Path to the frame_imu folder (required for frames source)')
    parser.add_argument('--clarius', '-c', nargs=2, metavar=('IP', 'PORT'),
                        help='Clarius probe IP address and port (required for clarius source)')
    parser.add_argument('--scale', type=float, default=2.0,
                        help='Scale factor for 3D objects')
    parser.add_argument('--wireframe', action='store_true',
                        help='Enable wireframe mode')
    parser.add_argument('--window-size', type=int, nargs=2, default=[400, 400],
                        metavar=('WIDTH', 'HEIGHT'), help='Window size for guidance renderer')
    parser.add_argument('--background-color', type=int, nargs=3, default=[0, 0, 0],
                        metavar=('R', 'G', 'B'), help='Background color RGB values')

    args = parser.parse_args()

    if args.source == 'frames':
        if not args.frames_folder:
            parser.error("frames_folder is required when using 'frames' source")
        if not os.path.exists(args.frames_folder):
            parser.error(f"Frames folder '{args.frames_folder}' does not exist")
        clarius_ip, clarius_port = None, None
    else:
        if not clarius_available:
            parser.error("Clarius SDK is not available. Use 'frames' source instead.")
        if not args.clarius:
            parser.error("--clarius IP PORT is required when using 'clarius' source")
        clarius_ip, clarius_port = args.clarius[0], int(args.clarius[1])
        args.frames_folder = None

    print("Guidance Widget Test")
    print("=" * 50)
    print(f"Source: {args.source}")
    if args.source == 'frames':
        print(f"Frames folder: {args.frames_folder}")
    else:
        print(f"Clarius IP: {clarius_ip}, Port: {clarius_port}")
    print(f"Window size: {args.window_size}")
    print(f"Background: {args.background_color}")
    print(f"Scale factor: {args.scale}")
    print(f"Wireframe: {args.wireframe}")
    print("=" * 50)

    try:
        bg = tuple(args.background_color) + (1,)
        run_guidance(
            source=args.source,
            frames_folder=args.frames_folder,
            clarius_ip=clarius_ip,
            clarius_port=clarius_port,
            window_size=tuple(args.window_size),
            background_color=bg,
            scale_factor=args.scale,
            wireframe=args.wireframe,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")


if __name__ == "__main__":
    main()
