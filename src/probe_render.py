import argparse
import os
import time

import cv2
import numpy as np

from data_access.simulator import FrameDataSimulator
from ui.guidance_widget.guidance_widget_renderer_3d import GuidanceWidgetRenderer3D
from ui.image_window import create_image_window


def _build_streamer(source: str, frames_folder: str, clarius_ip: str, clarius_port: int):
    if source == "simulator":
        if not frames_folder:
            raise ValueError("🛑 --frames-folder is required when --source simulator")
        return FrameDataSimulator(frames_folder=frames_folder)
    if source == "clarius":
        if not clarius_ip:
            raise ValueError("🛑 --clarius-ip is required when --source clarius")
        import data_access.clarius.clarius_streamer as clarius_streamer
        clarius_streamer.configure(clarius_ip, clarius_port, "research", 512, 512)
        return clarius_streamer
    raise ValueError(f"🛑 Unknown input source '{source}' (expected 'simulator' or 'clarius')")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Standalone 3D probe render with IMU tracking (no inference)")
    parser.add_argument("--source", type=str, default="simulator", choices=["simulator", "clarius"],
                        help="Data source: simulator or clarius")
    parser.add_argument("--frames-folder", type=str, default=None,
                        help="Folder containing frames and frame.imu.csv (required for simulator)")
    parser.add_argument("--clarius-ip", type=str, default=None, help="Clarius probe IP address")
    parser.add_argument("--clarius-port", type=int, default=5828, help="Clarius probe port")
    parser.add_argument("--window-size", type=int, default=640, help="Window side length in pixels")
    parser.add_argument("--scale-factor", type=float, default=3.0, help="Probe mesh scale factor")
    parser.add_argument("--rotation-amplification", type=float, default=3.0,
                        help="Rotation amplification factor for the renderer")
    parser.add_argument("--wireframe", action="store_true", help="Render in wireframe mode")
    args = parser.parse_args()

    probe_mesh_path = os.getenv("PROBE_MESH_PATH")
    if not probe_mesh_path:
        raise RuntimeError("🛑 PROBE_MESH_PATH is not set. Define it in .env or the environment.")
    if not os.path.exists(probe_mesh_path):
        raise FileNotFoundError(f"🛑 Probe mesh not found at PROBE_MESH_PATH={probe_mesh_path}")

    legend_png_path = os.getenv("GUIDANCE_WIDGET_LEGEND_PNG_PATH", "")

    streamer = _build_streamer(args.source, args.frames_folder, args.clarius_ip, args.clarius_port)

    renderer = GuidanceWidgetRenderer3D(
        probe_mesh_path,
        background_color=(0, 0, 0, 1),
        scale_factor=args.scale_factor,
        wireframe=args.wireframe,
        window_size=(args.window_size, args.window_size),
        render_mode="fullscreen",
        rotation_amplification=args.rotation_amplification,
        legend_image_path=legend_png_path,
    )

    window = create_image_window(
        title="Probe Render",
        width=args.window_size,
        height=args.window_size,
    )
    if not window.initialize():
        renderer.destroy()
        raise RuntimeError("🛑 Failed to initialize image window")

    state = {"quit": False}

    def request_quit():
        state["quit"] = True

    window.register_key_handler("q", request_quit)
    window.register_key_handler("Escape", request_quit)

    identity_rotation = np.eye(3)
    started = False

    try:
        streamer.start()
        started = True
        print("▶️  Probe render started. Press 'q' or Escape to quit.")

        while window.is_window_active() and not state["quit"]:
            frame_data = streamer.get_frame()

            if frame_data is None or frame_data.image is None:
                if not window.update():
                    break
                time.sleep(0.01)
                continue

            orig_frame = np.array(frame_data.image)

            if frame_data.imu is not None:
                imu = frame_data.imu
                qw, qx, qy, qz = imu.qw, imu.qx, imu.qy, imu.qz
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

            out_bgr = renderer.render_widget(orig_frame, qw, qx, qy, qz, identity_rotation)
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

            if not window.display_frame_rgb(out_rgb):
                print("❌ Failed to display frame")
                break

            if not window.update():
                break

            time.sleep(0.03)

    finally:
        if started:
            try:
                streamer.stop()
            except Exception as e:
                print(f"⚠️  Error stopping streamer: {e}")
        try:
            renderer.destroy()
        except Exception as e:
            print(f"⚠️  Error destroying renderer: {e}")
        try:
            window.destroy()
        except Exception as e:
            print(f"⚠️  Error destroying window: {e}")
        print("🛑 Probe render shutdown complete.")


if __name__ == "__main__":
    main()
