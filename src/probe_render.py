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
    parser.add_argument("--probe-units", type=str, default="cm",
                        help="Native unit of the probe mesh: mm, cm, dm, m, in, "
                             "or a meters-per-unit float")
    parser.add_argument("--camera-distance-m", type=float, default=None,
                        help="Camera distance from scene origin in meters. "
                             "If omitted, auto-fits to the scene AABB.")
    parser.add_argument("--camera-elevation-deg", type=float, default=None,
                        help="Camera elevation above the XZ plane in "
                             "degrees. Higher values look more top-down. "
                             "Defaults to the renderer's built-in value.")
    parser.add_argument("--rotation-amplification", type=float, default=3.0,
                        help="Rotation amplification factor for the renderer")
    parser.add_argument("--wireframe", action="store_true", help="Render in wireframe mode")
    args = parser.parse_args()

    try:
        probe_units = float(args.probe_units)
    except ValueError:
        probe_units = args.probe_units

    probe_mesh_path = os.getenv("PROBE_MESH_PATH")
    if not probe_mesh_path:
        raise RuntimeError("🛑 PROBE_MESH_PATH is not set. Define it in .env or the environment.")
    if not os.path.exists(probe_mesh_path):
        raise FileNotFoundError(f"🛑 Probe mesh not found at PROBE_MESH_PATH={probe_mesh_path}")

    legend_png_path = os.getenv("GUIDANCE_WIDGET_LEGEND_PNG_PATH", "")

    reference_body_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ui", "assets", "body_model.obj",
    )

    streamer = _build_streamer(args.source, args.frames_folder, args.clarius_ip, args.clarius_port)

    elevation_kwarg = (
        {"camera_elevation_deg": args.camera_elevation_deg}
        if args.camera_elevation_deg is not None else {}
    )
    renderer = GuidanceWidgetRenderer3D(
        probe_mesh_path,
        background_color=(0, 0, 0, 1),
        wireframe=args.wireframe,
        window_size=(args.window_size, args.window_size),
        render_mode="fullscreen",
        rotation_amplification=args.rotation_amplification,
        legend_image_path=legend_png_path,
        probe_units=probe_units,
        camera_distance_m=args.camera_distance_m,
        reference_body_path=reference_body_path,
        **elevation_kwarg,
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
    latest_imu = {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}

    def request_quit():
        state["quit"] = True

    def calibrate():
        renderer.calibrate(**latest_imu)

    def reset_calibration():
        renderer.reset_calibration()

    window.register_key_handler("q", request_quit)
    window.register_key_handler("Escape", request_quit)
    window.register_key_handler("c", calibrate)
    window.register_key_handler("r", reset_calibration)

    # Guidance arrow + layer toggles. Arrow keys pick which translation
    # arrow renders; Shift+arrow does the same for rotation. Pressing any
    # arrow key auto-enables the guidance_arrows layer so the selection
    # is visible without a separate "show" step.
    def select_translation(direction: str):
        def handler():
            renderer.set_layer_visible("guidance_arrows", True)
            renderer.set_active_translation_arrow(direction)
            print(f"➡️  translation arrow: {direction}")
        return handler

    def select_rotation(direction: str):
        def handler():
            renderer.set_layer_visible("guidance_arrows", True)
            renderer.set_active_rotation_arrow(direction)
            print(f"🔄 rotation arrow: {direction}")
        return handler

    window.register_key_handler("Left",  select_translation("left"))
    window.register_key_handler("Right", select_translation("right"))
    window.register_key_handler("Up",    select_translation("up"))
    window.register_key_handler("Down",  select_translation("down"))
    window.register_key_handler("Shift+Left",  select_rotation("left"))
    window.register_key_handler("Shift+Right", select_rotation("right"))
    window.register_key_handler("Shift+Up",    select_rotation("up"))
    window.register_key_handler("Shift+Down",  select_rotation("down"))

    def toggle_layer(name: str):
        def handler():
            on = renderer.toggle_layer(name)
            print(f"{'✅' if on else '⛔'} {name}: {'on' if on else 'off'}")
        return handler

    window.register_key_handler("a", toggle_layer("guidance_arrows"))
    window.register_key_handler("p", toggle_layer("predicted_axes"))

    identity_rotation = np.eye(3)
    started = False

    try:
        streamer.start()
        started = True
        print(
            "▶️  Probe render started.\n"
            "    Calibration: 'c' calibrate (hold probe parallel to long "
            "axis, button up, tip superior), 'r' reset calibration.\n"
            "    Layers:      'a' toggle guidance arrows, 'p' toggle "
            "predicted axes.\n"
            "    Arrows:      ←/→/↑/↓ pick translation arrow, "
            "Shift+←/→/↑/↓ pick rotation arrow.\n"
            "    Quit:        'q' or Escape."
        )

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
                latest_imu["qw"] = qw
                latest_imu["qx"] = qx
                latest_imu["qy"] = qy
                latest_imu["qz"] = qz
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
