import os
import sys

# PYOPENGL_PLATFORM must be set before any pyrender import in the
# process. Linux-only: default to NVIDIA EGL so headless / remote-desktop
# sessions get GPU-accelerated GL instead of Mesa llvmpipe (which on
# remote X11 forwarding software-rasterizes every fragment on the CPU
# and inflates gl_render ~10×). macOS uses Cocoa GL via pyglet and
# Windows uses WGL — both already hit hardware, and EGL isn't typically
# present on either. The shell can still override on Linux — e.g.
# PYOPENGL_PLATFORM=osmesa for hosts without NVIDIA EGL libs.
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import cv2
import numpy as np
from inference.guidance_pipeline import GuidancePipelineConfig, GuidancePipeline
from data_access.screen_capture.screen_capture import ScreenCapture
from ui.image_window import create_dual_window_setup
from ui.image_helpers import add_rgb_border_to_result, draw_text_on_frame
from ui.guidance_widget.guidance_widget_renderer_3d import GuidanceWidgetRenderer3D
from ui.fourch_landmark_renderer import FourChamberLandmarkRenderer
from data_access.simulator import FrameDataSimulator
from common.application_settings import ApplicationSettings


@dataclass
class GuidanceUIConfig:
    env_config: dict
    input_source: str # simulator, clarius, screen
    frames_folder: str = None
    screen_capture_window_title: str = "Input"
    clarius_ip: str = None
    clarius_port: int = 5828
    window_size: int = 512



class GuidanceUI:
    def __init__(self, config, logger=None):
        self._config = config
        self._env_config = config.env_config
        # UI components
        self._main_window = None
        self._guidance_window = None
        self._landmark_renderer = None
        self._guidance_widget_renderer = None
        self._application_settings = None
        # Data Access
        self._data_streamer = None
        # Inference Pipeline
        self._guidance_pipeline = None
        self._latest_lvef = None
        self._lvef_running = False
        # Logger
        self._logger = logger
        # Initialize application settings
        self._app_settings = ApplicationSettings()
        # Latest IMU quaternion, captured each frame so the calibration
        # hotkey can hand the renderer the operator's current pose.
        self._latest_imu_quat = (1.0, 0.0, 0.0, 0.0)
        # FPS readout state, toggled by the 'f' hotkey.
        self._show_fps = False
        self._fps_frame_times: deque = deque(maxlen=30)
        # Per-stage timing breakdown, toggled by the 't' hotkey. When on,
        # the outer loop stamps timestamps around each stage and the
        # pipeline populates last_timings; samples are averaged and
        # printed every TIMING_PRINT_EVERY frames.
        self._show_timing = False
        self._timing_samples: deque = deque(maxlen=30)
        self._TIMING_PRINT_EVERY = 30
        # Single-worker pool that runs run_inference concurrently with the
        # main thread's pyrender widget render. pyrender's GL context is
        # thread-bound, so we keep it on the main thread and ship the
        # CUDA/CPU inference work off-thread instead.
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="infer"
        )
        # Last frame's inference result, used to feed the widget render
        # for the current frame while inference runs in parallel.
        self._last_inference_output = None
    
    def initialize(self):
        """Initialize components"""

        # Initialize the guidance pipeline
        guidance_config = GuidancePipelineConfig(
            fourch_landmark_weights=self._env_config["fourch_landmark_weight_path"],
            rotation_prediction_config=self._env_config["rotation_prediction_config"],
            rotation_prediction_weights=self._env_config["rotation_prediction_weights"],
            pose_classification_weights=self._env_config["pose_est_img_weights"],
            lvef_weights=self._env_config["lvef_weights"],
            landmarks_per_chamber={
                50: 24,   # Left Ventricle (LV)
                100: 12,  # Right Ventricle (RV)
                150: 12,  # Left Atrium (LA)
                200: 12,  # Right Atrium (RA)
            }
        )
        self._guidance_pipeline = GuidancePipeline(config=guidance_config)

        # Initialize the Windows
        self._main_window, self._guidance_window = create_dual_window_setup(
            main_title="Annotated Ultrasound",
            guidance_title="Guidance Widget",
            width=self._config.window_size,
            height=self._config.window_size,
            x_pos=100,
            y_pos=100
        )
        self._create_key_bindings()

        # Initialize the renderers
        self._landmark_renderer = FourChamberLandmarkRenderer(
            landmark_radius=8,
            min_opacity=0.6,
            max_opacity=1.0,
            uncertainty_threshold=6.0,
            reference_size=None,  # Landmarks are already in crop_frame coordinates
            color_legend_png_path=self._env_config["landmark_color_legend_png_path"]
        )

        self._guidance_widget_renderer = GuidanceWidgetRenderer3D(
            self._env_config["probe_mesh_path"],
            background_color=(0, 0, 0, 1),
            wireframe=False,
            window_size=(512, 512),
            render_mode='fullscreen',
            rotation_amplification=3.0,
            legend_image_path=self._env_config["guidance_widget_legend_png_path"],
            probe_units='cm',
            reference_body_path=self._env_config.get("reference_body_path"),
        )

        # Initialize the data streamer
        if self._config.input_source == "simulator":
            self._data_streamer = FrameDataSimulator(frames_folder=self._config.frames_folder)
        elif self._config.input_source == "clarius":
            import data_access.clarius.clarius_streamer as clarius_streamer
            clarius_streamer.configure(
                self._config.clarius_ip,
                self._config.clarius_port,
                "research",
                512,
                512
            )
            self._data_streamer = clarius_streamer
        elif self._config.input_source == "screen":
            self._data_streamer = ScreenCapture(self._config.screen_capture_window_title)
        else:
            raise ValueError(f"🛑 Error: Unknown input source '{self._config.input_source}'.")

    def _create_key_bindings(self):
        """Create key bindings for the UI windows.

        Precondition: both ``_main_window`` and ``_guidance_window`` have
        been assigned in ``initialize``. Asserts narrow the type for the
        registration calls below.
        """
        assert self._main_window is not None
        assert self._guidance_window is not None

        # Quit
        def handle_quit():
            print("Quit key pressed!")
            self._main_window.destroy()
            self._guidance_window.destroy()

        self._main_window.register_key_handler('q', handle_quit)
        self._guidance_window.register_key_handler('q', handle_quit)

        # LVEF computation
        def handle_lvef():
            print("LVEF key pressed!")
            if self._lvef_running:
                print("LVEF computation is already running.")
                return
            if self._guidance_pipeline.run_lvef_async():
                self._latest_lvef = None
                self._lvef_running = True
            else:
                print("LVEF computation not started: buffer not ready.")

        self._main_window.register_key_handler('e', handle_lvef)

        # Overlay Visibility
        def handle_toggle_overlay():
            if self._landmark_renderer.overlay_visible:
                print("Hiding landmark overlay")
                self._app_settings.show_overlay = False
            else:
                print("Showing landmark overlay")
                self._app_settings.show_overlay = True

        self._main_window.register_key_handler('r', handle_toggle_overlay)

        # Renderer is assigned later in ``initialize`` (after this method
        # runs), so handlers must dereference ``self._guidance_widget_renderer``
        # at call time rather than capturing it now.

        # 'a' — toggle the guidance arrows layer (translation, rotation, roll).
        def handle_toggle_arrows():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "toggle_layer"):
                return
            on = renderer.toggle_layer("guidance_arrows")
            print(f"{'✅' if on else '⛔'} guidance arrows: {'on' if on else 'off'}")

        # 'x' — toggle probe + predicted axes together. Uses current_axes as
        # the source of truth so a single press flips both layers in lockstep.
        def handle_toggle_axes():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "set_layer_visible"):
                return
            new_state = not renderer.is_layer_visible("current_axes")
            renderer.set_layer_visible("current_axes", new_state)
            renderer.set_layer_visible("predicted_axes", new_state)
            print(f"{'✅' if new_state else '⛔'} axes: {'on' if new_state else 'off'}")

        # 'c' — calibrate using the most recent IMU sample.
        def handle_calibrate():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "calibrate"):
                return
            qw, qx, qy, qz = self._latest_imu_quat
            renderer.calibrate(qw, qx, qy, qz)

        # 'Shift+c' — drop calibration, render raw IMU rotation again.
        def handle_reset_calibration():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "reset_calibration"):
                return
            renderer.reset_calibration()

        # 'p' — toggle the reference patient mesh shown in the corner.
        def handle_toggle_patient():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "toggle_layer"):
                return
            on = renderer.toggle_layer("reference_body")
            print(f"{'✅' if on else '⛔'} patient mesh: {'on' if on else 'off'}")

        # 'f' — toggle the FPS readout drawn on the main frame.
        def handle_toggle_fps():
            self._show_fps = not self._show_fps
            if not self._show_fps:
                self._fps_frame_times.clear()
            print(f"{'✅' if self._show_fps else '⛔'} fps: {'on' if self._show_fps else 'off'}")

        # 't' — toggle per-stage timing breakdown printed to stdout.
        def handle_toggle_timing():
            self._show_timing = not self._show_timing
            self._timing_samples.clear()
            if self._guidance_pipeline is not None:
                self._guidance_pipeline.timing_enabled = self._show_timing
            if self._guidance_widget_renderer is not None:
                self._guidance_widget_renderer.timing_enabled = self._show_timing
            print(f"{'✅' if self._show_timing else '⛔'} timing: {'on' if self._show_timing else 'off'}")

        # 'd' — toggle the per-frame prediction-delta debug print.
        def handle_toggle_debug():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "toggle_debug_print"):
                return
            renderer.toggle_debug_print()

        # ',' '.' '\'' '/' — live camera nudges (azimuth / elevation).
        # Step size matches probe_render.py.
        camera_step_deg = 10.0

        def make_nudge(method_name: str, delta_deg: float):
            def handler():
                renderer = self._guidance_widget_renderer
                if renderer is None or not hasattr(renderer, method_name):
                    return
                getattr(renderer, method_name)(delta_deg)
            return handler

        handle_azimuth_minus = make_nudge("nudge_camera_azimuth", -camera_step_deg)
        handle_azimuth_plus = make_nudge("nudge_camera_azimuth", +camera_step_deg)
        handle_elevation_plus = make_nudge("nudge_camera_elevation", +camera_step_deg)
        handle_elevation_minus = make_nudge("nudge_camera_elevation", -camera_step_deg)

        # ';' — reset camera to the constructor's azimuth/elevation defaults.
        def handle_reset_camera():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "reset_camera"):
                return
            renderer.reset_camera()

        # Register on both windows so the hotkeys work regardless of which
        # one has keyboard focus.
        for window in (self._main_window, self._guidance_window):
            window.register_key_handler('a', handle_toggle_arrows)
            window.register_key_handler('x', handle_toggle_axes)
            window.register_key_handler('c', handle_calibrate)
            window.register_key_handler('Shift+c', handle_reset_calibration)
            window.register_key_handler('p', handle_toggle_patient)
            window.register_key_handler('f', handle_toggle_fps)
            window.register_key_handler('t', handle_toggle_timing)
            window.register_key_handler('d', handle_toggle_debug)
            window.register_key_handler(',', handle_azimuth_minus)
            window.register_key_handler('.', handle_azimuth_plus)
            window.register_key_handler("'", handle_elevation_plus)
            window.register_key_handler('/', handle_elevation_minus)
            window.register_key_handler(';', handle_reset_camera)

    def run(self):
        """Run the main UI loop."""
            # Initialize both windows
        if not self._main_window.initialize():
            print("❌ Failed to initialize main window")
            return

        if not self._guidance_window.initialize():
            print("❌ Failed to initialize guidance window")
            self._main_window.destroy()
            return
        
        try:
            if self._data_streamer:
                self._data_streamer.start()
            else:
                raise RuntimeError("🛑 Error: Data streamer not initialized. Call initialize() first.")

            while self._main_window.is_window_active() and self._guidance_window.is_window_active():
                # Snapshot once per iteration so toggling 't' mid-iteration
                # can't leave half the t_X stamps as 0.0 and the other
                # half as raw perf_counter() seconds.
                timing_on = self._show_timing
                t_loop_start = time.perf_counter() if timing_on else 0.0

                frame_data = self._data_streamer.get_frame()

                if frame_data is None or frame_data.image is None:
                    continue

                orig_frame = np.array(frame_data.image)
                if orig_frame is None:
                    continue

                if frame_data.imu is not None:
                    imu = frame_data.imu
                    qw, qx, qy, qz = imu.qw, imu.qx, imu.qy, imu.qz
                else:
                    qw, qx, qy, qz = 1.0, 1.0, 1.0, 1.0  # Default quaternion if no IMU data
                self._latest_imu_quat = (qw, qx, qy, qz)

                t_capture = time.perf_counter() if timing_on else 0.0

                # Preprocess frame
                orig_gray_512 = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
                orig_gray_512 = cv2.resize(orig_gray_512, (512, 512))

                t_preprocess = time.perf_counter() if timing_on else 0.0

                # Submit inference to the worker thread and render the
                # widget concurrently on the main thread (pyrender owns a
                # thread-bound GL context, so we can't move it). The
                # widget consumes the previous frame's rotation prediction
                # — the 1-frame lag is invisible because the prediction is
                # already smoothed by PoseClassifierResultPooler upstream.
                inference_future = self._inference_executor.submit(
                    self._guidance_pipeline.run_inference, orig_gray_512
                )

                last_rotation = (
                    self._last_inference_output.rotation_prediction
                    if self._last_inference_output is not None else None
                )
                out_frame_guidance_bgr = self._guidance_widget_renderer.render_widget(
                    orig_frame,
                    qw, qx, qy, qz,
                    last_rotation,
                )

                t_widget = time.perf_counter() if timing_on else 0.0

                # Block until the current-frame inference completes.
                inference_output = inference_future.result()
                self._last_inference_output = inference_output

                t_infer_wait = time.perf_counter() if timing_on else 0.0

                # Build the main (annotated) frame.
                out_frame_main_bgr = cv2.cvtColor(orig_gray_512, cv2.COLOR_GRAY2BGR)
                out_frame_main_bgr = self._landmark_renderer.overlay_landmarks(
                    out_frame_main_bgr,
                    inference_output.landmarks_tensor
                )

                out_frame_main_bgr = add_rgb_border_to_result(out_frame_main_bgr, inference_output.pose_classification_prediction_rgb)

                if self._lvef_running:
                    result = self._guidance_pipeline.get_latest_lvef()
                    if result is not None:
                        self._latest_lvef = result
                        self._lvef_running = False

                if self._latest_lvef is not None:
                    draw_text_on_frame(out_frame_main_bgr, "LVEF: " + str(self._latest_lvef) + "%")

                t_overlay = time.perf_counter() if timing_on else 0.0

                if self._show_fps:
                    self._fps_frame_times.append(time.perf_counter())
                    if len(self._fps_frame_times) >= 2:
                        elapsed = self._fps_frame_times[-1] - self._fps_frame_times[0]
                        if elapsed > 0:
                            fps = (len(self._fps_frame_times) - 1) / elapsed
                            self._draw_fps(out_frame_main_bgr, fps)

                out_frame_main = cv2.cvtColor(out_frame_main_bgr, cv2.COLOR_BGR2RGB)
                out_frame_guidance = cv2.cvtColor(out_frame_guidance_bgr, cv2.COLOR_BGR2RGB)


                # Display output frame
                if not self._main_window.display_frame_rgb(out_frame_main):
                    print("❌ Failed to display main frame")
                    break

                if not self._guidance_window.display_frame_rgb(out_frame_guidance):
                    print("❌ Failed to display guidance frame")
                    break

                # Update both windows
                if not self._main_window.update() or not self._guidance_window.update():
                    print("🛑 One or both windows closed")
                    break

                if timing_on:
                    t_display = time.perf_counter()
                    sample = {
                        "capture":    (t_capture     - t_loop_start) * 1e3,
                        "preprocess": (t_preprocess  - t_capture)    * 1e3,
                        "widget":     (t_widget      - t_preprocess) * 1e3,
                        "infer_wait": (t_infer_wait  - t_widget)     * 1e3,
                        "overlay":    (t_overlay     - t_infer_wait) * 1e3,
                        "display":    (t_display     - t_overlay)    * 1e3,
                        "total":      (t_display     - t_loop_start) * 1e3,
                    }
                    for k, v in self._guidance_pipeline.last_timings.items():
                        sample[f"infer.{k}"] = v
                    for k, v in self._guidance_widget_renderer.last_timings.items():
                        sample[f"widget.{k}"] = v
                    self._timing_samples.append(sample)
                    if len(self._timing_samples) >= self._TIMING_PRINT_EVERY:
                        self._print_timing_summary()
                        self._timing_samples.clear()

                # time.sleep(0.05)

        except Exception as e:
            print(f"🛑 [GuidanceUI] main loop exception: {e}")
            self.shutdown()


    def shutdown(self):
        """Shutdown the UI and release resources."""
        if self._data_streamer:
            self._data_streamer.stop()
        if self._inference_executor is not None:
            self._inference_executor.shutdown(wait=False, cancel_futures=True)
        print("🛑 Guidance UI shutdown complete.")

    def _print_timing_summary(self) -> None:
        samples = list(self._timing_samples)
        if not samples:
            return
        keys = list(samples[0].keys())
        avg = {k: sum(s.get(k, 0.0) for s in samples) / len(samples) for k in keys}
        total = avg.get("total", 0.0)
        fps = (1000.0 / total) if total > 0 else 0.0
        outer = ["capture", "preprocess", "widget", "infer_wait", "overlay", "display"]
        outer_str = " ".join(f"{k}={avg[k]:5.1f}" for k in outer if k in avg)
        infer_keys = [k for k in keys if k.startswith("infer.")]
        infer_str = " ".join(
            f"{k.split('.', 1)[1]}={avg[k]:5.1f}" for k in infer_keys
        )
        widget_keys = [k for k in keys if k.startswith("widget.")]
        widget_str = " ".join(
            f"{k.split('.', 1)[1]}={avg[k]:5.1f}" for k in widget_keys
        )
        print(
            f"[timing n={len(samples)}] total={total:5.1f}ms "
            f"({fps:4.1f} fps) | {outer_str}"
            + (f" | infer: {infer_str}" if infer_str else "")
            + (f" | widget: {widget_str}" if widget_str else "")
        )

    @staticmethod
    def _draw_fps(frame_bgr: np.ndarray, fps: float) -> None:
        """Draw an FPS readout in the top-right corner (in-place)."""
        text = f"{fps:5.1f} FPS"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        w = frame_bgr.shape[1]
        margin = 10
        x = w - tw - margin
        y = margin + th
        cv2.putText(frame_bgr, text, (x + 1, y + 1), font, scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (x, y), font, scale,
                    (0, 255, 0), thickness, cv2.LINE_AA)

    


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    import os

    parser = argparse.ArgumentParser(description="Run the Guidance UI")
    parser.add_argument("--source", type=str, default="simulator", help="Data source: simulator, clarius, screen")
    parser.add_argument("--clarius-ip", type=str, default=None, help="Clarius probe IP address")
    parser.add_argument("--clarius-port", type=int, default=5828, help="Clarius probe port number")
    parser.add_argument("--frames-folder", type=str, default=None, help="Folder containing frames for simulator")
    parser.add_argument("--capture-window-title", type=str, default="Input", help="Window title for screen capture")
    parser.add_argument("--window-size", type=int, default=512, help="Window size for the UI windows")

    options = parser.parse_args()

    config = GuidanceUIConfig(
        env_config={
            "fourch_landmark_weight_path": os.getenv("4CH_WEIGHTS", "weights/4ch-seg_v2.pt"),
            "rotation_prediction_config": os.getenv("ROTATION_PREDICTION_CONFIG", "src/inference/rotation/config.json"),
            "rotation_prediction_weights": os.getenv("ROTATION_PREDICTION_WEIGHTS", "weights/rotation-prediction_v1.pt"),
            "pose_est_img_weights": os.getenv("POSE_EST_IMG_WEIGHTS", "weights/pose-est-img_v1.pth"),
            "lvef_weights": os.getenv("LVEF_WEIGHTS", "weights/echonet_v1.pt"),
            "probe_mesh_path": os.getenv("PROBE_MESH_PATH", "assets/transducer.stl"),
            "landmark_color_legend_png_path": os.getenv("LANDMARK_COLOR_LEGEND_PNG_PATH", "assets/4ch-legend_static_v1.png"),
            "guidance_widget_legend_png_path": os.getenv("GUIDANCE_WIDGET_LEGEND_PNG_PATH", "assets/guidance-widget-legend_v1.png"),
            "reference_body_path": os.getenv(
                "REFERENCE_BODY_PATH",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "assets", "body_model.obj"),
            ),
        },
        input_source=options.source,
        frames_folder=options.frames_folder,
        screen_capture_window_title=options.capture_window_title,
        clarius_ip=options.clarius_ip,
        clarius_port=options.clarius_port,
        window_size=options.window_size
    )

    ui = GuidanceUI(config)
    ui.initialize()
    ui.run()

if __name__ == "__main__":
    main()