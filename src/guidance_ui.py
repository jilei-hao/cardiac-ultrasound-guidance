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
from inference.probe_pose import NUM_VIEWS, view_list_for
from data_access.screen_capture.screen_capture import ScreenCapture
from ui.image_window import create_window_row
from ui.image_helpers import add_rgb_border_to_result, draw_text_on_frame
from ui.guidance_widget.guidance_widget_renderer_3d import GuidanceWidgetRenderer3D
from ui.reticle import ReticleRenderer
from ui.fourch_landmark_renderer import FourChamberLandmarkRenderer
from data_access.simulator import FrameDataSimulator
from common.application_settings import ApplicationSettings


@dataclass
class GuidanceUIConfig:
    env_config: dict
    input_source: str # simulator, clarius, screen
    frames_folder: str = None
    sim_realtime: bool = False
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
        self._reticle_window = None
        self._guidance_window = None
        self._landmark_renderer = None
        self._reticle_renderer = None
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
        # Target standard view for probe-pose guidance (index into the active
        # view list). Default index 0. Switched live via digit keys.
        self._target_view_index = 0
        # View-name list for the HUD. Defaults to 'invivo' (the ema_11000
        # mapping; the simulated demo-convnext order is legacy and was the wrong
        # thing to show). Override with PROBE_POSE_VIEW_SET=simulated only if you
        # load an old simulated checkpoint. One-hot indices are shared.
        self._view_list = view_list_for(
            self._env_config.get("probe_pose_view_set") or "invivo"
        )
        # Which model drives the rotation arrows: 'usgr' (view-conditioned
        # PoseTransformer) or 'legacy' (RotationPredictor). Toggled with 'g'.
        # Falls back to 'legacy' automatically when the USGR model is absent.
        self._rotation_source = 'usgr'
        # Which model drives the traffic-light border: 'legacy' (ResNet18 pose
        # classifier) or 'quality' (USGR ScoreTransformer). Toggled with 'm'.
        # Defaults to 'legacy'; the quality model is opt-in and only runs while
        # selected. Falls back to 'legacy' when the quality model is absent.
        self._traffic_light_source = 'legacy'
        # Quality-border colour mode: discrete R/Y/G (default) or a continuous
        # red->yellow->green gradient. Toggled with 'Shift+m'; only affects the
        # 'quality' source. Seeded from QUALITY_SCORE_CONTINUOUS.
        self._traffic_light_continuous = bool(
            self._env_config.get("quality_score_continuous")
        )
    
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
            },
            # Optional USGR probe-pose (standard-view) guidance. Disabled
            # automatically when any path is missing/None.
            probe_pose_weights=self._env_config.get("probe_pose_weights"),
            probe_pose_norm_stats=self._env_config.get("probe_pose_norm_stats"),
            probe_pose_dino_weights=self._env_config.get("probe_pose_dino_weights"),
            probe_pose_dino_model=self._env_config.get("probe_pose_dino_model")
            or "dinov3_convnext_base",
            probe_pose_image_size=self._env_config.get("probe_pose_image_size") or 512,
            probe_pose_mirror_input=bool(
                self._env_config.get("probe_pose_mirror_input")
            ),
            # Optional USGR quality-score ("traffic light v2") classifier.
            # Disabled (predictor not built) when the head/backbone weights are
            # missing. Toggled at runtime with 'm'.
            quality_score_weights=self._env_config.get("quality_score_weights"),
            quality_score_dino_weights=self._env_config.get("quality_score_dino_weights"),
            quality_score_dino_model=self._env_config.get("quality_score_dino_model")
            or "dinov3_vitl16",
            quality_score_image_size=self._env_config.get("quality_score_image_size") or 512,
            quality_score_num_heads=self._env_config.get("quality_score_num_heads") or 16,
            quality_score_mirror_input=bool(
                self._env_config.get("quality_score_mirror_input")
            ),
            quality_score_red_below=(
                self._env_config["quality_score_red_below"]
                if self._env_config.get("quality_score_red_below") is not None
                else -1.0
            ),
            quality_score_green_above=(
                self._env_config["quality_score_green_above"]
                if self._env_config.get("quality_score_green_above") is not None
                else 1.0
            ),
            # Unified border-smoothing window for whichever source is active
            # (None -> per-source fallback below). Prefer this over the
            # deprecated per-source *_pool_size.
            traffic_light_pool_size=self._env_config.get("traffic_light_pool_size"),
            quality_score_pool_size=self._env_config.get("quality_score_pool_size") or 15,
            # Continuous-gradient border mode (toggled at runtime with 'Shift+m').
            quality_score_continuous=bool(
                self._env_config.get("quality_score_continuous")
            ),
            quality_score_score_min=(
                self._env_config["quality_score_score_min"]
                if self._env_config.get("quality_score_score_min") is not None
                else -3.0
            ),
            quality_score_score_max=(
                self._env_config["quality_score_score_max"]
                if self._env_config.get("quality_score_score_max") is not None
                else 3.0
            ),
            # fp16 autocast for both USGR ViT-L forwards (~2.25x; parity
            # validated — see GuidancePipelineConfig.usgr_autocast_fp16).
            usgr_autocast_fp16=bool(
                self._env_config.get("usgr_autocast_fp16")
            ),
        )
        self._guidance_pipeline = GuidancePipeline(config=guidance_config)

        # Initialize the Windows. Left-to-right: 3D guidance widget, the 2D
        # reticle, then the annotated ultrasound — the reticle sits between the
        # 3D view of the correction and the image it is derived from.
        (self._guidance_window,
         self._reticle_window,
         self._main_window) = create_window_row(
            ["Guidance Widget", "2D Reticle", "Annotated Ultrasound"],
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
            # 1.0 = show the true predicted correction. USGR's R_rel is a metric
            # transform, so amplifying it just magnifies prediction jitter
            # (the legacy >1 default existed to make tiny RotationPredictor
            # corrections visible). Overridable via PROBE_POSE_ROTATION_AMP.
            rotation_amplification=float(
                self._env_config.get("probe_pose_rotation_amp") or 1.0
            ),
            legend_image_path=self._env_config["guidance_widget_legend_png_path"],
            probe_units='cm',
            reference_body_path=self._env_config.get("reference_body_path"),
        )

        # 2D reticle (USGR). Driven by the same probe-pose prediction as the
        # 3D widget, so it shows nothing useful unless probe-pose is configured.
        self._reticle_renderer = ReticleRenderer(
            output_size=self._config.window_size
        )

        # Initialize the data streamer
        if self._config.input_source == "simulator":
            self._data_streamer = FrameDataSimulator(
                frames_folder=self._config.frames_folder,
                realtime=self._config.sim_realtime,
            )
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

        Precondition: ``_main_window``, ``_reticle_window`` and
        ``_guidance_window`` have been assigned in ``initialize``. Asserts
        narrow the type for the registration calls below.
        """
        assert self._main_window is not None
        assert self._reticle_window is not None
        assert self._guidance_window is not None

        # Quit
        def handle_quit():
            print("Quit key pressed!")
            self._main_window.destroy()
            self._reticle_window.destroy()
            self._guidance_window.destroy()

        self._main_window.register_key_handler('q', handle_quit)
        self._reticle_window.register_key_handler('q', handle_quit)
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

        # 'b' — toggle the semi-transparent target probe ghost. Only
        # visible when a rotation prediction is available; the toggle
        # state is preserved across frames either way.
        def handle_toggle_target_probe():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "toggle_layer"):
                return
            on = renderer.toggle_layer("target_probe")
            print(f"{'✅' if on else '⛔'} target probe: {'on' if on else 'off'}")

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

        # '[' / ']' — zoom out / zoom in. Multiplicative so repeats stay
        # geometric. Step matches probe_render.py.
        camera_zoom_step = 1.1
        handle_zoom_in = make_nudge("nudge_camera_zoom", camera_zoom_step)
        handle_zoom_out = make_nudge("nudge_camera_zoom", 1.0 / camera_zoom_step)

        # ';' — reset camera to the constructor's azimuth/elevation defaults.
        def handle_reset_camera():
            renderer = self._guidance_widget_renderer
            if renderer is None or not hasattr(renderer, "reset_camera"):
                return
            renderer.reset_camera()

        # '1'..'6' — select the target standard view for probe-pose guidance.
        #   1=A4C, 2=A2C, 3=PSAX_PAP, 4=PSAX_MV, 5=PLAX, 6=SC4C
        def make_select_view(view_idx: int):
            def handler():
                if not (0 <= view_idx < NUM_VIEWS):
                    return
                self._target_view_index = view_idx
                print(f"🎯 target view: {view_idx + 1}={self._view_list[view_idx]}")
            return handler

        # 'g' — toggle the rotation-arrow source between the view-conditioned
        # USGR model and the legacy rotation predictor.
        def handle_toggle_rotation_source():
            self._rotation_source = (
                'legacy' if self._rotation_source == 'usgr' else 'usgr'
            )
            available = (
                self._guidance_pipeline is not None
                and self._guidance_pipeline.probe_pose_available
            )
            note = "" if (available or self._rotation_source == 'legacy') \
                else " (USGR unavailable -> using legacy)"
            print(f"🔀 rotation source: {self._rotation_source}{note}")

        # 'm' — toggle the traffic-light border source between the legacy
        # ResNet18 pose classifier and the USGR quality ScoreTransformer.
        # No-ops back to legacy (with a note) when the quality model is absent.
        def handle_toggle_traffic_light():
            if self._traffic_light_source == 'legacy':
                enabled = (
                    self._guidance_pipeline is not None
                    and self._guidance_pipeline.set_quality_score_enabled(True)
                )
                if not enabled:
                    print("⛔ quality traffic-light model unavailable "
                          "(set QUALITY_SCORE_WEIGHTS + QUALITY_SCORE_DINO_WEIGHTS); "
                          "staying on legacy")
                    return
                self._traffic_light_source = 'quality'
            else:
                if self._guidance_pipeline is not None:
                    self._guidance_pipeline.set_quality_score_enabled(False)
                self._traffic_light_source = 'legacy'
            print(f"🚦 traffic light: {self._traffic_light_source}")

        # 'Shift+m' — toggle BOTH traffic-light borders (legacy + quality)
        # between the discrete three-colour light and a continuous
        # red->yellow->green gradient. Source-agnostic now that the legacy
        # classifier also exposes a continuous score (both map through the
        # shared inference/traffic_light scale).
        def handle_toggle_continuous_border():
            self._traffic_light_continuous = not self._traffic_light_continuous
            if self._guidance_pipeline is not None:
                self._guidance_pipeline.set_traffic_light_continuous(
                    self._traffic_light_continuous
                )
            mode = 'continuous gradient' if self._traffic_light_continuous \
                else 'discrete R/Y/G'
            print(f"🌈 traffic-light border: {mode} "
                  f"(source: {self._traffic_light_source})")

        # 'k' — toggle ESKF (Kalman) smoothing of the probe-pose prediction.
        def handle_toggle_filter():
            if self._guidance_pipeline is None:
                return
            new_state = self._guidance_pipeline.set_probe_pose_filter(
                not self._guidance_pipeline.probe_pose_filter_enabled
            )
            print(f"{'✅' if new_state else '⛔'} pose smoothing (Kalman): "
                  f"{'on' if new_state else 'off'}")

        # Register on every window so the hotkeys work regardless of which
        # one has keyboard focus.
        for window in (self._main_window, self._reticle_window, self._guidance_window):
            window.register_key_handler('a', handle_toggle_arrows)
            window.register_key_handler('x', handle_toggle_axes)
            window.register_key_handler('c', handle_calibrate)
            window.register_key_handler('Shift+c', handle_reset_calibration)
            window.register_key_handler('p', handle_toggle_patient)
            window.register_key_handler('b', handle_toggle_target_probe)
            window.register_key_handler('f', handle_toggle_fps)
            window.register_key_handler('t', handle_toggle_timing)
            window.register_key_handler('d', handle_toggle_debug)
            window.register_key_handler(',', handle_azimuth_minus)
            window.register_key_handler('.', handle_azimuth_plus)
            window.register_key_handler("'", handle_elevation_plus)
            window.register_key_handler('/', handle_elevation_minus)
            window.register_key_handler('[', handle_zoom_out)
            window.register_key_handler(']', handle_zoom_in)
            window.register_key_handler(';', handle_reset_camera)
            window.register_key_handler('g', handle_toggle_rotation_source)
            window.register_key_handler('m', handle_toggle_traffic_light)
            window.register_key_handler('Shift+m', handle_toggle_continuous_border)
            window.register_key_handler('k', handle_toggle_filter)
            for _vi in range(NUM_VIEWS):
                window.register_key_handler(str(_vi + 1), make_select_view(_vi))

    def run(self):
        """Run the main UI loop."""
            # Initialize both windows
        if not self._main_window.initialize():
            print("❌ Failed to initialize main window")
            return

        if not self._reticle_window.initialize():
            print("❌ Failed to initialize reticle window")
            self._main_window.destroy()
            return

        if not self._guidance_window.initialize():
            print("❌ Failed to initialize guidance window")
            self._main_window.destroy()
            self._reticle_window.destroy()
            return


        try:
            if self._data_streamer:
                self._data_streamer.start()
            else:
                raise RuntimeError("🛑 Error: Data streamer not initialized. Call initialize() first.")

            while (self._main_window.is_window_active()
                   and self._reticle_window.is_window_active()
                   and self._guidance_window.is_window_active()):
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
                    self._guidance_pipeline.run_inference,
                    orig_gray_512,
                    self._target_view_index,
                    frame_data.timestamp,
                )

                # Choose what drives the widget from the PREVIOUS frame's result.
                # Rotation source is selectable ('g'); translation always comes
                # from the USGR probe-pose prediction when available.
                last_out = self._last_inference_output
                last_rotation = None
                last_translation = None
                probe_pose = None
                if last_out is not None:
                    probe_pose = getattr(last_out, "probe_pose_prediction", None)
                    use_usgr_rot = (
                        self._rotation_source == 'usgr'
                        and probe_pose is not None and probe_pose.valid
                    )
                    # USGR's R_rel is the FORWARD correction (target = current @
                    # R_rel), but render_widget composes the INVERSE (legacy
                    # RotationPredictor convention, whose output was the
                    # already-applied misalignment). Transpose at the boundary so
                    # the ghost/arrows show the forward correction. The legacy
                    # path keeps the renderer's native inverse convention.
                    last_rotation = (
                        probe_pose.R_rel.T if use_usgr_rot
                        else last_out.rotation_prediction
                    )
                    if probe_pose is not None and probe_pose.valid:
                        last_translation = probe_pose.T_rel
                out_frame_guidance_bgr = self._guidance_widget_renderer.render_widget(
                    orig_frame,
                    qw, qx, qy, qz,
                    last_rotation,
                    last_translation,
                )
                self._draw_view_label(out_frame_guidance_bgr)

                t_widget = time.perf_counter() if timing_on else 0.0

                # 2D reticle — same prediction the 3D widget just consumed, so
                # the two panels never disagree by a frame. Fed the model's
                # native forward transform (no transpose; see ReticleRenderer).
                out_frame_reticle_bgr = self._reticle_renderer.render(probe_pose)
                self._draw_reticle_label(out_frame_reticle_bgr)

                t_reticle = time.perf_counter() if timing_on else 0.0

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

                # Traffic-light border: the legacy ResNet classifier by default,
                # or the USGR quality model when toggled on ('m'). The quality
                # rgb is only present while that model is enabled; fall back to
                # legacy otherwise so the border never goes blank mid-toggle.
                if (self._traffic_light_source == 'quality'
                        and inference_output.quality_rgb is not None):
                    border_rgb = inference_output.quality_rgb
                else:
                    border_rgb = inference_output.pose_classification_prediction_rgb
                out_frame_main_bgr = add_rgb_border_to_result(out_frame_main_bgr, border_rgb)
                self._draw_traffic_light_label(out_frame_main_bgr, inference_output)

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
                out_frame_reticle = cv2.cvtColor(out_frame_reticle_bgr, cv2.COLOR_BGR2RGB)
                out_frame_guidance = cv2.cvtColor(out_frame_guidance_bgr, cv2.COLOR_BGR2RGB)


                # Display output frame
                if not self._main_window.display_frame_rgb(out_frame_main):
                    print("❌ Failed to display main frame")
                    break

                if not self._reticle_window.display_frame_rgb(out_frame_reticle):
                    print("❌ Failed to display reticle frame")
                    break

                if not self._guidance_window.display_frame_rgb(out_frame_guidance):
                    print("❌ Failed to display guidance frame")
                    break

                # Update all windows
                if (not self._main_window.update()
                        or not self._reticle_window.update()
                        or not self._guidance_window.update()):
                    print("🛑 One or more windows closed")
                    break

                if timing_on:
                    t_display = time.perf_counter()
                    sample = {
                        "capture":    (t_capture     - t_loop_start) * 1e3,
                        "preprocess": (t_preprocess  - t_capture)    * 1e3,
                        "widget":     (t_widget      - t_preprocess) * 1e3,
                        "reticle":    (t_reticle     - t_widget)     * 1e3,
                        "infer_wait": (t_infer_wait  - t_reticle)    * 1e3,
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
        outer = ["capture", "preprocess", "widget", "reticle", "infer_wait", "overlay", "display"]
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

    def _draw_view_label(self, frame_bgr: np.ndarray) -> None:
        """Draw the active target view + rotation source bottom-left (in-place)."""
        view = self._view_list[self._target_view_index]
        # Reflect the EFFECTIVE source: 'usgr' silently falls back to legacy
        # when the probe-pose model is unavailable.
        usgr_ok = (
            self._guidance_pipeline is not None
            and self._guidance_pipeline.probe_pose_available
        )
        eff_source = self._rotation_source if (usgr_ok or self._rotation_source == 'legacy') else 'legacy*'
        filt = "on" if (usgr_ok and self._guidance_pipeline.probe_pose_filter_enabled) else "off"
        text = f"View: {self._target_view_index + 1} {view}  |  rot: {eff_source}  |  smooth: {filt}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        margin = 10
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = margin
        y = frame_bgr.shape[0] - margin
        cv2.putText(frame_bgr, text, (x + 1, y + 1), font, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (x, y), font, scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)

    def _draw_reticle_label(self, frame_bgr: np.ndarray) -> None:
        """Draw the reticle's target view bottom-left (in-place).

        Deliberately does NOT echo the 'g' rotation-source toggle: the reticle
        is always driven by the USGR probe-pose prediction, so reprinting
        'rot: legacy' here would claim otherwise.
        """
        view = self._view_list[self._target_view_index]
        usgr_ok = (
            self._guidance_pipeline is not None
            and self._guidance_pipeline.probe_pose_available
        )
        filt = "on" if (usgr_ok and self._guidance_pipeline.probe_pose_filter_enabled) else "off"
        text = f"Target: {self._target_view_index + 1} {view}  |  smooth: {filt}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        margin = 10
        x = margin
        y = frame_bgr.shape[0] - margin
        cv2.putText(frame_bgr, text, (x + 1, y + 1), font, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (x, y), font, scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)

    def _draw_traffic_light_label(self, frame_bgr: np.ndarray, inference_output) -> None:
        """Top-left readout of the active traffic-light model (in-place).

        Shows 'legacy' or 'quality'; for the quality model it also prints the
        raw continuous score. Reflects the EFFECTIVE source -- 'quality' falls
        back to legacy when that model is unavailable.
        """
        quality_ok = (
            self._guidance_pipeline is not None
            and self._guidance_pipeline.quality_score_available
        )
        if self._traffic_light_source == 'quality' and quality_ok:
            score = inference_output.quality_score
            score_str = f" s={score:+.2f}" if score is not None else ""
            mode_str = " [grad]" if self._traffic_light_continuous else ""
            text = f"TL: quality{score_str}{mode_str}"
        else:
            text = "TL: legacy"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        margin = 10
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = margin
        y = margin + th
        cv2.putText(frame_bgr, text, (x + 1, y + 1), font, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (x, y), font, scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    import os

    parser = argparse.ArgumentParser(description="Run the Guidance UI")
    parser.add_argument("--source", type=str, default="simulator", help="Data source: simulator, clarius, screen")
    parser.add_argument("--clarius-ip", type=str, default=None, help="Clarius probe IP address")
    parser.add_argument("--clarius-port", type=int, default=5828, help="Clarius probe port number")
    parser.add_argument("--frames-folder", type=str, default=None,
                        help="Folder containing frames for simulator: either a folder of "
                             "frame_*.png (+ frame.imu.csv/frame.meta.csv), or a USGR/"
                             "cusg-record capture root with an images/ subfolder")
    parser.add_argument("--sim-realtime", action="store_true", default=False,
                        help="Pace simulator playback to the recorded timestamps "
                             "instead of free-running at inference speed")
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
            # USGR probe-pose (standard-view) guidance assets. Unset -> the
            # feature is disabled and the app runs rotation-only.
            "probe_pose_weights": os.getenv("PROBE_POSE_WEIGHTS"),
            "probe_pose_norm_stats": os.getenv("PROBE_POSE_NORM_STATS"),
            "probe_pose_dino_weights": os.getenv("PROBE_POSE_DINO_WEIGHTS"),
            # Encoder variant + input size for the probe-pose checkpoint.
            # Default convnext_base/512 (demo checkpoints); set
            # PROBE_POSE_DINO_MODEL=dinov3_vitl16 for the 0610 invivo run.
            "probe_pose_dino_model": os.getenv("PROBE_POSE_DINO_MODEL"),
            # View-name set for the HUD: 'simulated' (demo convnext) or 'invivo'
            # (ema_11000). Default simulated when unset.
            "probe_pose_view_set": os.getenv("PROBE_POSE_VIEW_SET"),
            # Visual gain on the predicted rotation in the 3D widget. Default 1.0
            # (true correction); raise to exaggerate for visibility.
            "probe_pose_rotation_amp": os.getenv("PROBE_POSE_ROTATION_AMP"),
            # H-flip input to training chirality. ON for live Clarius (LV-right),
            # OFF for LV-left input (invivo reproduction data). Truthy strings only.
            "probe_pose_mirror_input": (
                str(os.getenv("PROBE_POSE_MIRROR_INPUT", "")).strip().lower()
                in ("1", "true", "yes", "on")
            ),
            "probe_pose_image_size": (
                int(os.environ["PROBE_POSE_IMAGE_SIZE"])
                if os.getenv("PROBE_POSE_IMAGE_SIZE") else None
            ),
            # USGR quality-score ("traffic light v2") classifier. Unset
            # QUALITY_SCORE_WEIGHTS -> the model isn't built and 'm' stays on
            # legacy. The DINOv3 backbone defaults to the same vitl16 file
            # probe_pose uses (PROBE_POSE_DINO_WEIGHTS) unless overridden.
            "quality_score_weights": os.getenv("QUALITY_SCORE_WEIGHTS"),
            "quality_score_dino_weights": (
                os.getenv("QUALITY_SCORE_DINO_WEIGHTS")
                or os.getenv("PROBE_POSE_DINO_WEIGHTS")
            ),
            "quality_score_dino_model": (
                os.getenv("QUALITY_SCORE_DINO_MODEL") or "dinov3_vitl16"
            ),
            "quality_score_image_size": (
                int(os.environ["QUALITY_SCORE_IMAGE_SIZE"])
                if os.getenv("QUALITY_SCORE_IMAGE_SIZE") else 512
            ),
            "quality_score_num_heads": (
                int(os.environ["QUALITY_SCORE_NUM_HEADS"])
                if os.getenv("QUALITY_SCORE_NUM_HEADS") else 16
            ),
            "quality_score_mirror_input": (
                str(os.getenv("QUALITY_SCORE_MIRROR_INPUT", "")).strip().lower()
                in ("1", "true", "yes", "on")
            ),
            "quality_score_red_below": (
                float(os.environ["QUALITY_SCORE_RED_BELOW"])
                if os.getenv("QUALITY_SCORE_RED_BELOW") else -1.0
            ),
            "quality_score_green_above": (
                float(os.environ["QUALITY_SCORE_GREEN_ABOVE"])
                if os.getenv("QUALITY_SCORE_GREEN_ABOVE") else 1.0
            ),
            # Unified border-smoothing window (frames) for whichever traffic-
            # light source is active: border = mean of the last N scores. One
            # knob drives both sources. Larger = steadier but laggier. Unset ->
            # per-source fallback (QUALITY_SCORE_POOL_SIZE below).
            "traffic_light_pool_size": (
                int(os.environ["TRAFFIC_LIGHT_POOL_SIZE"])
                if os.getenv("TRAFFIC_LIGHT_POOL_SIZE") else None
            ),
            # Deprecated per-source alias; prefer TRAFFIC_LIGHT_POOL_SIZE.
            "quality_score_pool_size": (
                int(os.environ["QUALITY_SCORE_POOL_SIZE"])
                if os.getenv("QUALITY_SCORE_POOL_SIZE") else 15
            ),
            # Start with the continuous-gradient border instead of discrete
            # R/Y/G (still toggleable at runtime with 'Shift+m').
            "quality_score_continuous": (
                str(os.getenv("QUALITY_SCORE_CONTINUOUS", "")).strip().lower()
                in ("1", "true", "yes", "on")
            ),
            # Score range the continuous gradient spans (red@min .. green@max,
            # yellow at the midpoint). Defaults match the USGR visualizer.
            "quality_score_score_min": (
                float(os.environ["QUALITY_SCORE_MIN"])
                if os.getenv("QUALITY_SCORE_MIN") else -3.0
            ),
            "quality_score_score_max": (
                float(os.environ["QUALITY_SCORE_MAX"])
                if os.getenv("QUALITY_SCORE_MAX") else 3.0
            ),
            # Run both USGR ViT-L forwards (probe_pose + quality) under fp16
            # autocast (~2.25x faster, parity-validated). Truthy strings only.
            "usgr_autocast_fp16": (
                str(os.getenv("USGR_AUTOCAST_FP16", "")).strip().lower()
                in ("1", "true", "yes", "on")
            ),
        },
        input_source=options.source,
        frames_folder=options.frames_folder,
        sim_realtime=options.sim_realtime,
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