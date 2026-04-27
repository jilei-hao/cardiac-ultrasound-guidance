from dataclasses import dataclass
import os
import time

import cv2
import numpy as np
from inference.guidance_pipeline import GuidancePipelineConfig, GuidancePipeline
from data_access.screen_capture.screen_capture import ScreenCapture
from ui.image_window import create_dual_window_setup
from ui.image_helpers import add_rgb_border_to_result, draw_text_on_frame
from ui.guidance_widget.guidance_widget_renderer_2d import GuidanceWidgetRenderer2D
from ui.guidance_widget.guidance_widget_renderer_3d import GuidanceWidgetRenderer3D
from ui.fourch_landmark_renderer import FourChamberLandmarkRenderer
from data_access.simulator import FrameDataSimulator
from common.application_settings import ApplicationSettings


@dataclass
class GuidanceUIConfig:
    env_config: dict
    input_source: str # simulator, clarius, screen
    force_2d_guidance_widget: bool = False
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

        if self._config.force_2d_guidance_widget:
            self._guidance_widget_renderer = GuidanceWidgetRenderer2D(
                self._env_config["probe_mesh_path"],
                background_color=(0, 0, 0),
                predicted_axis_opacity=0.6, #0-1
                axis_thickness=2,
                axis_colors=((200, 150, 100), (200, 180, 220), (150, 220, 200)),
                axis_labels=('x', 'y', 'z'),
                lod_level=0.25, #mesh simplification level, 0-1, 1 = full detail
                probe_scale=0.7, #probe size relative to axes, 0-1
                rotation_amplification=3.0,  # Amplification factor for rotation prediction
                render_mode='fullscreen',
                window_size=(512, 512),
            )
        else:
            self._guidance_widget_renderer = GuidanceWidgetRenderer3D(
                self._env_config["probe_mesh_path"],
                background_color=(0, 0, 0, 1),
                wireframe=False,
                window_size=(640, 480),
                render_mode='fullscreen',
                rotation_amplification=3.0,
                legend_image_path=self._env_config["guidance_widget_legend_png_path"],
                probe_units='cm',
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
        """Create key bindings for the UI windows."""

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
            if not self._lvef_running:
                self._lvef_running = True
                self._guidance_pipeline.run_lvef_async()
            else:
                print("LVEF computation is already running.")

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

                # Preprocess frame
                orig_gray_512 = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
                orig_gray_512 = cv2.resize(orig_gray_512, (512, 512))

                # Run inference
                inference_output = self._guidance_pipeline.run_inference(orig_gray_512)

                # Prepare output frames
                out_frame_main_bgr = cv2.cvtColor(orig_gray_512, cv2.COLOR_GRAY2BGR)
                out_frame_main_bgr = self._landmark_renderer.overlay_landmarks(
                    out_frame_main_bgr,
                    inference_output.landmarks_tensor
                )
            
                out_frame_main_bgr = add_rgb_border_to_result(out_frame_main_bgr, inference_output.pose_classification_prediction_rgb)
                if self._latest_lvef is not None:
                    self._lvef_running = False  # Reset running flag when result is ready
                    draw_text_on_frame(out_frame_main_bgr, "LVEF: " + str(self._latest_lvef) + "%")

                
                out_frame_guidance_bgr = self._guidance_widget_renderer.render_widget(
                    orig_frame,
                    qw, qx, qy, qz,
                    inference_output.rotation_prediction
                )

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

                time.sleep(0.05)

        except Exception as e:
            print(f"🛑 [GuidanceUI] main loop exception: {e}")
            self.shutdown()


    def shutdown(self):
        """Shutdown the UI and release resources."""
        if self._data_streamer:
            self._data_streamer.stop()
        print("🛑 Guidance UI shutdown complete.")

    


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
    parser.add_argument("--force-2d", action="store_true", default=False, help="Force use of 2D guidance widget")
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
        },
        input_source=options.source,
        frames_folder=options.frames_folder,
        screen_capture_window_title=options.capture_window_title,
        clarius_ip=options.clarius_ip,
        clarius_port=options.clarius_port,
        force_2d_guidance_widget=options.force_2d,
        window_size=options.window_size
    )

    ui = GuidanceUI(config)
    ui.initialize()
    ui.run()

if __name__ == "__main__":
    main()