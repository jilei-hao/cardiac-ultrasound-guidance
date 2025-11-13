from dataclasses import dataclass
from time import time

import cv2
import numpy as np
from inference.guidance_pipeline import GuidancePipelineConfig, GuidancePipeline
from data_access.screen_capture.screen_capture import ScreenCapture
from ui.image_window import create_dual_window_setup
from ui.guidance_widget.guidance_widget_renderer_2d import GuidanceWidgetRenderer2D
from ui.guidance_widget.guidance_widget_renderer_3d import GuidanceWidgetRenderer3D
from ui.fourch_landmark_renderer import FourChamberLandmarkRenderer
from data_access.simulator import FrameDataSimulator


@dataclass
class GuidanceUIConfig:
    env_config: dict
    input_source: str # simulator, clarius, screen
    force_2d_guidance_widget: bool = False
    frames_folder: str = None
    screen_capture_window_title: str = "Input"
    clarius_ip: str = None
    clarius_port: int = 5828



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
        self._logger = logger
    
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
            width=1024,
            height=1024,
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
                scale_factor=3.0,
                 wireframe=False,
                 window_size=(640, 480),
                 render_mode='fullscreen',
                 rotation_amplification=3.0,
                 legend_image_path = self._env_config["guidance_widget_legend_png_path"]
            )

        # Initialize the data streamer
        if self._config.input_source == "simulator":
            self._data_streamer = FrameDataSimulator(frames_folder=self._config.frames_folder)
        elif self._config.input_source == "clarius":
            import data_access.clarius as ClariusStreamer
            # To be implemented: Clarius data streamer
            ClariusStreamer.configure(
                self._config.clarius_ip,
                self._config.clarius_port,
                "research",
                512,
                512
            )
            self._data_streamer = ClariusStreamer
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
                orig_frame = np.array(frame_data.image)
                if orig_frame is None:
                    continue

                # Preprocess frame
                orig_gray_512 = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
                orig_gray_512 = cv2.resize(orig_gray_512, (512, 512))

                # Run inference
                cla, landmarks_tensor, rotation_prediction, pose_classification_prediction = self._guidance_pipeline.run_inference(orig_gray_512)


                # Prepare output frames
                out_main_frame = orig_frame.copy()
                out_guidance_frame = orig_frame.copy()

                # Display output frame
                if not self._main_window.display_frame_rgb(out_main_frame):
                    print("❌ Failed to display main frame")
                    break

                if not self._guidance_window.display_frame_rgb(out_guidance_frame):
                    print("❌ Failed to display guidance frame")
                    break
                
                # Update both windows
                if not self._main_window.update() or not self._guidance_window.update():
                    print("🛑 One or both windows closed")
                    break

                time.sleep(0.1)

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
        force_2d_guidance_widget=options.force_2d
    )

    ui = GuidanceUI(config)
    ui.initialize()
    ui.run()

if __name__ == "__main__":
    main()