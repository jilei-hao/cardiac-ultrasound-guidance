#!/usr/bin/env python3
"""
Simple test for the AbstractImageWindow system using only basic functionality.
This test focuses on the macOS Tkinter implementation.
"""

import sys
import os
import time
import numpy as np
from dotenv import load_dotenv
import cv2
from PIL import Image



load_dotenv()  # Load environment variables from .env file

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Fix imports - remove 'src.' prefix since we added src to the path
from inference.fourch_landmark.fourch_landmark_pipeline import run_fourch_core_prediction
from ui.image_window import create_dual_window_setup
from data_access.simulator import FrameDataSimulator
from inference.fourch_landmark.fourch_landmark_pipeline import configure_fourch_landmark_pipeline, run_fourch_core_prediction
from inference.rotation.rotation_predictor import create_rotation_predictor
from inference import inference_helper as ihp
from ui.guidance_widget_renderer import GuidanceWidgetRenderer
from inference.pose_classification import pose_classification
from inference.pose_classification.pose_classification import PoseClassifierResultPooler
from ui.fourch_landmark_renderer import FourChamberLandmarkRenderer
import ui.image_helpers as ih

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


        _model, _preprocess = pose_classification.load_pretrained_model(self.__device, weight_path=os.getenv("POSE_EST_WEIGHTS"))
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


    def render_probe(self, rotation_quaternion):
        self.__probe_renderer.render(
            rotation_quaternion=rotation_quaternion
        )

def test_guidance_2d(imu_frames_folder):
    """Test guidance 2D functionality with dual window setup."""
    print("=== Guidance 2D Test ===")

    
    # Create dual window setup
    main_window, guidance_window = create_dual_window_setup(
        main_title="Main View",
        guidance_title="Guidance View",
        width=512,
        height=512,
        x_pos=100,
        y_pos=100
    )
    
    # Set up key handlers for both windows
    def handle_quit():
        print("Quit key pressed!")
        main_window.destroy()
        guidance_window.destroy()
    
    main_window.register_key_handler('q', handle_quit)
    guidance_window.register_key_handler('q', handle_quit)
    
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
    
    # Create simulator with the provided frame folder
    simulator = FrameDataSimulator(
        frames_folder=imu_frames_folder
    )

    inference_runner = InferenceRunner()

    probe_renderer = GuidanceWidgetRenderer(
        probe_path=os.getenv("PROBE_MESH_PATH"),
        background_color=(0, 0, 0),  # Black background
        predicted_axis_opacity=0.6,  # 60% opacity for predicted axes
        axis_thickness=3,  # 
        # axis_colors=((255, 100, 100), (100, 255, 100), (100, 100, 255)),  # Bright RGB colors
        axis_labels=('x', 'y', 'z'),  
        lod_level=0.1,  # Level of detail
        probe_scale=0.7,  # Probe size relative to axes (70% of axis length)
        rotation_amplification=5.0,  # Amplification factor for rotation prediction
        render_mode='fullscreen',  #
        window_size=(256, 256)  # 
    )

    landmark_renderer = FourChamberLandmarkRenderer(
        landmark_radius=8,
        min_opacity=0.6,
        max_opacity=1.0,
        uncertainty_threshold=6.0,
        reference_size=None  # Landmarks are already in crop_frame coordinates
    )
    
    try:
        simulator.start()
        
        frame_count = 0
        print("Starting display loop. Press 'q' to quit, or close either window.")
        
        while main_window.is_window_active() and guidance_window.is_window_active():
            # Get frame from simulator
            frame_data = simulator.get_frame()
            
            if frame_data.image is None:
                # End of sequence, reset
                simulator.reset()
                continue

            imu = frame_data.imu
            qw, qx, qy, qz = imu.qw, imu.qx, imu.qy, imu.qz
            
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

            cla, landmarks_tensor, rotation_prediction, pose_classification_prediction = inference_runner.run_inference(orig_gray)

            # ---------------------------------------
            # render ultrasound window
            # ---------------------------------------
            result_frame = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB)
            result_frame = landmark_renderer.overlay_landmarks(
                result_frame,
                landmarks_tensor,
                reference_size=None
            )
            result_frame = ih.add_rgb_border_to_result(result_frame, pose_classification_prediction)




            # --------------------------------------
            # render guidance window
            # --------------------------------------

            widget_display = probe_renderer.render_widget(
                np.zeros_like(orig_gray),
                qw, qx, qy, qz,
                predicted_rotation_matrix=rotation_prediction
            )

            # Convert widget_display to numpy array for Linux OpenCV windows
            # No need for ImageTk.PhotoImage conversion in Linux
            if isinstance(widget_display, np.ndarray):
                guidance_img_2d = widget_display
            else:
                # If it's a PIL Image, convert to numpy array
                guidance_img_2d = np.array(widget_display)

            # Ensure the image has the right format (RGB for display)
            if len(guidance_img_2d.shape) == 3 and guidance_img_2d.shape[2] == 4:
                # Convert RGBA to RGB
                guidance_img_2d = cv2.cvtColor(guidance_img_2d, cv2.COLOR_RGBA2RGB)
            elif len(guidance_img_2d.shape) == 2:
                # Convert grayscale to RGB
                guidance_img_2d = cv2.cvtColor(guidance_img_2d, cv2.COLOR_GRAY2RGB)

            # ==========================================
            main_frame = result_frame.copy()
            
            guidance_frame = guidance_img_2d.copy()

            
            # Display frames in both windows
            if not main_window.display_frame(main_frame):
                print("❌ Failed to display main frame")
                break
                
            if not guidance_window.display_frame(guidance_frame):
                print("❌ Failed to display guidance frame")
                break
            
            # Update both windows
            if not main_window.update() or not guidance_window.update():
                print("🛑 One or both windows closed")
                break
            
            frame_count += 1
            
            # Limit to ~10 FPS for testing
            time.sleep(0.1)
            
            # Show progress every 10 frames
            if frame_count % 10 == 0:
                print(f"Displayed {frame_count} frames")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulator.stop()
        main_window.destroy()
        guidance_window.destroy()
        print("✅ Test completed")



def main():
    """Run Guidance 2D tests."""
    print("Guidance 2D Tests")
    print("=" * 40)

    # take input frame_imu folder path from the argument
    imu_frames_folder = sys.argv[1] if len(sys.argv) > 1 else None
    if imu_frames_folder is None:
        raise ValueError("Please provide the frame_imu folder path as an argument.")
    
    print(f"Using frame_imu folder: {imu_frames_folder}")

    
    try:
        # Test guidance 2D functionality
        test_guidance_2d(imu_frames_folder)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Basic tests completed!")


if __name__ == "__main__":
    main()