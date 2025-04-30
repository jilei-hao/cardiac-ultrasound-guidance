import os
import time
import mss
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import macos_window_helpers as mwh
import landmark_prediction as lp
import pose_classification as pc
import image_helpers as ih
from frame_buffer import FrameBuffer

def run_capture(options):
    monitor = mwh.get_window_bbox(options.window_title)

    if not monitor:
        print(f" 🟡 Warning: Window '{options.window_title}' not found.")
        monitor = {"top": 60, "left": 45, "width": 450, "height": 650}

    print(f"🎥 Capturing frames every {options.frame_interval} seconds...")

    if options.render:
        root = tk.Tk()
        root.title("Results")

        # Set the size and position of the "Capture" window
        capture_width = monitor['width']
        capture_height = monitor['height'] - 28  # Adjust height to fit the title bar
        capture_left = monitor['left'] + monitor['width']  # Place it to the right of the "Monitor" window
        capture_top = monitor['top']
        root.geometry(f"{capture_width}x{capture_height}+{capture_left}+{capture_top}")
        
        # Set background color to black
        root.configure(bg='black')

        # Create a label to show the image with a black background
        label = tk.Label(root, bg='black')
        label.pack(expand=True, fill='both')

    if options.save and not os.path.exists(options.save_folder):
        os.makedirs(options.save_folder)

    landmark_model, device = lp.configure_torch()
    
    options.use_landmarks_for_pose = options.use_landmarks_for_pose.lower() == 'true' # grab flag for using landmarks for estimation
    print("options.use_landmarks_for_pose: ", options.use_landmarks_for_pose)
    classifier_model, preprocess = pc.load_pretrained_model(device, options.use_landmarks_for_pose)

    crnt_classifier_result = (255, 255, 255)

    # Initialize the frame buffer
    frame_buffer = FrameBuffer(buffer_size=100)


    with mss.mss() as sct:
        start_time = time.time()
        frame_count = 0

        try:
            # Main loop starts, everything inside should be efficient
            # Should achieve 30 FPS. Frame time (runtime for each iteration) should be less than 33ms
            while True:
                screenshot = sct.grab(monitor)
                orig_frame = np.array(screenshot)
                orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGRA2GRAY)

                h, w = orig_gray.shape
                pad = int(min(h, w) * 0.1)
                if h > w:
                    diff = (h - w) // 2
                    crop_frame = orig_gray[diff + pad:diff + w - pad, pad:w - pad]
                elif w > h:
                    diff = (w - h) // 2
                    crop_frame = orig_gray[:, diff:diff + h]
                else:
                    crop_frame = orig_gray[pad:h-pad, pad:w-pad]

                resized_frame = cv2.resize(crop_frame, (112, 112), interpolation=cv2.INTER_AREA)

                # push an RGB frame to the buffer
                resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
                frame_buffer.add_frame(resized_frame_rgb)

                # retrieve all frames from the buffer
                buffer_frames = frame_buffer.get_all_frames()
                buffer_frames = np.array(buffer_frames)
                

                if options.save:
                    crnt_timestamp = int(time.time() * 1000)
                    filename = os.path.join(options.save_folder, f"{crnt_timestamp}.png")
                    cv2.imwrite(filename, resized_frame)
                    print(f"✅ Saved: {filename}")

                result_frame, cla, landmarks_tensor = lp.run_core_prediction(landmark_model, device, resized_frame)

                if frame_count % 3 == 0:
                    crnt_classifier_result = pc.run_pose_classifier(classifier_model, device, resized_frame, landmarks_tensor, preprocess, options.use_landmarks_for_pose)

                vis_frame = cv2.cvtColor(crop_frame, cv2.COLOR_GRAY2BGR)
                vis_frame = lp.overlay_landmarks(vis_frame, landmarks_tensor)
                vis_frame = ih.add_rgb_frame_to_result(vis_frame, crnt_classifier_result)

                # get the current geometry of the window
                window_geometry = root.geometry()
                window_width = int(window_geometry.split('x')[0])
                window_height = int(window_geometry.split('x')[1].split('+')[0])
                # get the minimum width and height of the window
                frame_width = min(window_width, window_height)

                display_frame = cv2.resize(vis_frame, (frame_width, frame_width), interpolation=cv2.INTER_AREA)

                if options.render:
                    img = Image.fromarray(display_frame)
                    img_tk = ImageTk.PhotoImage(img)

                    label.config(image=img_tk)
                    label.image = img_tk

                    root.update_idletasks()
                    root.update()

                frame_count += 1
                time.sleep(options.frame_interval)

                if options.duration and (time.time() - start_time) > options.duration:
                    break

        except KeyboardInterrupt:
            print("\n🛑 Capture stopped manually.")

        finally:
            print(f"📂 {frame_count} frames captured.")



def main():
  import argparse
  parser = argparse.ArgumentParser(description='Capture screen frames.')
  parser.add_argument('--save', action='store_true', help='Save frames as images.')
  parser.add_argument('--render', action='store_true', help='Render frames in a window.')
  parser.add_argument('--save_folder', type=str, default="output", help='Folder to save images.')
  parser.add_argument('--frame_interval', type=float, default=0.05, help='Interval between frames in seconds.')
  parser.add_argument('--duration', type=float, default=None, help='Duration to capture in seconds. Set to None for infinite capture.')
  parser.add_argument('--window_title', type=str, default="SM-T860", help='Title of the window to capture.')
  parser.add_argument('--use_landmarks_for_pose', type=str, default='true',
                    help='Use landmarks for pose estimation: true or false (default: true)')
  options = parser.parse_args()
  run_capture(options)


if __name__ == "__main__":
  main()
