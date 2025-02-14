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

def run_capture(options):
  # Get the bounding box of the specified window
  monitor = mwh.get_window_bbox(options.window_title)

  # Define screen region to capture (Adjust based on scrcpy window position)
  if not monitor:
    print(f" 🟡 Warning: Window with title '{options.window_title}' not found.")
    print(f" -- Using default monitor settings.")
    monitor = {
      "top": 60,  
      "left": 45, 
      "width": 450, 
      "height": 650,
    }

  print(f"🎥 Capturing frames every {options.frame_interval} seconds...")


  if options.render:
    # create a tkinter window
    root = tk.Tk()
    root.title("Capture")

    # create a label to show the image
    label = tk.Label(root)
    label.pack(expand=True, fill='both')


  if options.save and not os.path.exists(options.save_folder):
    os.makedirs(options.save_folder)


  with mss.mss() as sct:
  # get the initial time stamp
    try:
      start_time = time.time()
      timestamp = int(start_time * 1000)
      frame_count = 0

      landmark_model, device = lp.configure_torch()
      classifier_model, preprocess = pc.load_pretrained_model(device)

      crnt_classifier_result = (255, 255, 255)

      while True:
        # Capture screen
        screenshot = sct.grab(monitor)
        
        # Convert to OpenCV format
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)  # Convert from BGRA to Grayscale

        # crop to a largest square from the center
        h, w = frame.shape
        pad = int(min(h, w) * 0.1)
        if h > w:
          diff = (h - w) // 2
          frame = frame[diff + pad:diff+w - pad, pad : w - pad]
        elif w > h:
          diff = (w - h) // 2
          frame = frame[:, diff:diff+h]

        # resample to 112 x 112
        frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA)

        if options.save:
          # Save frame as PNG
          crnt_timestamp = int(time.time() * 1000)
          diff = crnt_timestamp - timestamp
          
          # filename print diff with 8 digits
          filename = os.path.join(options.save_folder, f"{diff:08d}.png")
          cv2.imwrite(filename, frame)
          print(f"✅ Saved: {filename}")



        ################################################################
        # Real-time Inference Goes Here
        ################################################################
        # result_frame = lp.predict_landmark(frame)

        result_frame, cla= lp.run_core_prediction(landmark_model, device, frame)
        

        if (frame_count % 3 == 0):
          crnt_classifier_result = pc.run_pose_classifier(classifier_model, device, frame, preprocess)
          crnt_classifier_result = cla
          # print(cla)
        
        result_frame = ih.add_rgb_frame_to_result(result_frame, crnt_classifier_result)

        print("classifier result: ", crnt_classifier_result)


        ################################################################
        # result_frame = frame
        # resample the result to 512 x 512
        result_frame = cv2.resize(result_frame, (512, 512), interpolation=cv2.INTER_AREA)


        if options.render:
          # convert to PIL image
          img = Image.fromarray(result_frame)
          img_tk = ImageTk.PhotoImage(img)

          # update the label
          label.config(image=img_tk)
          label.image = img_tk

          # update the tkinter window
          root.update_idletasks()
          root.update()


        frame_count += 1

        # Wait for the specified interval
        time.sleep(options.frame_interval)

        # Stop capturing if duration limit is set
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

  options = parser.parse_args()
  run_capture(options)


if __name__ == "__main__":
  main()
