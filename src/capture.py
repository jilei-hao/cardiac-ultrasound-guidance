import os
import time
import mss
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import threading
import macos_window_helpers as mwh
import landmark_prediction as lp
import pose_classification as pc
import image_helpers as ih
from frame_buffer import FrameBuffer
from lvef_prediction import LVEFPredictor
from application_settings import ApplicationSettings
import requests
from queue import Queue

# define a queue
classifier_queue = Queue()

def run_capture(options):
  app_settings = ApplicationSettings()

  monitor = mwh.get_window_bbox(options.window_title)

  if not monitor:
    print(f" 🟡 Warning: Window '{options.window_title}' not found.")
    monitor = {"top": 60, "left": 45, "width": 450, "height": 650}

  print(f"🎥 Capturing frames...")

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

  # Define the key event handler
  def handle_key(event):
    if event.keysym == "e":  # run lvef prediction
      app_settings.run_lvef = not app_settings.run_lvef
      print("Keyboard Event: [e] pressed. Toggling LVEF prediction to ", app_settings.run_lvef)
    elif event.keysym == "f":  # print frame rate
      app_settings.print_fps = not app_settings.print_fps
      print("Keyboard Event: [f] pressed. Toggling FPS printing to ", app_settings.print_fps)
    elif event.keysym == "t": # toggle landmarks for pose classification
      app_settings.use_threshold_for_pose_classification = not app_settings.use_threshold_for_pose_classification
      print("Keyboard Event: [t] pressed. Toggling thresholding for pose classification to ", app_settings.use_threshold_for_pose_classification)
    elif event.keysym == "u":  # Upload the current frame
      app_settings.run_image_upload = not app_settings.run_image_upload
      print("Keyboard Event: [u] pressed. Toggling image upload to ", app_settings.run_image_upload)
    elif event.keysym == "Up":  # Increase the radius of the landmarks
      app_settings.increment_landmark_radius()
      print("Keyboard Event: [Up] pressed. Increasing landmark radius to ", app_settings.landmark_radius)
    elif event.keysym == "Down":  # Decrease the radius of the landmarks
      app_settings.decrement_landmark_radius()
      print("Keyboard Event: [Down] pressed. Decreasing landmark radius to ", app_settings.landmark_radius)
    elif event.keysym == "Right":  # Increase the alpha of the landmarks
      app_settings.increment_landmark_alpha()
      print("Keyboard Event: [Right] pressed. Increasing landmark alpha to ", app_settings.landmark_alpha)
    elif event.keysym == "Left":  # Decrease the alpha of the landmarks
      app_settings.decrement_landmark_alpha()
      print("Keyboard Event: [Left] pressed. Decreasing landmark alpha to ", app_settings.landmark_alpha)

  # Bind key events to the root window
  root.bind("<Key>", handle_key)

  if options.save and not os.path.exists(options.save_folder):
    os.makedirs(options.save_folder)

  landmark_model, device = lp.configure_torch()
  # TODO: flag not updating in real-time
  # should we reload the model every time the flag is toggled? or can we just pass in True or False for all senarios?
  classifier_model, preprocess = pc.load_pretrained_model(device) 
  
  # Initialize the LVEF predictor
  lvef_predictor = LVEFPredictor(device)  

  crnt_classifier_result = (255, 255, 255)
  crnt_classifier_result_pooled = (255, 255, 255)

  # Initialize the frame buffer
  frame_buffer = FrameBuffer(buffer_size=100)

  # Initialize lvef thread runner
  lvef_thread = None
  lvef_result = None


  def run_lvef_prediction():
    nonlocal lvef_result
    nonlocal lvef_predictor
    nonlocal frame_buffer
    video = frame_buffer.get_all_frames()
    result = lvef_predictor.run_prediction(video)

    # with lvef_lock:
    lvef_result = result 
    app_settings.run_lvef = False # only run once
    # print("LVEF prediction: ", lvef_result)


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
         
        # ---- Optional: Save frames ----
        # Save the screenshot without prediction to disk
        if options.save:
          crnt_timestamp = int(time.time() * 1000)
          filename = os.path.join(options.save_folder, f"{crnt_timestamp}.png")
          cv2.imwrite(filename, resized_frame)
          print(f"✅ Saved: {filename}")

        cla, landmarks_tensor = lp.run_core_prediction(landmark_model, device, resized_frame)

        # ---- Run Pose Classification ----
        if frame_count % 3 == 0:
          if app_settings.use_threshold_for_pose_classification:
            crnt_classifier_result = cla
          else:
            crnt_classifier_result = pc.run_pose_classifier(classifier_model, device, resized_frame, landmarks_tensor, preprocess)
            
        # push the classifier result to the queue
        classifier_queue.put(crnt_classifier_result)
        # only run the following code if the queue is at least 5 elemenets long
        sliding_window_size = 15
        if classifier_queue.qsize() > sliding_window_size:
          # pop the oldest result from the queue
          classifier_queue.get()
          # print result from the queue
          # print(f"🗃️ Classifier Queue: {list(classifier_queue.queue)}")
          # find the most common result in the queue
          most_common_result = None
          if not classifier_queue.empty():
            most_common_result = max(set(list(classifier_queue.queue)), key=list(classifier_queue.queue).count)
            crnt_classifier_result_pooled = most_common_result
          else:
            crnt_classifier_result_pooled = crnt_classifier_result
          

        vis_frame = cv2.cvtColor(crop_frame, cv2.COLOR_GRAY2BGR)
        vis_frame = ih.overlay_landmarks(vis_frame, landmarks_tensor, app_settings, ere_threshold=6)
        vis_frame = ih.add_rgb_border_to_result(vis_frame, crnt_classifier_result_pooled)
        
        # stack resized frames as RGB 
        resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
        # add resized frame to the buffer
        frame_buffer.add_frame(resized_frame_rgb, crnt_classifier_result_pooled)
        # print("Frame buffer size: ", frame_buffer.get_number_of_frames(), " is ok: ", frame_buffer.is_video_good())


        # Start the LVEF prediction thread if not already running
        if app_settings.run_lvef and frame_buffer.is_video_good():
          # with lvef_lock:
            if lvef_thread is None or not lvef_thread.is_alive():
              lvef_thread = threading.Thread(target=run_lvef_prediction, daemon=True)
              lvef_thread.start()

        if lvef_result is not None:
          # Draw the LVEF prediction on the frame
          vis_frame = ih.draw_text_on_frame(vis_frame, "LVEF: " + str(lvef_result.round(1)) + "%")

        try:
          # get the current geometry of the window
          window_geometry = root.geometry()
          window_width = int(window_geometry.split('x')[0])
          window_height = int(window_geometry.split('x')[1].split('+')[0])
          app_settings.result_window_dimensions = (window_width, window_height)
          
          # get the minimum width and height of the window
          frame_width = min(window_width, window_height)
          app_settings.frame_dimensions = (frame_width, frame_width)

          display_frame = cv2.resize(vis_frame, app_settings.frame_dimensions, interpolation=cv2.INTER_AREA)

          # Convert the frame to RGB format
          img = Image.fromarray(display_frame)
          img_tk = ImageTk.PhotoImage(img)

          # Update the label with the new image 
          label.config(image=img_tk)
          label.image = img_tk

          # Update the window to show the new image
          root.update_idletasks()
          root.update()
        except tk.TclError:
          print("🛑 Tkinter window has been destroyed. Exiting capture loop.")
          break


        if app_settings.run_image_upload:
          print("Sending current frame...")
          # immediately turn off the flag to avoid senting multiple frames
          app_settings.run_image_upload = False
          try:
            _, encoded_img = cv2.imencode('.png', resized_frame)
            response = requests.post(options.upload_url, files={
              "file": ("frame.png", encoded_img.tobytes(), "image/png")
            })
            print("Upload response:", response.json())
          except Exception as e:
            print("Upload failed:", str(e))

        frame_count += 1

        if app_settings.print_fps:
          elapsed_time = time.time() - start_time
          fps = frame_count / elapsed_time
          print(f"FPS: {fps:.2f}")

    except KeyboardInterrupt:
      print("\n🛑 Capture stopped manually.")

    finally:
      print(f"📂 {frame_count} frames captured.")



def main():
  import argparse
  parser = argparse.ArgumentParser(description='Capture screen frames.')
  parser.add_argument('--save', action='store_true', help='Save frames as images.')
  parser.add_argument('--save_folder', type=str, default="output", help='Folder to save images.')
  parser.add_argument('--window_title', type=str, default="SM-T860", help='Title of the window to capture.')
  parser.add_argument('--upload_url', type=str, default="http://localhost:8000/upload", help='URL to upload the frame to.')
  options = parser.parse_args()
  run_capture(options)


if __name__ == "__main__":
  main()
