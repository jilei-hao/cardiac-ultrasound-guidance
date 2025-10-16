import os
import time
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import threading
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'inference'))
# Import with proper module name
import importlib.util
spec = importlib.util.spec_from_file_location("fourch_landmark_pipeline", 
    os.path.join(os.path.dirname(__file__), 'inference', '4ch-landmark', '4ch_landmark_pipeline.py'))
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
import inference.pose_classification.pose_classification as pc
import ui.image_helpers as ih
from common.frame_buffer import FrameBuffer
from inference.lvef.lvef_prediction import LVEFPredictor
from common.application_settings import ApplicationSettings
import requests
from queue import Queue
import data_access.clarius.clarius_streamer as ClariusStreamer
from data_access.screen_capture.screen_capture import ScreenCapture

# define a queue
classifier_queue = Queue()

def run_event_loop(options):
  app_settings = ApplicationSettings() # settings for realtime controls

  # Check source and initialize 
  if options.source == 'clarius':
    print(f"🔌 Connecting to Clarius probe at {options.clarius_ip}:{options.clarius_port}...")
    ClariusStreamer.configure(options.clarius_ip, options.clarius_port, "research")
    ClariusStreamer.set_print_debug_info(app_settings.print_debug_info)
    ClariusStreamer.start()
    print("✅ Connected to Clarius probe.")
  else:
    print("🖥️ Using screen capture as source.")
    screen_capture = ScreenCapture(options.window_title, app_settings)
    screen_capture.start()

  print(f"🎥 Capturing frames...")

  root = tk.Tk()
  root.title("Results")

  # Set the size and position of the "Result" window
  result_width = app_settings.input_window_bbox['width']
  result_height = app_settings.input_window_bbox['height'] - 28  # Adjust height to fit the title bar
  result_left = app_settings.input_window_bbox['left'] + app_settings.input_window_bbox['width']  # Place it to the right of the "scr_input_region" window
  result_top = app_settings.input_window_bbox['top']
  root.geometry(f"{result_width}x{result_height}+{result_left}+{result_top}")
  
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
    elif event.keysym == "b":  # Toggle the border
      app_settings.show_border = not app_settings.show_border
      print("Keyboard Event: [b] pressed. Toggling border to ", app_settings.show_border)
    elif event.keysym == "d":  # Toggle debug info
      app_settings.print_debug_info = not app_settings.print_debug_info
      ClariusStreamer.set_print_debug_info(app_settings.print_debug_info)
      print("Keyboard Event: [d] pressed. Toggling debug info to ", app_settings.print_debug_info)

  # Bind key events to the root window
  root.bind("<Key>", handle_key)

  if options.save and not os.path.exists(options.save_folder):
    os.makedirs(options.save_folder)

  landmark_pipeline = lp.configure_4ch_landmark_pipeline()
  device = landmark_pipeline.device
  classifier_model, preprocess = pc.load_pretrained_model(device) 
  
  # Initialize the LVEF predictor
  print("Initializing LVEF predictor...")
  lvef_predictor = LVEFPredictor(device)  

  crnt_classifier_result = (255, 255, 255)
  crnt_classifier_result_pooled = (255, 255, 255)

  # Initialize the frame buffer
  print("Initializing frame buffer...")
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

  print("Starting frame capture...")

  start_time = time.time()
  frame_count = 0

  # Main Event Loop
  try:
    # Should achieve 30 FPS. Frame time (runtime for each iteration) should be less than 33ms
    while True:
      if options.source == 'screen':
        frame_data = screen_capture.get_frame()
      elif options.source == 'clarius':
        frame_data = ClariusStreamer.get_frame()

      if frame_data is None:
        continue

      image = frame_data.image

      orig_frame = np.array(image)
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

      # Use 512x512 for 4-chamber segmentation
      resized_frame = cv2.resize(crop_frame, (512, 512), interpolation=cv2.INTER_AREA)
        
      # ---- Optional: Save frames ----
      # Save the screenshot without prediction to disk
      if options.save:
        crnt_timestamp = int(time.time() * 1000)
        filename = os.path.join(options.save_folder, f"{crnt_timestamp}.png")
        cv2.imwrite(filename, resized_frame)
        print(f"✅ Saved: {filename}")

      cla, landmarks_tensor = lp.run_4ch_core_prediction(landmark_pipeline, resized_frame)

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
      if app_settings.show_border:
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
        vis_frame = ih.draw_text_on_frame(vis_frame, "LVEF: " + str(round(lvef_result.item())) + "%")

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

      # Always print FPS
      elapsed_time = time.time() - start_time
      fps = frame_count / elapsed_time
      print(f"FPS: {fps:.2f}")
      
      # Also print FPS when the flag is enabled (for backward compatibility)
      if app_settings.print_fps:
        print(f"FPS (flag enabled): {fps:.2f}")

  except KeyboardInterrupt:
    print("\n🛑 Capture stopped manually.")
    if options.source == 'clarius':
      ClariusStreamer.stop()
    else:
      screen_capture.stop()

  finally:
    print(f"📂 {frame_count} frames captured.")


def main():
  import argparse
  parser = argparse.ArgumentParser(description='Capture screen frames.')
  parser.add_argument('--source', type=str, default='screen', help='Source of the frames (screen | clarius).')
  parser.add_argument('--clarius_ip', type=str, default='192.168.1.1', help='IP address of the Clarius probe.')
  parser.add_argument('--clarius_port', type=int, default=5828, help='Port of the Clarius probe.')
  parser.add_argument('--save', action='store_true', help='Save frames as images.')
  parser.add_argument('--save_folder', type=str, default="output", help='Folder to save images.')
  parser.add_argument('--window_title', type=str, default="SM-T860", help='Title of the window to capture.')
  parser.add_argument('--upload_url', type=str, default="http://localhost:8000/upload", help='URL to upload the frame to.')
  options = parser.parse_args()

  run_event_loop(options)


if __name__ == "__main__":
  main()
