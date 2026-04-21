import os
import time
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import threading
import inference.landmark.landmark_prediction as lp
# Import fourch_landmark pipeline
from inference.fourch_landmark.fourch_landmark_pipeline import configure_fourch_landmark_pipeline, run_fourch_core_prediction
import inference.pose_classification.pose_classification as pc
from inference.pose_classification.pose_classification import PoseClassifierResultPooler
import ui.image_helpers as ih
from ui.guidance_widget import create_guidance_widget_renderer
from ui.fourch_landmark_renderer import FourChamberLandmarkRenderer
import inference.inference_helper as ihp
from inference.rotation.rotation_predictor import create_rotation_predictor
from inference.lvef.frame_buffer import FrameBuffer
from inference.lvef.lvef_prediction import LVEFPredictor
from common.application_settings import ApplicationSettings
import requests
from queue import Queue
from common.add_clarius_lib_path import setup_clarius_environment
from data_access.screen_capture.screen_capture import ScreenCapture

# Conditionally import ClariusStreamer
try:
    # Setup Clarius environment first
    clarius_available, clarius_error = setup_clarius_environment()
    if clarius_available:
        import data_access.clarius.clarius_streamer as ClariusStreamer
    else:
        print(f"Warning: Clarius not available - {clarius_error}")
        ClariusStreamer = None
except Exception as e:
    print(f"Warning: Failed to setup Clarius environment - {e}")
    ClariusStreamer = None

# define a queue
classifier_queue = Queue()

def run_event_loop(options, config):
  app_settings = ApplicationSettings() # settings for realtime controls

  # Check source and initialize 
  if options.source == 'clarius':
    if ClariusStreamer is None:
      print("❌ Error: Clarius SDK is not available. Cannot connect to Clarius probe.")
      print("   Please ensure the Clarius SDK is properly installed.")
      return
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

  if options.guidance_window:
    # render guidance widget in an additional window
    root_guidance = tk.Toplevel(root)
    root_guidance.title("Guidance")
    # Set the size and position of the "Guidance" window
    # to be on the right of the Result Window
    guidance_width = app_settings.input_window_bbox['width']
    guidance_height = app_settings.input_window_bbox['height'] - 28 # Adjust height to fit the title bar
    guidance_left = app_settings.input_window_bbox['left'] + app_settings.input_window_bbox['width'] * 2  # Place it to the right of the "Result" window
    guidance_top = app_settings.input_window_bbox['top']
    root_guidance.geometry(f"{guidance_width}x{guidance_height}+{guidance_left}+{guidance_top}")
    # Set background color to black
    root_guidance.configure(bg='black')
    # Create a label to show the image with a black background
    label_guidance = tk.Label(root_guidance, bg='black')
    label_guidance.pack(expand=True, fill='both')



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
      if ClariusStreamer is not None:
        ClariusStreamer.set_print_debug_info(app_settings.print_debug_info)
      print("Keyboard Event: [d] pressed. Toggling debug info to ", app_settings.print_debug_info)

  # Bind key events to the root window
  root.bind("<Key>", handle_key)

  if options.save and not os.path.exists(options.save_folder):
    os.makedirs(options.save_folder)

  # Initialize Inference Modules
  device = ihp.get_device()
  print(f"Using device: {device}")

  # Initialize 4-chamber landmark pipeline
  # Configure number of landmarks per chamber (optional - defaults shown)
  landmark_pipeline = configure_fourch_landmark_pipeline(
    weight_path=config["fourch_landmark_weight_path"],
    device=device,
    landmarks_per_chamber={
        50: 24,   # Left Ventricle (LV)
        100: 12,  # Right Ventricle (RV)
        150: 12,  # Left Atrium (LA)
        200: 12,  # Right Atrium (RA)
    }

  )
  # landmark_model = None
  weight_path = ""
  if options.imglmk:
    weight_path = config["pose_classifier_imglmk_weight_path"]
  else:
    weight_path = config["pose_classifier_img_weight_path"]
  classifier_model, preprocess = pc.load_pretrained_model(device, weight_path=weight_path, use_landmarks_for_estimation=options.imglmk)
  classifier_result_pooler = PoseClassifierResultPooler(pool_size=15, min_size=1)

  # Initialize the LVEF predictor
  print("Initializing LVEF predictor...")
  lvef_predictor = LVEFPredictor(device, config["lvef_weight_path"])  
  
  # Initialize rotation predictor (hardcoded paths)
  rotation_predictor = None
  last_predicted_rotation_matrix = None  # Store last prediction
  try:
    print("Initializing rotation predictor...")
    rotation_config_path = config["rotation_config_path"]  # Hardcoded config path
    rotation_checkpoint_path = config["rotation_checkpoint_path"]  # Hardcoded checkpoint path
    print(f"🔧 Loading weights from: {rotation_checkpoint_path}")
    rotation_predictor = create_rotation_predictor(
      rotation_config_path, 
      rotation_checkpoint_path, 
      device
    )
  except Exception as e:
    print(f"⚠️ Rotation predictor not available: {e}")
    rotation_predictor = None

  crnt_classifier_result = (255, 255, 255)
  crnt_classifier_result_pooled = (255, 255, 255)

  # Initialize the frame buffer
  print("Initializing frame buffer...")
  frame_buffer = FrameBuffer(buffer_size=100)
  
  # Initialize 4-chamber landmark renderer
  print("Initializing 4-chamber landmark renderer...")
  landmark_renderer = FourChamberLandmarkRenderer(
    landmark_radius=app_settings.landmark_radius,
    min_opacity=0.6,
    max_opacity=1.0,
    uncertainty_threshold=6.0,
    reference_size=None,  # Landmarks are already in crop_frame coordinates
    color_map={
            50: ih.rgb_to_bgr((180, 100, 255)),      # Left Ventricle (LV) - Bright Pink/Magenta (RGB: 255,100,180)
            100: ih.rgb_to_bgr((50, 200, 200)),      # Right Ventricle (RV) - Bright Yellow (RGB: 200,200,50)
            150: ih.rgb_to_bgr((150, 255, 100)),     # Left Atrium (LA) - Bright Green (RGB: 100,255,150)
            200: ih.rgb_to_bgr((100, 150, 255)),     # Right Atrium (RA) - Bright Orange (RGB: 255,150,100)
    },
    color_legend_png_path="assets/4ch-legend_static_v1.png"  # Overlay legend in top-left corner
  )

  # Initialize probe renderer (loads probe once for efficiency)
  print("Initializing probe renderer...")

  # Initialize probe renderer with automatic 3D/2D selection and fallback
  probe_renderer = create_guidance_widget_renderer(
    probe_path=config["probe_mesh_path"],
    renderer_type='auto',  # Try 3D first, fallback to 2D
    fallback_to_2d=True,
    # Common parameters (work for both 2D and 3D)
    background_color=(0, 0, 0),  # Black background
    render_mode='fullscreen',
    window_size=(256, 256),
    # 2D-specific parameters (ignored by 3D renderer)
    predicted_axis_opacity=0.6,  # 60% opacity for predicted axes
    axis_thickness=3,
    axis_labels=('x', 'y', 'z'),  
    lod_level=0.1,  # Level of detail
    probe_scale=0.7,  # Probe size relative to axes (70% of axis length)
    rotation_amplification=5.0,  # Amplification factor for rotation prediction
    # 3D-specific parameters (ignored by 2D renderer)
    probe_units='mm',
    wireframe=False
  )
  


  # Initialize lvef thread runner
  lvef_thread = None
  lvef_result = None

  def run_lvef_prediction():
    nonlocal lvef_result
    nonlocal lvef_predictor
    result = lvef_predictor.run_prediction()

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
        if ClariusStreamer is not None:
          frame_data = ClariusStreamer.get_frame()
        else:
          print("Error: ClariusStreamer is not available")
          break

      if frame_data is None:
        continue

      image = frame_data.image
      
      # Get quaternion data - use IMU data if available, otherwise use default values
      if options.source == 'clarius' and frame_data.imu is not None:
        imu = frame_data.imu
        qw, qx, qy, qz = imu.qw, imu.qx, imu.qy, imu.qz
      else:
        # Default quaternion values (no rotation)
        qw, qx, qy, qz = 1.0, 1.0, 0.0, 0.0
      # print(f"Quaternion: qw={qw:.4f}, qx={qx:.4f}, qy={qy:.4f}, qz={qz:.4f}")
      
      
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

      # All pipelines handle their own resizing internally:
      # - 4-chamber landmark: 512x512 (handled in pipeline)
      # - Rotation predictor: 256x256 (handled in pipeline)
      # - Pose classifier: 256x256 (handled in pipeline)
        
      # ---- Optional: Save frames ----
      # Save the screenshot without prediction to disk
      if options.save:
        crnt_timestamp = int(time.time() * 1000)
        filename = os.path.join(options.save_folder, f"{crnt_timestamp}.png")
        cv2.imwrite(filename, crop_frame)
        print(f"✅ Saved: {filename}")

      # Run 4-chamber landmark prediction on crop_frame (pipeline handles 512x512 resize internally)
      cla, landmarks_tensor = run_fourch_core_prediction(landmark_pipeline, crop_frame)

      # ---- Run Rotation Prediction ----
      if rotation_predictor is not None and frame_count % 1 == 0: 
        new_prediction = rotation_predictor.predict_rotation(crop_frame)
        if new_prediction is not None:
          last_predicted_rotation_matrix = new_prediction

      # ---- Run Pose Classification ----
      if frame_count % 3 == 0:
        if app_settings.use_threshold_for_pose_classification:
          crnt_classifier_result = cla
        else:
          crnt_classifier_result = pc.run_pose_classifier(classifier_model, device, crop_frame, landmarks_tensor, preprocess, use_landmarks_for_estimation=options.imglmk)
          
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
      # Landmarks are already in crop_frame coordinates, so no scaling needed
      # Update renderer with current landmark radius from app_settings
      landmark_renderer.set_landmark_radius(app_settings.landmark_radius)
      vis_frame = landmark_renderer.overlay_landmarks(vis_frame, landmarks_tensor, reference_size=None)
      
   
      widget_display = probe_renderer.render_widget(np.zeros_like(vis_frame), qw, qx, qy, qz, predicted_rotation_matrix=last_predicted_rotation_matrix)
     
      if app_settings.show_border:
        vis_frame = ih.add_rgb_border_to_result(vis_frame, ih.rgb_to_bgr(crnt_classifier_result_pooled))
      
      # Convert crop_frame to RGB for frame buffer
      crop_frame_rgb = cv2.cvtColor(crop_frame, cv2.COLOR_GRAY2RGB)
      # add frame to the buffer (pipeline will handle resizing if needed)
      # lvef_predictor.add_frame(crop_frame_rgb, crnt_classifier_result_pooled)
      lvef_predictor.add_frame_to_buffer(crop_frame_rgb, crnt_classifier_result_pooled)
      # print("Frame buffer size: ", frame_buffer.get_number_of_frames(), " is ok: ", frame_buffer.is_video_good())


      # Start the LVEF prediction thread if not already running
      if app_settings.run_lvef and lvef_predictor.is_buffer_video_good():
        # with lvef_lock:
          if lvef_thread is None or not lvef_thread.is_alive():
            lvef_thread = threading.Thread(target=run_lvef_prediction, daemon=True)
            lvef_thread.start()

      if lvef_result is not None:
        # Draw the LVEF prediction on the frame
        vis_frame = ih.draw_text_on_frame(vis_frame, "LVEF: " + str(lvef_result) + "%")

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

        if options.guidance_window:
          # get the curent geometry of the guidance window
          guidance_geometry = root_guidance.geometry()
          guidance_width = int(guidance_geometry.split('x')[0])
          guidance_height = int(guidance_geometry.split('x')[1].split('+')[0])

          # get the minimum width and height of the guidance window
          frame_width = min(guidance_width, guidance_height)
          app_settings.guidance_frame_dimensions = (frame_width, frame_width)

          # similarly update the guidance window
          guidance_frame = cv2.resize(widget_display, app_settings.guidance_frame_dimensions, interpolation=cv2.INTER_AREA)
          img_guidance = Image.fromarray(guidance_frame)
          img_tk_guidance = ImageTk.PhotoImage(img_guidance)
          label_guidance.config(image=img_tk_guidance)
          label_guidance.image = img_tk_guidance
          root_guidance.update_idletasks()
          root_guidance.update()
      except tk.TclError:
        print("🛑 Tkinter window has been destroyed. Exiting capture loop.")
        break


      if app_settings.run_image_upload:
        print("Sending current frame...")
        # immediately turn off the flag to avoid senting multiple frames
        app_settings.run_image_upload = False
        try:
          _, encoded_img = cv2.imencode('.png', crop_frame)
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
    if options.source == 'clarius' and ClariusStreamer is not None:
      ClariusStreamer.stop()
    else:
      screen_capture.stop()

  finally:
    print(f"📂 {frame_count} frames captured.")


def main():
  import argparse
  from dotenv import load_dotenv
  load_dotenv()

  parser = argparse.ArgumentParser(description='Run cardiac ultrasound guidance inference.')
  parser.add_argument('--source', type=str, default='screen', help='Source of the frames (screen | clarius | frames).')
  parser.add_argument('--clarius_ip', type=str, default='192.168.1.1', help='IP address of the Clarius probe.')
  parser.add_argument('--clarius_port', type=int, default=5828, help='Port of the Clarius probe.')
  parser.add_argument('--guidance_window', action='store_true', help='Enable guidance widget rendering window.')
  parser.add_argument('--save', type=str, default=None, metavar='PATH', help='Save frames as images to the specified folder.')
  parser.add_argument('--window_title', type=str, default="SM-T860", help='Title of the window to capture.')
  parser.add_argument('--upload_url', type=str, default="http://localhost:8000/upload", help='URL to upload the frame to.')
  parser.add_argument('--imglmk', default=False, action='store_true', help='Use image + landmarks for pose classification.')
  options = parser.parse_args()

  env_config = {
    "landmark_weight_path": os.getenv("LANDMARK_WEIGHTS", "weights/landmark_v1.pth"),
    "pose_classifier_imglmk_weight_path": os.getenv("POSE_EST_IMGLMK_WEIGHTS", "weights/pose-est-imglmk_v1.pth"),
    "pose_classifier_img_weight_path": os.getenv("POSE_EST_IMG_WEIGHTS", "weights/pose-est-img_v1.pth"),
    "lvef_weight_path": os.getenv("LVEF_WEIGHTS", "weights/echonet_v1.pt"),
    "rotation_config_path": os.getenv("ROTATION_PREDICTION_CONFIG", "src/inference/rotation/config.json"),
    "rotation_checkpoint_path": os.getenv("ROTATION_PREDICTION_WEIGHTS", "weights/rotation-prediction_v1.pt"),
    "probe_mesh_path": os.getenv("PROBE_MESH_PATH", "assets/transducer.stl"),
    "fourch_landmark_weight_path": os.getenv("4CH_WEIGHTS", "weights/4ch-seg_v2.pt"),
    "landmark_color_legend_png_path": os.getenv("LANDMARK_COLOR_LEGEND_PNG_PATH", "assets/4ch-legend_static_v1.png"),
    "guidance_widget_legend_png_path": os.getenv("GUIDANCE_WIDGET_LEGEND_PNG_PATH", "assets/guidance_widget_legend_static_v1.png")
  }

  run_event_loop(options, env_config)


if __name__ == "__main__":
  main()
