import os
import time
import mss
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk

# Configuration
SAVE_FOLDER = "/Users/jileihao/dev/arpa-h-dev/lumify_streaming/output/250129/loop7"  # Folder to save images
FRAME_INTERVAL = 0.05  # Capture interval in seconds (e.g., 0.5s = 500ms)
DURATION = 3  # Set to None for infinite capture, or specify total seconds

# Ensure save directory exists
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Define screen region to capture (Adjust based on scrcpy window position)
monitor = {
    "top": 60,  # Adjust based on your screen
    "left": 45,  # Adjust based on your screen
    "width": 450,  # Set width of scrcpy window
    "height": 650,  # Set height of scrcpy window
}

print(f"🎥 Capturing frames every {FRAME_INTERVAL} seconds...")



# cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
# print("Press 's' to start capturing...")
# while True:
#   if cv2.waitKey(1) & 0xFF == ord('s'):
#     break

start_time = time.time()
frame_count = 0

with mss.mss() as sct:
  # get the initial time stamp
  timestamp = int(time.time() * 1000)
  try:
    while True:
      # Capture screen
      screenshot = sct.grab(monitor)
      
      # Convert to OpenCV format
      frame = np.array(screenshot)
      # print(frame)
      # print(frame.shape)

      frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)  # Convert from BGRA to Grayscale
      
      # Save frame as PNG
      crnt_timestamp = int(time.time() * 1000)
      diff = crnt_timestamp - timestamp
      
      # filename print diff with 8 digits
      filename = os.path.join(SAVE_FOLDER, f"{diff:08d}.png")
      cv2.imwrite(filename, frame)
      print(f"✅ Saved: {filename}")

      # render frame on screen
      # cv2.imshow("Screen Capture", frame)

      frame_count += 1

      # img = Image.fromarray(frame)
      # img.show()

      # Wait for the specified interval
      time.sleep(FRAME_INTERVAL)

      # Stop capturing if duration limit is set
      if DURATION and (time.time() - start_time) > DURATION:
        break

  except KeyboardInterrupt:
    print("\n🛑 Capture stopped manually.")

  finally:
    cv2.destroyAllWindows()
    print(f"📂 {frame_count} frames saved in '{SAVE_FOLDER}'.")
        
