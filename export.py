import os
import time
import mss
import numpy as np
import cv2

# Configuration
SAVE_FOLDER = "/Users/jileihao/dev/arpa-h-dev/lumify_streaming/output/250129/loop7"  # Folder to save images
FRAME_INTERVAL = 0.0  # Capture interval in seconds (e.g., 0.5s = 500ms)
DURATION = 12  # Set to None for infinite capture, or specify total seconds

# Ensure save directory exists
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Define screen region to capture (Adjust based on scrcpy window position)
monitor = {
    "top": 10,  # Adjust based on your screen
    "left": 0,  # Adjust based on your screen
    "width": 700,  # Set width of scrcpy window
    "height": 1000,  # Set height of scrcpy window
}

print(f"🎥 Capturing frames every {FRAME_INTERVAL} seconds...")

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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
            
            # Save frame as PNG
            crnt_timestamp = int(time.time() * 1000)
            diff = crnt_timestamp - timestamp
            
            # filename print diff with 8 digits
            filename = os.path.join(SAVE_FOLDER, f"{diff:08d}.png")
            cv2.imwrite(filename, frame)
            print(f"✅ Saved: {filename}")
            frame_count += 1

            # Wait for the specified interval
            time.sleep(FRAME_INTERVAL)

            # Stop capturing if duration limit is set
            if DURATION and (time.time() - start_time) > DURATION:
                break

    except KeyboardInterrupt:
        print("\n🛑 Capture stopped manually.")

    finally:
        print(f"📂 {frame_count} frames saved in '{SAVE_FOLDER}'.")
