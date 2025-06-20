import data_access.clarius.clarius_streamer as ClariusStreamer
import argparse
import time
import os
import numpy as np
import cv2
import sys
import termios
import tty
import select


def save_result(output_data, output_dir):
  output_data.sort(key=lambda x: x.ts)

  # get the first ts as folder name
  if not output_data:
    print("-- No data to save.")
    return
  
  # a time-of-the-day timestamp (without dates) in nano second
  # 14 digits, padded with zeros
  first_timestamp = f"{output_data[0].timestamp:014d}"
  folder_name = os.path.join(output_dir, first_timestamp)

  print(f"-- Saving data to folder: {folder_name}")
  os.makedirs(folder_name, exist_ok=True)

  # for each frame, save the image with timestamp
  for i, frame in enumerate(output_data):
    image_path = os.path.join(folder_name, f"frame_{i:04d}_{frame.timestamp:014d}.png")

    if frame.image is not None:
      orig_frame = np.array(frame.image)
      orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
      cv2.imwrite(image_path, orig_gray)

  # create a csv file for imu data
  imu_file_path = os.path.join(folder_name, "frame.imu.csv")
  with open(imu_file_path, 'w') as imu_file:
    imu_file.write("timestamp,gx,gy,gz,ax,ay,az,mx,my,mz,qw,qx,qy,qz\n")
    for frame in output_data:
      # print(frame.imu)
      if frame.imu is not None:
        imu_file.write(f"{frame.timestamp},{frame.imu.gx},{frame.imu.gy},{frame.imu.gz},{frame.imu.ax},{frame.imu.ay},{frame.imu.az},{frame.imu.mx},{frame.imu.my},{frame.imu.mz},{frame.imu.qw},{frame.imu.qx},{frame.imu.qy},{frame.imu.qz}\n")
  
  print(f"-- Data saved to {folder_name} with {len(output_data)} frames.")


def main():
  parser = argparse.ArgumentParser(description="Record clarius data with imu")
  parser.add_argument("--ip", type=str, default="192.168.1.1", help="IP address of the Clarius probe")
  parser.add_argument("--port", type=int, default=5828, help="Port of the Clarius probe")
  parser.add_argument("--outputdir", type=str, help="Output directory for recorded data")
  parser.add_argument("--duration", type=int, default=10, help="Duration to record in seconds")
  parser.add_argument("--interval", type=int, default=50, help="Interval in ms between frames")

  options = parser.parse_args()

  print("-- Initializing Clarius Streamer with IP: {0}, Port: {1}".format(options.ip, options.port))
  ClariusStreamer.configure(options.ip, options.port, 'research', 512, 512)
  ClariusStreamer.start()

  is_recording = False
  output_data = []

  while True:
    def get_key():
      fd = sys.stdin.fileno()
      old_settings = termios.tcgetattr(fd)
      try:
        tty.setraw(fd)
        [i, _, _] = select.select([sys.stdin], [], [], 0.1)
        if i:
          ch = sys.stdin.read(1)
        else:
          ch = ''
      finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
      return ch

    if not is_recording:
      print("Press 'space' to record, 'q' to quit:")
      key = ''
      while not key:
        key = get_key()
      if key == ' ':
        print("-- Recording for {0} seconds...".format(options.duration))
        start_time = time.time()
        is_recording = True
      elif key.lower() == 'q':
        print("-- Quitting...")
        ClariusStreamer.stop()
        break
      else:
        print("-- Invalid input. Please try again.")
        continue

    elapsed_time = time.time() - start_time
    if elapsed_time >= options.duration:
      print(f"-- Recording finished. Elapsed time: {elapsed_time:.2f} seconds.")
      print(f"-- Saving {len(output_data)} frames to folder...")

      save_result(output_data, options.outputdir)

      output_data = []
      
      is_recording = False
      continue

    output_data.append(ClariusStreamer.get_frame())

    # sleep for the specified interval
    time.sleep(options.interval / 1000.0)


if __name__ == "__main__":
  main()