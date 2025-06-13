import data_access.clarius.clarius_streamer as ClariusStreamer
import argparse
import time
import os
import numpy as np
import cv2


def save_result(output_data, output_dir):
  # sort the data by timestamp
  output_data.sort(key=lambda x: x.timestamp)

  # get the first timestamp as folder name
  if not output_data:
    print("-- No data to save.")
    return
  
  first_timestamp = output_data[0].timestamp
  folder_name = f"{output_dir}/{first_timestamp}"
  print(f"-- Saving data to folder: {folder_name}")
  os.makedirs(folder_name, exist_ok=True)

  # for each frame, save the image with timestamp
  for i, frame in enumerate(output_data):
    image_path = os.path.join(folder_name, f"frame_{i}_{frame.timestamp}.png")

    if frame.image is not None:
      orig_frame = np.array(frame.image)
      orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
      cv2.imwrite(image_path, orig_gray)

  # create a csv file for imu data
  imu_file_path = os.path.join(folder_name, "imu_data.csv")
  with open(imu_file_path, 'w') as imu_file:
    imu_file.write("timestamp,gx,gy,gz,ax,ay,az,mx,my,mz,qw,qx,qy,qz\n")
    for frame in output_data:
      if frame.imu is not None:
        imu_data = frame.imu
        imu_file.write(f"{frame.timestamp},{imu_data.gx},{imu_data.gy},{imu_data.gz},"
                      f"{imu_data.ax},{imu_data.ay},{imu_data.az},"
                      f"{imu_data.mx},{imu_data.my},{imu_data.mz},"
                      f"{imu_data.qw},{imu_data.qx},{imu_data.qy},{imu_data.qz}\n")
  
  print(f"-- Data saved to {folder_name} with {len(output_data)} frames.")


def main():
  parser = argparse.ArgumentParser(description="Record clarius data with imu")
  parser.add_argument("--ip", type=str, default="192.168.1.1", help="IP address of the Clarius probe")
  parser.add_argument("--port", type=int, default=5828, help="Port of the Clarius probe")
  parser.add_argument("--outputdir", type=str, help="Output directory for recorded data")
  parser.add_argument("--duration", type=int, default=10, help="Duration to record in seconds")

  options = parser.parse_args()

  print("-- Initializing Clarius Streamer with IP: {0}, Port: {1}".format(options.ip, options.port))
  ClariusStreamer.configure(options.ip, options.port, options.outputdir, 512, 512)
  ClariusStreamer.start()

  is_recording = False
  output_data = []

  while True:
    if not is_recording:
      input_str = input("Press 'r' to record, 'q' to quit: ").strip().lower()
      if input_str == 'r':
        print("-- Recording for {0} seconds...".format(options.duration))
        start_time = time.time()
        is_recording = True
        print(f"-- Recording finished. Elapsed time: {elapsed_time:.2f} seconds.")
      elif input_str == 'q':
        print("-- Quitting...")
        ClariusStreamer.stop()
        break
      else:
        print("-- Invalid input. Please try again.")
        continue

    elapsed_time = time.time() - start_time
    if elapsed_time >= options.duration:
      print(f"-- Recording finished. Elapsed time: {elapsed_time:.2f} seconds.")
      print(f"-- Saving {0} frames to folder...".format(len(output_data)))
      
      is_recording = False
      continue

    output_data.push(ClariusStreamer.get_frame())


if __name__ == "__main__":
  main()