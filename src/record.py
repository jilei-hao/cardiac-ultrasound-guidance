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

def get_timestamp(preserve_date: bool) -> int:
  """Get a timestamp in nanoseconds.

  Args:
      preserve_date: If True, use the full timestamp including date.
                     If False, use time-of-day only.
  Returns:
      int: Timestamp in nanoseconds.
  """
  if preserve_date:
    ts = int(time.time() * 1e9)
  else:
    local_time = time.localtime()
    ns_since_midnight = (local_time.tm_hour * 3600 + local_time.tm_min * 60 + local_time.tm_sec) * int(1e9)
    ts = ns_since_midnight + int((time.time() % 1) * 1e9)
  return ts

def save_result(output_data, output_dir):
  output_data.sort(key=lambda x: x['timestamp'])

  # get the first ts as folder name
  if not output_data:
    print("-- No data to save.")
    return

  print(f"-- Saving data to folder: {output_dir}")
  os.makedirs(output_dir, exist_ok=True)

  # for each frame, save the image with timestamp
  for i, crnt_record in enumerate(output_data):
    timestamp_str = f"{crnt_record['timestamp']:019d}"
    device_timestamp_str = f"{crnt_record['frame'].timestamp:014d}"
    
    image_path = os.path.join(output_dir, f"frame_{i:04d}_{timestamp_str}.png")

    if crnt_record['frame'].image is not None:
      orig_frame = np.array(crnt_record['frame'].image)
      orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
      cv2.imwrite(image_path, orig_gray)

  # create a csv file for imu data
  imu_file_path = os.path.join(output_dir, "frame.imu.csv")
  with open(imu_file_path, 'w') as imu_file:
    imu_file.write("timestamp,gx,gy,gz,ax,ay,az,mx,my,mz,qw,qx,qy,qz,device_timestamp\n")
    for crnt_record in output_data:
      device_timestamp_str = f"{crnt_record['frame'].timestamp:014d}"
      if crnt_record['frame'].imu is not None:
        imu_file.write(f"{crnt_record['frame'].timestamp},"
                       f"{crnt_record['frame'].imu.gx},{crnt_record['frame'].imu.gy},{crnt_record['frame'].imu.gz},"
                       f"{crnt_record['frame'].imu.ax},{crnt_record['frame'].imu.ay},{crnt_record['frame'].imu.az},"
                       f"{crnt_record['frame'].imu.mx},{crnt_record['frame'].imu.my},{crnt_record['frame'].imu.mz},"
                       f"{crnt_record['frame'].imu.qw},{crnt_record['frame'].imu.qx},{crnt_record['frame'].imu.qy},{crnt_record['frame'].imu.qz},"
                       f"{device_timestamp_str}\n")
  
  print(f"-- Data saved to {output_dir} with {len(output_data)} frames.")


def main():
  parser = argparse.ArgumentParser(description="Record clarius data with imu")
  parser.add_argument("--ip", type=str, default="192.168.1.1", help="IP address of the Clarius probe")
  parser.add_argument("--port", type=int, default=5828, help="Port of the Clarius probe")
  parser.add_argument("--outputdir", type=str, help="Output directory for recorded data")
  parser.add_argument("--duration", type=int, default=10, help="Duration to record in seconds")
  parser.add_argument("--interval", type=int, default=50, help="Interval in ms between frames")
  parser.add_argument("--preserve-date", action='store_true', default=False, help="Preserve date in timestamps")

  options = parser.parse_args()

  print("-- Initializing Clarius Streamer with IP: {0}, Port: {1}".format(options.ip, options.port))
  ClariusStreamer.configure(options.ip, options.port, 'research', 512, 512)
  ClariusStreamer.start()

  is_recording = False
  output_data = []
  start_time = 0
  start_timestamp = 0

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
        start_timestamp = get_timestamp(options.preserve_date)
        is_recording = True
      elif key.lower() == 'q':
        print("-- Quitting...")
        ClariusStreamer.stop()
        break
      else:
        print("-- Invalid input. Please try again.")
        continue

    if is_recording:
      elapsed_time = time.time() - start_time
      if elapsed_time >= options.duration:
        print(f"-- Recording finished. Elapsed time: {elapsed_time:.2f} seconds.")
        print(f"-- Saving {len(output_data)} frames to folder...")

        study_output_dir = os.path.join(options.outputdir, str(start_timestamp))
        save_result(output_data, study_output_dir)

        output_data = []
        
        is_recording = False
        continue

      output_data.append({
        'timestamp': get_timestamp(options.preserve_date),
        'frame': ClariusStreamer.get_frame()
      })

    # sleep for the specified interval
    time.sleep(options.interval / 1000.0)


if __name__ == "__main__":
  main()