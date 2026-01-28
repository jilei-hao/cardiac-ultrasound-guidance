import data_access.clarius.clarius_streamer as ClariusStreamer
import argparse
import time
import os
import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from dataclasses import dataclass
from typing import Optional
import queue

from pynput import keyboard


@dataclass
class FrameWriteRequest:
  """Data for writer process - only what's needed to write the image."""
  seq_num: int
  timestamp: int
  image: np.ndarray  # grayscale image
  output_dir: str


class KeyboardHandler:
  """Handles keyboard input with support for detecting held keys."""

  def __init__(self):
    self._pressed_keys = set()
    self._key_events = []  # Queue of key press events for toggle detection

    # Start the keyboard listener
    self._listener = keyboard.Listener(
      on_press=self._on_press,
      on_release=self._on_release
    )
    self._listener.start()

  def _on_press(self, key):
    """Called when a key is pressed."""
    try:
      key_char = key.char if hasattr(key, 'char') else None
    except AttributeError:
      key_char = None

    # Track special keys
    if key == keyboard.Key.space:
      self._pressed_keys.add('space')
    elif key == keyboard.Key.enter:
      self._key_events.append('enter')
    elif key_char:
      self._pressed_keys.add(key_char.lower())
      # Track 'q' as an event for quitting
      if key_char.lower() == 'q':
        self._key_events.append('q')

  def _on_release(self, key):
    """Called when a key is released."""
    try:
      key_char = key.char if hasattr(key, 'char') else None
    except AttributeError:
      key_char = None

    # Remove from pressed keys
    if key == keyboard.Key.space:
      self._pressed_keys.discard('space')
    elif key_char:
      self._pressed_keys.discard(key_char.lower())

  def is_held(self, key_name: str) -> bool:
    """Check if a key is currently held down."""
    return key_name.lower() in self._pressed_keys

  def get_event(self) -> Optional[str]:
    """Get the next key event (for toggle keys like enter, q)."""
    if self._key_events:
      return self._key_events.pop(0)
    return None

  def stop(self):
    """Stop the keyboard listener."""
    self._listener.stop()


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


def writer_process(frame_queue: Queue, shutdown_event: Event, writer_id: int):
  """Writer process: pulls frames from queue and writes to disk.

  Args:
      frame_queue: Queue containing FrameWriteRequest objects to write
      shutdown_event: Event signaling shutdown
      writer_id: ID of this writer process (for debugging)
  """
  while not shutdown_event.is_set() or not frame_queue.empty():
    try:
      request: FrameWriteRequest = frame_queue.get(timeout=0.1)

      # Write image
      timestamp_str = f"{request.timestamp:019d}"
      image_path = os.path.join(
        request.output_dir,
        f"frame_{request.seq_num:04d}_{timestamp_str}.png"
      )

      if request.image is not None:
        cv2.imwrite(image_path, request.image)

    except queue.Empty:
      continue
    except Exception as e:
      print(f"Writer {writer_id} error: {e}")


def write_metadata_csv(metadata_list: list, output_dir: str):
  """Write the metadata CSV file from collected metadata.

  Args:
      metadata_list: List of metadata dictionaries
      output_dir: Output directory for the CSV file
  """
  # Sort by sequence number
  metadata_list.sort(key=lambda x: x['seq_num'])

  csv_path = os.path.join(output_dir, "frame.meta.csv")
  with open(csv_path, 'w') as f:
    f.write("timestamp,gx,gy,gz,ax,ay,az,mx,my,mz,qw,qx,qy,qz,"
            "device_timestamp,microns_per_pixel,valid,reference\n")

    for meta in metadata_list:
      timestamp_str = f"{meta['timestamp']:019d}"
      device_timestamp_str = f"{meta['device_timestamp']:014d}"

      if meta['imu'] is not None:
        imu = meta['imu']
        f.write(f"{timestamp_str},"
                f"{imu['gx']},{imu['gy']},{imu['gz']},"
                f"{imu['ax']},{imu['ay']},{imu['az']},"
                f"{imu['mx']},{imu['my']},{imu['mz']},"
                f"{imu['qw']},{imu['qx']},{imu['qy']},{imu['qz']},"
                f"{device_timestamp_str},"
                f"{meta['microns_per_pixel']},"
                f"{meta['valid']},{meta['reference']}\n")
      else:
        print(f"-- Warning: No IMU data for frame at timestamp {timestamp_str}")
        f.write(f"{timestamp_str},"
                f"0,0,0,0,0,0,0,0,0,1,0,0,0,"
                f"{device_timestamp_str},"
                f"{meta['microns_per_pixel']},"
                f"{meta['valid']},{meta['reference']}\n")


def main():
  parser = argparse.ArgumentParser(description="Record clarius data with imu (async)")
  parser.add_argument("--ip", type=str, default="192.168.1.1",
                      help="IP address of the Clarius probe")
  parser.add_argument("--port", type=int, default=5828,
                      help="Port of the Clarius probe")
  parser.add_argument("--outputdir", type=str, required=True,
                      help="Output directory for recorded data")
  parser.add_argument("--interval", type=int, default=50,
                      help="Interval in ms between frames")
  parser.add_argument("--writers", type=int, default=1,
                      help="Number of writer processes")
  parser.add_argument("--preserve-date", action='store_true', default=True,
                      help="Preserve date in timestamps")
  parser.add_argument("--debug", action='store_true', default=False,
                      help="Enable debug information")

  options = parser.parse_args()

  print(f"-- Initializing Clarius Streamer with IP: {options.ip}, Port: {options.port}")
  ClariusStreamer.configure(options.ip, options.port, 'research', 512, 512)
  if options.debug:
    ClariusStreamer.set_print_debug_info(True)
  ClariusStreamer.start()

  # Initialize keyboard handler
  kb_handler = KeyboardHandler()

  # Create multiprocessing primitives
  frame_queue = mp.Queue()
  shutdown_event = mp.Event()

  # Start writer processes
  writers = []
  for i in range(options.writers):
    w = Process(
      target=writer_process,
      args=(frame_queue, shutdown_event, i)
    )
    w.start()
    writers.append(w)

  print(f"-- Started {options.writers} writer process(es)")

  # Recording state
  is_recording = False
  seq_num = 0
  output_dir = ""
  metadata_buffer = []  # Kept in main process
  frame_count = 0

  print("Controls:")
  print("  [Enter] - Start/stop recording")
  print("  [Space] - Hold for 'valid' label")
  print("  [F]     - Hold for 'reference' label")
  print("  [Q]     - Quit")
  print("")
  print("Ready. Press [Enter] to start recording...")

  try:
    while True:
      # Check for key events
      event = kb_handler.get_event()

      if event == 'q':
        print("-- Quitting...")
        break

      elif event == 'enter':
        if not is_recording:
          # Start recording
          is_recording = True
          seq_num = 0
          frame_count = 0
          metadata_buffer = []
          start_timestamp = get_timestamp(options.preserve_date)
          output_dir = os.path.join(options.outputdir, str(start_timestamp))
          os.makedirs(output_dir, exist_ok=True)
          print(f"-- Recording started. Output: {output_dir}")
        else:
          # Stop recording
          is_recording = False
          print(f"-- Recording stopped. {frame_count} frames captured.")
          print("-- Waiting for writers to finish...")

          # Wait for frame queue to drain
          while not frame_queue.empty():
            time.sleep(0.1)

          # Give writers a moment to finish any in-flight writes
          time.sleep(0.3)

          # Write CSV from local metadata buffer
          if metadata_buffer:
            write_metadata_csv(metadata_buffer, output_dir)
            print(f"-- Saved {len(metadata_buffer)} frames to {output_dir}")
          else:
            print("-- No frames to save.")

          print("")
          print("Ready. Press [Enter] to start recording...")

      # If recording, fetch and queue frames
      if is_recording:
        # Get current key states for labeling
        valid = kb_handler.is_held('space')
        reference = kb_handler.is_held('f')

        # Fetch frame
        frame = ClariusStreamer.get_frame()
        timestamp = get_timestamp(options.preserve_date)

        # Convert to grayscale numpy array
        if frame.image is not None:
          orig_frame = np.array(frame.image)
          image_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        else:
          image_gray = None

        # Store metadata locally
        imu_dict = None
        if frame.imu is not None:
          imu_dict = {
            'gx': frame.imu.gx, 'gy': frame.imu.gy, 'gz': frame.imu.gz,
            'ax': frame.imu.ax, 'ay': frame.imu.ay, 'az': frame.imu.az,
            'mx': frame.imu.mx, 'my': frame.imu.my, 'mz': frame.imu.mz,
            'qw': frame.imu.qw, 'qx': frame.imu.qx, 'qy': frame.imu.qy, 'qz': frame.imu.qz
          }

        metadata_buffer.append({
          'seq_num': seq_num,
          'timestamp': timestamp,
          'device_timestamp': frame.timestamp,
          'microns_per_pixel': frame.microns_per_pixel,
          'imu': imu_dict,
          'valid': valid,
          'reference': reference
        })

        # Queue image for writing
        write_request = FrameWriteRequest(
          seq_num=seq_num,
          timestamp=timestamp,
          image=image_gray,
          output_dir=output_dir
        )
        frame_queue.put(write_request)

        seq_num += 1
        frame_count += 1

        # Warn if queue is getting large
        qsize = frame_queue.qsize()
        if qsize > 100 and qsize % 50 == 0:
          print(f"-- Warning: frame queue size is {qsize}")

      # Sleep for the specified interval
      time.sleep(options.interval / 1000.0)

  except KeyboardInterrupt:
    print("\n-- Interrupted by user")

  finally:
    # Cleanup
    print("-- Shutting down...")

    # Signal shutdown to writers
    shutdown_event.set()

    # Wait for writers to finish
    for w in writers:
      w.join(timeout=5.0)
      if w.is_alive():
        print("-- Warning: Writer process did not terminate, forcing...")
        w.terminate()

    # Stop keyboard handler
    kb_handler.stop()

    # Stop Clarius streamer
    ClariusStreamer.stop()

    print("-- Done.")


if __name__ == "__main__":
  main()
