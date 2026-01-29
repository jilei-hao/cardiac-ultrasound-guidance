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

try:
  import openvr
  HAS_OPENVR = True
except ImportError:
  HAS_OPENVR = False


@dataclass
class FrameWriteRequest:
  """Data for writer process - only what's needed to write the image."""
  seq_num: int
  timestamp: int
  image: np.ndarray  # grayscale image
  output_dir: str


class KeyboardHandler:
  """Handles keyboard input with support for detecting held keys."""

  # Keys reserved for control actions (not available as reference labels)
  _RESERVED_KEYS = {'q'}

  def __init__(self):
    self._pressed_keys = set()
    self._key_events = []  # Queue of key press events for toggle detection
    self._reference_key = ""  # Most recently pressed alphanumeric reference key

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
      ch = key_char.lower()
      self._pressed_keys.add(ch)
      if ch == 'q':
        self._key_events.append('q')
      elif ch not in self._RESERVED_KEYS:
        self._reference_key = ch

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
      ch = key_char.lower()
      self._pressed_keys.discard(ch)
      if ch == self._reference_key:
        self._reference_key = ""

  def is_held(self, key_name: str) -> bool:
    """Check if a key is currently held down."""
    return key_name.lower() in self._pressed_keys

  def get_reference_label(self) -> str:
    """Get the currently held reference label, or "" if none."""
    return self._reference_key

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


def init_vr_tracking():
  """Initialize OpenVR and find the first generic tracker.

  Returns:
      Tuple of (vr_system, tracker_device_index) or (None, None) on failure.
  """
  if not HAS_OPENVR:
    print("-- Warning: openvr not installed, tracking disabled")
    return None, None

  try:
    vr_system = openvr.init(openvr.VRApplication_Other)
  except openvr.OpenVRError as e:
    print(f"-- Warning: Could not initialize OpenVR: {e}")
    print("-- Tracking disabled. Make sure SteamVR is running.")
    return None, None

  print("-- Connected to SteamVR")

  # Find the first generic tracker
  for i in range(openvr.k_unMaxTrackedDeviceCount):
    device_class = vr_system.getTrackedDeviceClass(i)
    if device_class == openvr.TrackedDeviceClass_GenericTracker:
      serial = vr_system.getStringTrackedDeviceProperty(
        i, openvr.Prop_SerialNumber_String
      )
      print(f"-- Found tracker: Device {i} ({serial})")
      return vr_system, i

  print("-- Warning: No trackers found, tracking disabled")
  openvr.shutdown()
  return None, None


def get_tracker_pose(vr_system, device_index: int) -> Optional[list]:
  """
  Get the flattened 12-element pose vector for a tracked device.

  The OpenVR 3x4 matrix is column-major (columns are axes)
      m[0][0] m[0][1] m[0][2] m[0][3]     AXx  AYx  AZx  Tx
      m[1][0] m[1][1] m[1][2] m[1][3]  =  AXy  AYy  AZy  Ty
      m[2][0] m[2][1] m[2][2] m[2][3]     AXz  AYz  AZz  Tz

  Returns the rotation (3x3 flattened row-by-row) followed by translation:
      [r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz]

  Returns None if the pose is not valid.
  """
  
  poses = vr_system.getDeviceToAbsoluteTrackingPose(
    openvr.TrackingUniverseStanding, 0,
    openvr.k_unMaxTrackedDeviceCount
  )
  pose = poses[device_index]

  if not pose.bPoseIsValid:
    return None

  m = pose.mDeviceToAbsoluteTracking

  # Flatten rotation (3x3) row-by-row, then translation
  return [
    m[0][0], m[0][1], m[0][2],
    m[1][0], m[1][1], m[1][2],
    m[2][0], m[2][1], m[2][2],
    m[0][3], m[1][3], m[2][3]
  ]


POSE_ZERO = [0.0] * 12


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
        f"frame_{request.seq_num:06d}_{timestamp_str}.png"
      )

      if request.image is not None:
        cv2.imwrite(image_path, request.image)

    except queue.Empty:
      continue
    except Exception as e:
      print(f"Writer {writer_id} error: {e}")


POSE_CSV_COLUMNS = "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz"


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
            f"device_timestamp,microns_per_pixel,valid,reference,{POSE_CSV_COLUMNS}\n")

    for meta in metadata_list:
      timestamp_str = f"{meta['timestamp']:019d}"
      device_timestamp_str = f"{meta['device_timestamp']:014d}"
      valid_int = int(meta['valid'])
      reference = meta['reference']
      pose = meta['pose']
      pose_str = ",".join(f"{v:.6f}" for v in pose)

      if meta['imu'] is not None:
        imu = meta['imu']
        f.write(f"{timestamp_str},"
                f"{imu['gx']},{imu['gy']},{imu['gz']},"
                f"{imu['ax']},{imu['ay']},{imu['az']},"
                f"{imu['mx']},{imu['my']},{imu['mz']},"
                f"{imu['qw']},{imu['qx']},{imu['qy']},{imu['qz']},"
                f"{device_timestamp_str},"
                f"{meta['microns_per_pixel']},"
                f"{valid_int},{reference},{pose_str}\n")
      else:
        print(f"-- Warning: No IMU data for frame at timestamp {timestamp_str}")
        f.write(f"{timestamp_str},"
                f"0,0,0,0,0,0,0,0,0,1,0,0,0,"
                f"{device_timestamp_str},"
                f"{meta['microns_per_pixel']},"
                f"{valid_int},{reference},{pose_str}\n")


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
  parser.add_argument("--enable-tracking", action='store_true', default=True,
                      help="Enable VR tracker pose recording (default: True)")
  parser.add_argument("--disable-tracking", action='store_true', default=False,
                      help="Disable VR tracker pose recording")

  options = parser.parse_args()
  tracking_enabled = options.enable_tracking and not options.disable_tracking

  print(f"-- Initializing Clarius Streamer with IP: {options.ip}, Port: {options.port}")
  ClariusStreamer.configure(options.ip, options.port, 'research', 512, 512)
  if options.debug:
    ClariusStreamer.set_print_debug_info(True)
  ClariusStreamer.start()

  # Initialize VR tracking
  vr_system = None
  tracker_index = None
  if tracking_enabled:
    vr_system, tracker_index = init_vr_tracking()

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
  print("  [Enter]       - Start/stop recording")
  print("  [Space]       - Hold for 'valid' label")
  print("  [a-p, r-z, 0-9] - Hold for reference label (key = class)")
  print("  [Q]           - Quit")
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
        reference = kb_handler.get_reference_label()

        # Fetch frame
        frame = ClariusStreamer.get_frame()
        timestamp = get_timestamp(options.preserve_date)

        # Get tracker pose
        pose = POSE_ZERO
        if vr_system is not None:
          tracker_pose = get_tracker_pose(vr_system, tracker_index)
          if tracker_pose is not None:
            pose = tracker_pose
          else:
            print(f"-- Warning: Invalid tracker pose at frame {seq_num}")

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
          'reference': reference,
          'pose': pose
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

    # Shutdown OpenVR
    if vr_system is not None:
      openvr.shutdown()

    print("-- Done.")


if __name__ == "__main__":
  main()
