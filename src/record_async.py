import data_access.clarius.clarius_streamer as ClariusStreamer
import argparse
import time
import os
import json
from datetime import datetime
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


def make_recording_dir(base_dir: str) -> str:
  """Create a timestamped recording directory.

  Directory name format: usgr_invivo_YYYY-MM-DD--HH-MM-SS

  Args:
      base_dir: Parent directory for recordings.
  Returns:
      str: Path to the created recording directory.
  """
  now = datetime.now()
  dirname = now.strftime("usgr_invivo_%Y-%m-%d--%H-%M-%S")
  recording_dir = os.path.join(base_dir, dirname)
  os.makedirs(recording_dir, exist_ok=True)
  os.makedirs(os.path.join(recording_dir, "images"), exist_ok=True)
  return recording_dir


def init_vr_tracking():
  """Initialize OpenVR and find the first generic tracker.

  Returns:
      Tuple of (vr_system, tracker_device_index, tracker_serial)
      or (None, None, None) on failure.
  """
  if not HAS_OPENVR:
    print("-- Warning: openvr not installed, tracking disabled")
    return None, None, None

  try:
    vr_system = openvr.init(openvr.VRApplication_Other)
  except openvr.OpenVRError as e:
    print(f"-- Warning: Could not initialize OpenVR: {e}")
    print("-- Tracking disabled. Make sure SteamVR is running.")
    return None, None, None

  print("-- Connected to SteamVR")

  # Find the first generic tracker
  for i in range(openvr.k_unMaxTrackedDeviceCount):
    device_class = vr_system.getTrackedDeviceClass(i)
    if device_class == openvr.TrackedDeviceClass_GenericTracker:
      serial = vr_system.getStringTrackedDeviceProperty(
        i, openvr.Prop_SerialNumber_String
      )
      print(f"-- Found tracker: Device {i} ({serial})")
      return vr_system, i, serial

  print("-- Warning: No trackers found, tracking disabled")
  openvr.shutdown()
  return None, None, None


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

      # Write image to images/ subdirectory
      timestamp_str = f"{request.timestamp:019d}"
      image_path = os.path.join(
        request.output_dir, "images",
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
    f.write("index,timestamp,gx,gy,gz,ax,ay,az,mx,my,mz,qw,qx,qy,qz,"
            f"clarius_timestamp,microns_per_pixel,valid,reference,{POSE_CSV_COLUMNS}\n")

    for meta in metadata_list:
      seq_num = meta['seq_num']
      timestamp_str = f"{meta['timestamp']:019d}"
      device_timestamp_str = f"{meta['device_timestamp']:014d}"
      valid_int = int(meta['valid'])
      reference = meta['reference']
      pose = meta['pose']
      pose_str = ",".join(f"{v:.6f}" for v in pose)

      if meta['imu'] is not None:
        imu = meta['imu']
        f.write(f"{seq_num},{timestamp_str},"
                f"{imu['gx']},{imu['gy']},{imu['gz']},"
                f"{imu['ax']},{imu['ay']},{imu['az']},"
                f"{imu['mx']},{imu['my']},{imu['mz']},"
                f"{imu['qw']},{imu['qx']},{imu['qy']},{imu['qz']},"
                f"{device_timestamp_str},"
                f"{meta['microns_per_pixel']},"
                f"{valid_int},{reference},{pose_str}\n")
      else:
        print(f"-- Warning: No IMU data for frame at timestamp {timestamp_str}")
        f.write(f"{seq_num},{timestamp_str},"
                f"0,0,0,0,0,0,0,0,0,1,0,0,0,"
                f"{device_timestamp_str},"
                f"{meta['microns_per_pixel']},"
                f"{valid_int},{reference},{pose_str}\n")


def write_recording_metadata(output_dir: str, operator_name: str, patient_name: str,
                             tracker_serial: Optional[str]):
  """Write per-recording metadata to a JSON file.

  Args:
      output_dir: Recording directory
      operator_name: Name of the operator
      patient_name: Name of the patient
      tracker_serial: Serial number of the VR tracker, or None if not used
  """
  metadata = {
    "operator": operator_name,
    "patient": patient_name,
    "tracking_device": tracker_serial if tracker_serial else "none",
    "recording_time": datetime.now().isoformat()
  }

  json_path = os.path.join(output_dir, "recording.meta.json")
  with open(json_path, 'w') as f:
    json.dump(metadata, f, indent=2)


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

  # Prompt for operator and patient names
  print("=== Recording Session Setup ===")
  operator_name = input("Operator name: ").strip()
  patient_name = input("Patient name: ").strip()
  if not operator_name:
    print("-- Error: Operator name is required.")
    return
  if not patient_name:
    print("-- Error: Patient name is required.")
    return
  print("")

  print(f"-- Initializing Clarius Streamer with IP: {options.ip}, Port: {options.port}")
  ClariusStreamer.configure(options.ip, options.port, 'research', 512, 512)
  if options.debug:
    ClariusStreamer.set_print_debug_info(True)
  ClariusStreamer.start()

  # Initialize VR tracking
  vr_system = None
  tracker_index = None
  tracker_serial = None
  if tracking_enabled:
    vr_system, tracker_index, tracker_serial = init_vr_tracking()

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

  # Stale frame detection
  STALE_THRESHOLD = 4
  last_device_timestamp = None
  stale_repeat_count = 0
  is_stale = False

  print("Controls:")
  print("  [Enter]          - Start/stop recording")
  print("  [Space]          - Hold for 'valid' label")
  print("  [a-p, r-z, 0-9]  - Hold for reference label (key = class)")
  print("  [Q]              - Quit")
  print("")
  print("Ready. Press [Enter] to start recording...")

  interval_sec = options.interval / 1000.0

  try:
    while True:
      cycle_start = time.perf_counter()

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
          last_device_timestamp = None
          stale_repeat_count = 0
          is_stale = False
          output_dir = make_recording_dir(options.outputdir)
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

          # Write CSV and JSON metadata
          if metadata_buffer:
            write_metadata_csv(metadata_buffer, output_dir)
            write_recording_metadata(output_dir, operator_name, patient_name,
                                     tracker_serial)
            print(f"-- Saved {len(metadata_buffer)} frames to {output_dir}")
          else:
            print("-- No frames to save.")

          print("")
          print("Ready. Press [Enter] to start recording...")

      # If recording, fetch and queue frames
      if is_recording:
        # Fetch frame
        frame = ClariusStreamer.get_frame()

        if frame is None:
          elapsed = time.perf_counter() - cycle_start
          remaining = interval_sec - elapsed
          if remaining > 0:
            time.sleep(remaining)
          continue

        # Stale frame detection: track repeated device timestamps
        if frame.timestamp == last_device_timestamp:
          stale_repeat_count += 1
          if stale_repeat_count >= STALE_THRESHOLD:
            if not is_stale:
              is_stale = True
              print(f"-- Warning: Stale frames detected (device timestamp {frame.timestamp} "
                    f"repeated {stale_repeat_count}+ times), rejecting until new frame arrives")
            elapsed = time.perf_counter() - cycle_start
            remaining = interval_sec - elapsed
            if remaining > 0:
              time.sleep(remaining)
            continue
        else:
          if is_stale:
            print(f"-- New frame received (device timestamp {frame.timestamp}), "
                  f"resuming recording after {stale_repeat_count} stale frames")
          last_device_timestamp = frame.timestamp
          stale_repeat_count = 0
          is_stale = False

        # Get current key states for labeling
        valid = kb_handler.is_held('space')
        reference = kb_handler.get_reference_label()

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

      # Sleep for the remaining interval to maintain consistent cycle timing
      elapsed = time.perf_counter() - cycle_start
      remaining = interval_sec - elapsed
      if remaining > 0:
        time.sleep(remaining)

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
