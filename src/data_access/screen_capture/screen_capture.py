import mss
import data_access.screen_capture.macos_window_helpers as mwh
from data_access.frame_data import FrameData
import time

class ScreenCapture:
  def __init__(self, window_title, app_settings):
    self.window_title = window_title
    self.app_settings = app_settings
    self.sct = None  # Placeholder for mss object
    window_bbox = mwh.get_window_bbox(window_title)
    if window_bbox:
      app_settings.input_window_bbox = window_bbox
    else:
      print(f"🟡 Warning: Window '{window_title}' not found. Using default input window dimensions...")

  def start(self):
    """Initialize the mss object."""
    self.sct = mss.mss()
    print("✅ ScreenCapture started.")

  def stop(self):
    """Clean up the mss object."""
    if self.sct:
      self.sct.close()
      self.sct = None
    print("🛑 ScreenCapture stopped.")

  def get_frame(self):
    """
    Capture a frame from the specified window using mss.
    Returns a FrameData object containing the captured image and timestamp.
    """
    if not self.sct:
      print("🛑 Error: ScreenCapture not started. Call start() first.")
      return FrameData(None, None, None)

    try:
      screenshot = self.sct.grab(self.app_settings.input_window_bbox)
      image = screenshot
      timestamp = time.time_ns()
      return FrameData(image, timestamp, None) # No IMU data for screen capture
    except Exception as e:
      print(f"🛑 Error capturing frame: {str(e)}")
      return FrameData(None, None, None)