import mss
import data_access.screen_capture.macos_window_helpers as mwh
from data_access.frame_data import FrameData
from data_access.frame_data_input import FrameDataInput
import time

class ScreenCapture(FrameDataInput):
  def __init__(self, window_title):
    # raise error for non-macOS systems
    if not mwh.is_macos():
      raise EnvironmentError("🛑 ScreenCapture is only supported on macOS systems."
                             " Please use a different data source.")
    
    self.window_title = window_title
    self.sct = None  # Placeholder for mss object
    window_bbox = mwh.get_window_bbox(window_title)
    if not window_bbox:
      print(f"🟡 Warning: Window '{window_title}' not found. Using default input window dimensions...")
      window_bbox = {"top": 60, "left": 45, "width": 450, "height": 650}

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