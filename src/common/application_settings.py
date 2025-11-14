import matplotlib.colors as mcolors
import numpy as np 

SHARED_CMAP = mcolors.LinearSegmentedColormap.from_list(
  "GreenYellowRed",
  [
    (0.0, "green"),
    (0.3, "green"),
    (0.4, "yellow"),
    (0.6, "yellow"),
    (0.7, "red"),
    (1.0, "red")
  ]
)

class ApplicationSettings:
  """
  ApplicationSettings class to manage application settings and configurations.
  This class is used to store the settings during the application runtime.
  Any module or method that need to access the real-time settings need take the instance of this class.
  """
  def __init__(self):
    self.result_window_dimensions = (512, 512)
    self.input_window_bbox = {"top": 60, "left": 45, "width": 450, "height": 650}
    self.frame_dimensions = (112, 112)
    self.guidance_window_dimensions = (512, 512)
    self.guidance_frame_dimensions = (256, 256)
    self.run_lvef = False
    self.print_fps = True
    self.use_threshold_for_pose_classification = False
    self.landmark_radius = 5
    self.landmark_alpha = 0.8
    self.run_image_upload = False
    # add hotkey to NOT show border (default is True)
    self.show_border = True
    self.show_overlay = True
    self.print_debug_info = False

  def increment_landmark_radius(self):
    self.landmark_radius += 1

  def decrement_landmark_radius(self):
    if self.landmark_radius > 1:
      self.landmark_radius -= 1

  def increment_landmark_alpha(self):
    self.landmark_alpha += 0.1

  def decrement_landmark_alpha(self):
    if self.landmark_alpha > 0.1:
      self.landmark_alpha -= 0.1
    

  def get_color_for_ere(self, ere, max_ere):
    # Normalize the ERE value: 0 (best) to max_ere (worst)
    normalized = ere / max_ere
    normalized = np.clip(normalized, 0, 1)  # Ensure normalized is within [0, 1]

    # Ensure normalized is a NumPy array with at least one dimension
    normalized = np.array([normalized])

    # Get the color from the colormap
    rgba = SHARED_CMAP(normalized)[0]  # Extract the first (and only) RGBA tuple

    r, g, b, a = rgba
    return (int(r * 255), int(g * 255), int(b * 255))