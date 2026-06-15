import queue
import threading
import numpy as np

from ..pose_classification.pose_classification import PoseResult


def _as_pose_result(classifier_result):
  """Normalize a classification into a PoseResult.

  Accepts either a PoseResult (preferred) or a bare (R, G, B) rgb tuple from
  legacy/transitional callers. A bare tuple is wrapped with score=NaN so that
  ``is_video_good`` and diagnostics can always rely on ``.rgb`` / ``.score``.
  """
  if isinstance(classifier_result, PoseResult):
    return classifier_result
  return PoseResult(score=float("nan"), rgb=tuple(classifier_result))


class ClassifiedFrame:
  """
  A class to represent a frame with its traffic-light classification result.

  Attributes:
    frame (any): The frame data.
    classifier_result (PoseResult): The pose-classification result for this
      frame, carrying both the continuous ``score`` and the discretized
      ``rgb`` traffic-light color. (Historically this was just the rgb tuple;
      it is now a ``pose_classification.PoseResult``.)
  """
  def __init__(self, frame, classifier_result):
    self.frame = frame
    self.classifier_result = classifier_result  # PoseResult(score, rgb)

class FrameBuffer:
  """
  A thread-safe class to manage a fixed-size buffer for storing frames.

  Attributes:
    buffer_size (int): The maximum number of frames the buffer can hold.
    frame_queue (queue.Queue): A queue to store the frames with a fixed size.
    lock (threading.Lock): A lock to ensure thread-safe access to the buffer.

  Methods:
    add_frame(frame): Adds a new frame to the buffer, removing the oldest frame if the buffer is full.
    get_number_of_frames(): Returns the current number of frames in the buffer.
    get_all_frames(): Returns a list of all frames currently in the buffer.
  """
  def __init__(self, buffer_size: int):
    self.buffer_size = buffer_size
    self.frame_queue = queue.Queue(maxsize=buffer_size)
    self.lock = threading.Lock()  # Lock for thread-safe access

  def add_frame(self, frame, classifier_result):
    pose_result = _as_pose_result(classifier_result)
    with self.lock:  # Acquire the lock
      if self.frame_queue.full():
        old_frame = self.frame_queue.get()  # Remove the oldest frame
        # print old frame dimensions
        # print(f"Removed frame with dimensions: {old_frame.frame.shape}")
        del old_frame

      self.frame_queue.put(ClassifiedFrame(frame, pose_result))  # Add the new frame

  def get_number_of_frames(self):
    with self.lock:  # Acquire the lock
      return self.frame_queue.qsize()

  def get_all_frames(self):
    with self.lock:  # Acquire the lock
      frame_array = list(self.frame_queue.queue)
      return np.array([frame.frame for frame in frame_array])

  def get_all_classifications(self):
    """Return the per-frame PoseResult for every frame currently buffered.

    Useful for troubleshooting (e.g. logging the score/colour distribution
    the LVEF model is actually seeing at prediction time).
    """
    with self.lock:
      return [cf.classifier_result for cf in list(self.frame_queue.queue)]


  def is_video_good(self):
    """
    Compute the quality of the buffer based on the classification results.
    This is a placeholder for a more complex quality computation.
    returns boolean
    """
    with self.lock:  # Acquire the lock
      if self.frame_queue.empty():
        return False  # No frames to compute quality
      
      if self.frame_queue.qsize() < self.buffer_size:
        return False

      # Example: Green + Yellow > 90% and Green > 30%
      green_count = sum(1 for frame in self.frame_queue.queue if frame.classifier_result.rgb == (0, 255, 0))
      yellow_count = sum(1 for frame in self.frame_queue.queue if frame.classifier_result.rgb == (255, 255, 0))
      total_count = self.frame_queue.qsize()

      pct_yellow = yellow_count / total_count * 100
      pct_green = green_count / total_count * 100


      return (pct_yellow + pct_green) > 80