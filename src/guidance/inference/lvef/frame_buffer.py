import queue
import threading
import numpy as np

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
    self.minimum_acceptable_frames = int(buffer_size * 0.8) + 1
    self.frame_queue = queue.Queue(maxsize=buffer_size)
    self.acceptable_frames = 0
    self.lock = threading.Lock()  # Lock for thread-safe access

  @staticmethod
  def _is_acceptable(classified_frame):
    return classified_frame.classifier_result.rgb in {
      (0, 255, 0), (255, 255, 0)
    }

  def add_frame(self, frame, classifier_result):
    with self.lock:  # Acquire the lock
      if self.frame_queue.full():
        old_frame = self.frame_queue.get()  # Remove the oldest frame
        if self._is_acceptable(old_frame):
          self.acceptable_frames -= 1
        # print old frame dimensions
        # print(f"Removed frame with dimensions: {old_frame.frame.shape}")
        del old_frame

      classified_frame = ClassifiedFrame(frame, classifier_result)
      self.frame_queue.put(classified_frame)
      if self._is_acceptable(classified_frame):
        self.acceptable_frames += 1

  def get_number_of_frames(self):
    with self.lock:  # Acquire the lock
      return self.frame_queue.qsize()

  def get_all_frames(self):
    with self.lock:  # Acquire the lock
      frame_array = list(self.frame_queue.queue)
      return np.array([frame.frame for frame in frame_array])

  def clear(self):
    with self.lock:
      self.frame_queue = queue.Queue(maxsize=self.buffer_size)
      self.acceptable_frames = 0

  def is_full(self):
    with self.lock:
      return self.frame_queue.full()

  def capture_progress(self):
    with self.lock:
      return (
        self.frame_queue.qsize(), self.acceptable_frames,
        self.buffer_size, self.minimum_acceptable_frames
      )

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

      return self.acceptable_frames >= self.minimum_acceptable_frames
