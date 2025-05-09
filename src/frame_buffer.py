import queue
import threading
import numpy as np

class ClassifiedFrame:
  """
  A class to represent a frame with its classification result.

  Attributes:
    frame (any): The frame data.
    classifier_result (any): The result of the classification.
  """
  def __init__(self, frame, classifier_result):
    self.frame = frame
    self.classifier_result = classifier_result # in rgb (0-255, 0-255, 0-255) format

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
    with self.lock:  # Acquire the lock
      if self.frame_queue.full():
        old_frame = self.frame_queue.get()  # Remove the oldest frame
        # print old frame dimensions
        # print(f"Removed frame with dimensions: {old_frame.frame.shape}")
        del old_frame

      self.frame_queue.put(ClassifiedFrame(frame, classifier_result))  # Add the new frame

  def get_number_of_frames(self):
    with self.lock:  # Acquire the lock
      return self.frame_queue.qsize()

  def get_all_frames(self):
    with self.lock:  # Acquire the lock
      frame_array = list(self.frame_queue.queue)
      return np.array([frame.frame for frame in frame_array])
    

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
      green_count = sum(1 for frame in self.frame_queue.queue if frame.classifier_result == (0, 255, 0))
      yellow_count = sum(1 for frame in self.frame_queue.queue if frame.classifier_result == (255, 255, 0))
      total_count = self.frame_queue.qsize()

      pct_yellow = yellow_count / total_count * 100
      pct_green = green_count / total_count * 100


      return (pct_yellow + pct_green) > 90