import queue


class FrameBuffer:
  """
  A class to manage a fixed-size buffer for storing frames.

  Attributes:
      buffer_size (int): The maximum number of frames the buffer can hold.
      frame_queue (queue.Queue): A queue to store the frames with a fixed size.

  Methods:
      add_frame(frame): Adds a new frame to the buffer, removing the oldest frame if the buffer is full.
      get_number_of_frames(): Returns the current number of frames in the buffer.
      get_all_frames(): Returns a list of all frames currently in the buffer.
  """
  def __init__(self, buffer_size: int):
    self.buffer_size = buffer_size
    self.frame_queue = queue.Queue(maxsize=buffer_size)


  def add_frame(self, frame):
    if self.frame_queue.full():
      self.frame_queue.get()  # Remove the oldest frame
    self.frame_queue.put(frame)


  def get_number_of_frames(self):
    return self.frame_queue.qsize()


  def get_all_frames(self):
    return list(self.frame_queue.queue)