import queue


class FrameBuffer:
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