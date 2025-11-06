from dataclasses import dataclass

@dataclass
class FrameData:
  image: any
  timestamp: float
  imu: any