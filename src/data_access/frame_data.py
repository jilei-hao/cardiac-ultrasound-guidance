from dataclasses import dataclass
from typing import Optional

@dataclass
class IMUData:
  gx: float
  gy: float
  gz: float
  ax: float
  ay: float
  az: float
  mx: float
  my: float
  mz: float
  qw: float
  qx: float
  qy: float
  qz: float

  @staticmethod
  def from_dict(data: dict):
      return IMUData(
          gx=float(data.get('gx', 0.0)),
          gy=float(data.get('gy', 0.0)),
          gz=float(data.get('gz', 0.0)),
          ax=float(data.get('ax', 0.0)),
          ay=float(data.get('ay', 0.0)),
          az=float(data.get('az', 0.0)),
          mx=float(data.get('mx', 0.0)),
          my=float(data.get('my', 0.0)),
          mz=float(data.get('mz', 0.0)),
          qw=float(data.get('qw', 1.0)),
          qx=float(data.get('qx', 0.0)),
          qy=float(data.get('qy', 0.0)),
          qz=float(data.get('qz', 0.0))
      )

@dataclass
class TrackerData:
  r00: float
  r01: float
  r02: float
  r10: float
  r11: float
  r12: float
  r20: float
  r21: float
  r22: float
  tx:  float
  ty:  float
  tz:  float

  @staticmethod
  def from_dict(data: dict):
      return TrackerData(
          r00 = float(data.get('r00', 0.0)),
          r01 = float(data.get('r01', 0.0)),
          r02 = float(data.get('r02', 0.0)),
          r10 = float(data.get('r10', 0.0)),
          r11 = float(data.get('r11', 0.0)),
          r12 = float(data.get('r12', 0.0)),
          r20 = float(data.get('r20', 0.0)),
          r21 = float(data.get('r21', 0.0)),
          r22 = float(data.get('r22', 0.0)),
          tx  = float(data.get('tx',  0.0)),
          ty  = float(data.get('ty',  0.0)),
          tz  = float(data.get('tz',  0.0)),
      )

@dataclass
class FrameData:
  image: any
  timestamp: float
  microns_per_pixel: float
  imu: Optional[IMUData] = None
  tracker: Optional[TrackerData] = None
