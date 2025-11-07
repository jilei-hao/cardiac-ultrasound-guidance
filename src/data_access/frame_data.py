from dataclasses import dataclass

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
class FrameData:
  image: any
  timestamp: float
  imu: IMUData