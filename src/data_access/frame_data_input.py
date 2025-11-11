from abc import ABC, abstractmethod
from .frame_data import FrameData

class FrameDataInput(ABC):
    """
    Base class for all data access classes that provide frame data.
    This abstract class defines the interface that all frame data sources must implement.
    """
    
    @abstractmethod
    def get_frame(self) -> FrameData:
        """
        Retrieve the next frame from the data source.
        
        Returns:
            FrameData: A FrameData object containing image, timestamp, and IMU data (if available)
        """
        pass
    
    @abstractmethod
    def start(self):
        """
        Initialize and start the data source.
        """
        pass
    
    @abstractmethod
    def stop(self):
        """
        Stop and cleanup the data source.
        """
        pass