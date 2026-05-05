import torchvision
import torch.nn as nn
import torch
import numpy as np
from .frame_buffer import FrameBuffer

class LVEFPredictor:
    """
    A class for predicting Left Ventricular Ejection Fraction (LVEF) using a pre-trained model.
    Attributes:
        model (torch.nn.Module): The pre-trained model for LVEF prediction.
        device (torch.device): The device to run the model on (CPU or GPU).
    """
    def __init__(self, device, weight_path=None):
        self.device = device
        # load model
        lvef_model_name = "r2plus1d_18"
        self.model = torchvision.models.video.__dict__[lvef_model_name](pretrained=None)
        # change model structure according to echonet
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model.fc.bias.data[0] = 55.6 # this is from echonet/utils/video.py
        # send model to device
        self.model = self.model.to(device)
        self._frame_buffer = FrameBuffer(buffer_size=100)

        # load model weights
        if weight_path is None:
            raise ValueError("[lvef_prediction] weight_path cannot be None")
        else:
            print("Loading checkpoint {}".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=device)
            # Strip "module." from keys for CPU/MPS
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(new_state_dict)
            

    def _get_mean_and_std(self, video_matrix):
        """
        video_matrix is a 4D array, shaped (N, W, D, 3), 
        N is the number of frames, W is the width, D is the depth, 3 is the RGB channels
        """
        s1 = 0 # sum of elements along channels
        s2 = 0 # sum of squares of elements along channels
        # first bring channel to 0th dimension
        x = torch.from_numpy(video_matrix).float()
        x = x.permute(3, 0, 1, 2)
        x_flattened = x.reshape(3, -1)
        n = x_flattened.shape[1] # number of elements along channels
        s1 = torch.sum(x_flattened, dim=1).numpy()
        s2 = torch.sum(x_flattened**2, dim=1).numpy()
        mean = s1.astype(np.float32) / n
        std = np.sqrt(s2 / n - mean**2).astype(np.float32)
        return mean, std
    
    
    def add_frame_to_buffer(self, frame, pose_classification_result):
        """
        Add a frame to the internal frame buffer.
        Args:
            frame (np.ndarray): A single video frame (H, W, 3).
            pose_classification_result: The pose classification result associated with the frame.
        """
        self._frame_buffer.add_frame(frame, pose_classification_result)


    def is_buffer_video_good(self): 
        """
        Check if the current video in the buffer is good for prediction.
        Returns:
            bool: True if the video is good, False otherwise.
        """
        return self._frame_buffer.is_video_good() # this should not be in frame buffer

    
    def run_prediction(self):
        """
        Run LVEF prediction on a video matrix.
        Args:
            video_matrix (np.ndarray): A 4D array of shape (N, W, D, 3) representing the video frames.
        Returns:
            float: The predicted LVEF value.
        """
        video_matrix = self._frame_buffer.get_all_frames()

        # get mean and std
        mean, std = self._get_mean_and_std(video_matrix)
        # normalize video matrix
        x = torch.from_numpy(video_matrix).float()
        x = (x - mean) / std
        
        # run through model
        x = x.permute(3, 0, 1, 2)
        x = x.unsqueeze(0)
        x = x.float().to(self.device)
        try:
            with torch.no_grad():
                y = self.model(x)
            lvef = int(y.squeeze().cpu().numpy())
        finally:
            del x
            if 'y' in locals():
                del y
            # Return the inference-time activations to the driver instead
            # of letting the caching allocator hold onto them indefinitely.
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()

        return lvef