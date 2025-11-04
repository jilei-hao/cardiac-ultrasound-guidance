#!/usr/bin/env python3
"""
4-Chamber Landmark Pipeline
Combines segmentation and landmark extraction for real-time processing.
"""

import os
import torch
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import tempfile
import shutil

# Import with absolute paths to avoid relative import issues
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from echo_dataset import LABEL_VALUES
from convnext_fpn import ConvNeXtFPN
from qc_seg2landmarks import (
    process_sequence, 
    intersect_labels, 
    biggest_contour, 
    sample_equally_spaced_points,
    landmark_uncertainty,
    compute_uncertainty_maps,
    adjust_color_by_uncertainty,
    color_for_label
)


class FourChamberLandmarkPipeline:
    """
    Real-time 4-chamber landmark pipeline that:
    1. Performs segmentation using ConvNeXt-FPN
    2. Extracts landmarks from segmentation masks
    3. Returns landmarks in the same format as the original landmark prediction
    """
    
    def __init__(self, weight_path: str, device: str = "auto", img_size: int = 512, 
                 landmarks_per_chamber: Optional[Dict[int, int]] = None):
        """
        Initialize the pipeline.
        
        Args:
            weight_path: Path to the 4ch-seg model weights (.pt file)
            device: Device to run on ("auto", "cuda", "cpu", "mps")
            img_size: Input image size for segmentation model
            landmarks_per_chamber: Dictionary mapping chamber label to number of landmarks.
                                 Default: {50: 24 (LV), 100: 12, 150: 12, 200: 12}
        """
        self.img_size = img_size
        
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ConvNeXtFPN(
            num_classes=len(LABEL_VALUES),
            hub_weights=None
        )
        
        # Load checkpoint
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        try:
            self.model.load_state_dict(state, strict=True)
            print("Loaded checkpoint strictly.")
        except Exception as e:
            print(f"Strict load failed ({e}). Retrying non-strict...")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"Non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
        
        self.model = self.model.to(self.device).eval()
        
        # Initialize frame buffer for temporal consistency
        self.frame_buffer = []
        self.buffer_size = 3  # Use 3 frames for temporal consistency
        
        # Landmark extraction parameters
        self.labels = [50, 100, 150, 200]  # Skip background (0)
        self.n_landmarks = 12
        if landmarks_per_chamber is not None:
            self.landmarks_per_chamber = landmarks_per_chamber
        else:
            self.landmarks_per_chamber = {
                50: 24,   # Left Ventricle: 24 landmarks
                100: 12,  # Right Ventricle: 12 landmarks
                150: 12,  # Left Atrium: 12 landmarks
                200: 12,  # Right Atrium: 12 landmarks
            }
        self.min_area = 50
        
        # Uncertainty parameters
        self.grad_radius = 4
        self.d0 = 5.0
        self.w_dist = 0.5
        self.w_grad = 0.5
        
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for segmentation model.
        
        Args:
            frame: Input grayscale frame (H, W)
            
        Returns:
            Preprocessed tensor (1, 3, img_size, img_size)
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        # Convert to RGB (replicate grayscale to 3 channels)
        rgb_frame = np.stack([resized, resized, resized], axis=-1)
        
        # Normalize using ImageNet statistics (what the model was trained with)
        rgb_frame = rgb_frame.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_frame = (rgb_frame - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).unsqueeze(0).float()
        
        return tensor.to(self.device)
    
    def _postprocess_segmentation(self, logits: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess segmentation logits to mask.
        
        Args:
            logits: Model output (1, num_classes, H, W)
            original_size: Original frame size (H, W)
            
        Returns:
            Segmentation mask with original label values (H, W)
        """
        # Get predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()[0]  # (H, W)
        
        # Resize to original size
        preds_resized = cv2.resize(preds, (original_size[1], original_size[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Map back to original label values
        mask = np.zeros_like(preds_resized, dtype=np.uint8)
        for i, label_val in enumerate(LABEL_VALUES):
            mask[preds_resized == i] = label_val
            
        return mask
    
    def _extract_landmarks_from_mask(self, mask: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """
        Extract landmarks from segmentation mask.
        
        Args:
            mask: Segmentation mask with label values
            frame: Original grayscale frame
            
        Returns:
            List of landmark dictionaries
        """
        landmarks = []
        
        # Compute uncertainty maps
        edges, dist_map, gradmag_norm = compute_uncertainty_maps(frame)
        
        for label in self.labels:
            # Create binary mask for this label
            binary_mask = (mask == label).astype(np.uint8)
            
            if binary_mask.sum() == 0:
                continue
                
            # Find largest contour
            contour = biggest_contour(binary_mask, min_area=self.min_area)
            if contour is None:
                continue
                
            # Sample equally spaced points based on landmarks_per_chamber configuration
            n_points = self.landmarks_per_chamber.get(label, self.n_landmarks)
            pts = sample_equally_spaced_points(contour, n_points)
            if not pts:
                continue
                
            # Extract landmarks with uncertainty
            for x, y in pts:
                uncertainty, _, _, _ = landmark_uncertainty(
                    x, y, dist_map, gradmag_norm,
                    grad_radius=self.grad_radius, d0=self.d0, 
                    w_dist=self.w_dist, w_grad=self.w_grad
                )
                
                landmarks.append({
                    'x': x,
                    'y': y,
                    'label': label,
                    'uncertainty': uncertainty
                })
        
        return landmarks
    
    def _landmarks_to_tensor(self, landmarks: List[Dict], frame_size: Tuple[int, int]) -> torch.Tensor:
        """
        Convert landmarks to tensor format compatible with existing pipeline.
        
        Args:
            landmarks: List of landmark dictionaries
            frame_size: Frame size (H, W)
            
        Returns:
            Tensor of shape (num_landmarks, 4) with [x, y, ere, label] format
        """
        # Group landmarks by label
        landmarks_by_label = {}
        for lm in landmarks:
            label = lm['label']
            if label not in landmarks_by_label:
                landmarks_by_label[label] = []
            landmarks_by_label[label].append(lm)
        
        # Convert to tensor format (x, y, ere, label)
        # Use uncertainty as a proxy for ERE (lower uncertainty = lower ERE)
        # Include label as 4th element for color coding by chamber
        tensor_landmarks = []
        
        for label in self.labels:
            if label in landmarks_by_label:
                label_landmarks = landmarks_by_label[label]
                for lm in label_landmarks:
                    # Use raw uncertainty value (0-1 range), include label for color coding
                    tensor_landmarks.append([lm['x'], lm['y'], lm['uncertainty'], float(label)])
            else:
                # Fill with dummy values if no landmarks for this label
                # Use landmarks_per_chamber configuration
                n_dummy = self.landmarks_per_chamber.get(label, self.n_landmarks)
                for _ in range(n_dummy):
                    tensor_landmarks.append([0, 0, 1.0, float(label)])  # High uncertainty for missing landmarks
        
        return torch.tensor(tensor_landmarks, dtype=torch.float32)
    
    def _compute_classification_color(self, landmarks: List[Dict]) -> Tuple[int, int, int]:
        """
        Compute classification color based on landmark quality.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            RGB color tuple
        """
        if not landmarks:
            return (255, 255, 255)  # White for no landmarks
        
        # Count valid landmarks (low uncertainty)
        valid_landmarks = sum(1 for lm in landmarks if lm['uncertainty'] < 0.5)
        total_landmarks = len(landmarks)
        
        if total_landmarks == 0:
            return (255, 255, 255)
        
        valid_ratio = valid_landmarks / total_landmarks
        
        if valid_ratio > 0.8:
            return (0, 255, 0)  # Green for good quality
        elif valid_ratio > 0.6:
            return (255, 255, 0)  # Yellow for medium quality
        else:
            return (255, 0, 0)  # Red for poor quality
    
    def run_prediction(self, frame: np.ndarray) -> Tuple[Tuple[int, int, int], torch.Tensor]:
        """
        Run the complete 4-chamber landmark pipeline.
        
        Args:
            frame: Input grayscale frame (H, W)
            
        Returns:
            Tuple of (classification_color, landmarks_tensor)
            - classification_color: RGB tuple for visualization
            - landmarks_tensor: Tensor of shape (num_landmarks, 4) with [x, y, ere, label]
        """
        original_size = frame.shape[:2]
        
        # Preprocess frame
        input_tensor = self._preprocess_frame(frame)
        
        # Run segmentation
        with torch.no_grad():
            logits = self.model(input_tensor)
        
        # Postprocess segmentation
        mask = self._postprocess_segmentation(logits, original_size)
        
        # Extract landmarks
        landmarks = self._extract_landmarks_from_mask(mask, frame)
        
        # Convert to tensor format
        landmarks_tensor = self._landmarks_to_tensor(landmarks, original_size)
        
        # Compute classification color
        classification_color = self._compute_classification_color(landmarks)
        
        return classification_color, landmarks_tensor


def configure_fourch_landmark_pipeline(weight_path: str = "weights/4ch-seg-251021.pt", 
                                       device: str = "auto",
                                       landmarks_per_chamber: Optional[Dict[int, int]] = None) -> FourChamberLandmarkPipeline:
    """
    Configure the 4-chamber landmark pipeline.
    
    Args:
        weight_path: Path to the 4ch-seg model weights
        device: Device to run on
        landmarks_per_chamber: Dictionary mapping chamber label to number of landmarks.
                              Default: {50: 24 (LV), 100: 12, 150: 12, 200: 12}
        
    Returns:
        Configured pipeline instance
    """
    return FourChamberLandmarkPipeline(weight_path, device, landmarks_per_chamber=landmarks_per_chamber)


def run_fourch_core_prediction(pipeline: FourChamberLandmarkPipeline, 
                               input_frame: np.ndarray) -> Tuple[Tuple[int, int, int], torch.Tensor]:
    """
    Run core prediction using the 4-chamber landmark pipeline.
    Compatible with the existing landmark prediction interface.
    
    Args:
        pipeline: Configured 4-chamber landmark pipeline
        input_frame: Input grayscale frame
        
    Returns:
        Tuple of (classification_color, landmarks_tensor)
    """
    return pipeline.run_prediction(input_frame)
