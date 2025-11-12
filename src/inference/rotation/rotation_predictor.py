import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

from .dinov3_convnext import load_backbone_from_local_hub, infer_feat_dim, DINOConvNeXt6DDual
from .rotmat import rot6d_to_matrix

class RotationPredictor:
    """Real-time rotation prediction for ultrasound frames"""
    
    def __init__(self, config_path, checkpoint_path, device, dinov3_repo_dir: str | None = None):
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.transform = None
        
        if dinov3_repo_dir is None:
            print("[RotationPredictor] Using default DINOv3 repo path.")
            self.dinov3_repo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dinov3")
        else:
            self.dinov3_repo_dir = dinov3_repo_dir

        self._load_model()


    def _load_model(self):
        """Load the rotation prediction model"""
        try:
            # Load config (simplified - you may need to adjust based on your config format)
            import json
            with open(self.config_path, 'r') as f:
                cfg = json.load(f)
            
            # Use the local DINOv3 repo path
            hub_entry = cfg["model"].get("hub_entry", "dinov3_convnext_base")  # Default to convnext_base

            print(f"🔧 Loading DINOv3 backbone from: {self.dinov3_repo_dir}")
            print(f"🔧 Using hub entry: {hub_entry}")
            
            # Build model
            bb = load_backbone_from_local_hub(
                self.dinov3_repo_dir, 
                hub_entry
            )
            
            feat_dim = infer_feat_dim(bb, cfg["dataset"]["img_size"])
            self.model = DINOConvNeXt6DDual(
                bb, feat_dim, 
                head_dropout=cfg["model"].get("head_dropout", 0.0)
            ).to(self.device)
            
            # Load checkpoint
            print(f"🔧 Loading checkpoint from: {self.checkpoint_path}")
            state = torch.load(self.checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                self.model.load_state_dict(state["model"])
                print(f"✅ Loaded model weights from checkpoint dict")
            else:
                self.model.load_state_dict(state)
                print(f"✅ Loaded model weights directly from checkpoint")
            
            self.model.eval()
            print(f"✅ Model set to evaluation mode")
            
            # Setup image transform
            img_size = cfg["dataset"]["img_size"]
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            print(f"✅ Rotation predictor loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load rotation predictor: {e}")
            self.model = None
    
    def predict_rotation(self, frame):
        """Predict rotation matrix for a single frame"""
        if self.model is None:
            return None
        
        try:
            # Convert frame to PIL Image
            if len(frame.shape) == 2:  # Grayscale
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:  # Already RGB
                frame_rgb = frame
            
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply transform
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                out = self.model(input_tensor, x0=None, return_both=True)
                y6 = out["solo"]
                R_pred = rot6d_to_matrix(y6)
                
            return R_pred.cpu().numpy()[0]  # Return as numpy array
            
        except Exception as e:
            print(f"❌ Rotation prediction failed: {e}")
            return None

def create_rotation_predictor(config_path, checkpoint_path, device):
    """Factory function to create rotation predictor"""
    return RotationPredictor(config_path, checkpoint_path, device)
