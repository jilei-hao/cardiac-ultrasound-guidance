# import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
# import imageio
import matplotlib.colors as mcolors
# import pandas as pd

from landmark_model import ResNet34UNet

SHARED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "GreenYellowRed",
    [
        (0.0, "green"),
        (0.4, "green"),
        (0.7, "yellow"),
        (0.9, "red"),
        (1.0, "red")
    ]
)

def get_color_for_ere(ere):

    # Normalize the ERE value to the range [0, 1]
    normalized = ere / 1.8
    normalized = min(max(normalized, 0), 1)  # Clamp between 0 and 1

    r, g, b, a = SHARED_CMAP(normalized)

    return (int(r * 255), int(g * 255), int(b * 255))


def softmax_2d(logits):
    B, C, H, W = logits.shape
    logits_2d = logits.view(B*C, -1)        
    prob_2d   = F.softmax(logits_2d, dim=1) 
    probs     = prob_2d.view(B, C, H, W)    
    return probs


def compute_ere(prob_map, pred_x, pred_y, threshold=0.03):
    H, W = prob_map.shape
    max_val = prob_map.max()
    cutoff = threshold * max_val

    truncated_map = prob_map.copy()
    mask = (prob_map >= cutoff)
    truncated_map[~mask] = 0.0

    s = truncated_map.sum()
    if s > 0:
        truncated_map /= s
    else:
        return 1e5
      
    
    grid_y, grid_x = np.indices((H, W))
    dist_map = np.sqrt((grid_x - pred_x)**2 + (grid_y - pred_y)**2)
    ERE = np.sum(dist_map * truncated_map)
    return float(ERE)
  
  

  

def predict_landmark(input_frame, ere_threshold=1.5):
  
    # landmark_counts = []
    
    model = ResNet34UNet(num_landmarks=42)
  
    # model_path = "/Users/jileihao/dev/arpa-h-dev/lumify_streaming/weights/bigger_net_final.pth"
    model_path = "/Users/aarongzy/Programming/Penn/PICSL/github_repo/lumify_streaming/weights/bigger_net_final.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    
    H, W = input_frame.shape
    # single-channel => model-channels

    
    

    # Convert grayscale to BGR for overlays
    overlay_img = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)
    # valid_landmarks = 0

    probs_np = run_core_prediction()
    
    for c in range(probs_np.shape[0]):
      heatmap_c = probs_np[c]
      idx = np.argmax(heatmap_c)
      y, x = np.unravel_index(idx, (H, W))

      ere_val = compute_ere(heatmap_c, x, y, threshold=0.03)

 

      # draw only if ERE <= threshold
      if ere_val <= ere_threshold:
          fixed_radius = 1
          color = get_color_for_ere(ere_val)
          cv2.circle(overlay_img, (x, y), fixed_radius, color,-1)
        #   valid_landmarks += 1


      
      # landmark_counts.append(valid_landmarks)

        ## of valid landmarks at top-left
    #   cv2.putText(overlay_img, f"{valid_landmarks}", (5,12),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

  
    return overlay_img

def configure_torch():

    model = ResNet34UNet(num_landmarks=42)
  
    # model_path = "/Users/jileihao/dev/arpa-h-dev/lumify_streaming/weights/bigger_net_final.pth"
    model_path = "/Users/aarongzy/Programming/Penn/PICSL/github_repo/lumify_streaming/weights/bigger_net_final.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def run_core_prediction(model, device, input_frame):
    inp = torch.from_numpy(input_frame).unsqueeze(0).repeat(3,1,1).float()
    inp = inp.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)  # shape => (1, 42, H, W)
    probs = softmax_2d(logits)  # => (1, 42, H, W)
    probs_np = probs.squeeze(0).cpu().numpy()  # => (42, H, W)

    # Convert grayscale to BGR for overlays
    overlay_img = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)
    valid_landmarks = 0
    
    for c in range(probs_np.shape[0]):
      heatmap_c = probs_np[c]
      idx = np.argmax(heatmap_c)
      H,W = 112,112
      y, x = np.unravel_index(idx, (H, W))

      ere_val = compute_ere(heatmap_c, x, y, threshold=0.03)

 

      # draw only if ERE <= threshold
      ere_threshold = 2
      if ere_val <= ere_threshold:
          fixed_radius = 1
          color = get_color_for_ere(ere_val)
          cv2.circle(overlay_img, (x, y), fixed_radius, color,-1)
          valid_landmarks += 1
        
        
      
    #   landmark_counts.append(valid_landmarks)
      cla = (255, 255, 255)
      if valid_landmarks > 25:
          cla = (0, 255, 0)
      elif valid_landmarks > 15:
          cla = (255, 255, 0)
      else:
          cla = (255, 0, 0)

        ## of valid landmarks at top-left
    #   cv2.putText(overlay_img, f"{valid_landmarks}", (5,12),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

  
    return overlay_img, cla