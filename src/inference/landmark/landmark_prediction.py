# import os
import torch
import torch.nn.functional as F
import numpy as np

from .landmark_model import ResNet34UNet


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


def configure_torch(device):

    model = ResNet34UNet(num_landmarks=47)
  
    # model_path = "/Users/jileihao/dev/arpa-h-dev/lumify_streaming/weights/bigger_net_final.pth"
    model_path = "weights/landmark_prediction.pth"

    

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def run_core_prediction(model, device, input_frame):
    inp = torch.from_numpy(input_frame).unsqueeze(0).repeat(3,1,1).float()
    inp = inp.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)  # shape => (1, 42, H, W)
    probs = softmax_2d(logits)  # => (1, 42, H, W)
    probs_np = probs.squeeze(0).cpu().numpy()  # => (42, H, W)

    # Convert grayscale to BGR for overlays
    valid_landmarks = 0
    
    landmarks_x, landmarks_y, landmarks_ere = [], [], []
    for c in range(probs_np.shape[0]):
      heatmap_c = probs_np[c]
      idx = np.argmax(heatmap_c)
      H,W = 112,112
      y, x = np.unravel_index(idx, (H, W))

      ere_val = compute_ere(heatmap_c, x, y, threshold=0.03)
      landmarks_x.append(x)
      landmarks_y.append(y)
      landmarks_ere.append(ere_val)

      ere_threshold = 6

      if ere_val <= ere_threshold:
        if c in [42, 45, 46]:
          ere_val = ere_val / 3
          
        valid_landmarks += 1

    # return landmark coords and ere as tensor
    landmarks_tensor = torch.tensor([landmarks_x, landmarks_y, landmarks_ere]).T
        
    # landmark_counts.append(valid_landmarks)
    cla = (255, 255, 255)
    if valid_landmarks > 30:
      cla = (0, 255, 0)
    elif valid_landmarks > 25:
      cla = (255, 255, 0)
    else:
      cla = (255, 0, 0)

    return cla, landmarks_tensor
