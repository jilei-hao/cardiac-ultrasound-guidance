from __future__ import annotations
import math, numpy as np, torch
import torch.nn.functional as F

def rotation_to_6d_np(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:,0], R[:,1]], axis=0).astype(np.float32)

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1 = x[:, 0:3]; a2 = x[:, 3:6]
    b1 = F.normalize(a1, dim=1, eps=1e-6)
    a2_proj = (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(a2 - a2_proj, dim=1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=2)

def geodesic_angle_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    M = torch.matmul(R_pred.transpose(1,2), R_gt)
    tr = M[:,0,0] + M[:,1,1] + M[:,2,2]
    cos = (tr - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos)
    return theta * (180.0 / math.pi)

def frobenius_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    return ((R_pred - R_gt) ** 2).sum(dim=(1,2)).mean()
