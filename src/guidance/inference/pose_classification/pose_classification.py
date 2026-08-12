"""Legacy image-only traffic-light classifier used by cardiac guidance."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.models import resnet18


@dataclass(frozen=True)
class PoseResult:
    """One continuous quality score and its discrete RGB classification."""

    score: float
    rgb: tuple


def load_pretrained_model(device, weight_path):
    if not weight_path:
        raise ValueError("[pose_classification] weight_path cannot be empty")

    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.conv1 = torch.nn.Conv2d(
        1,
        64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device).eval()


def run_pose_classifier(model, device, input_image):
    image = (
        torch.from_numpy(input_image)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )
    if image.shape[-2:] != (112, 112):
        image = F.interpolate(
            image,
            size=(112, 112),
            mode="bilinear",
            align_corners=False,
        )

    image = image / torch.max(image)
    image = F.interpolate(
        image, size=(256, 256), mode="bilinear", align_corners=False
    )
    with torch.no_grad():
        score = model(image).item()
    return PoseResult(score=score, rgb=num_to_rgb(score))


def num_to_rgb(score):
    if score <= -1:
        return (255, 0, 0)
    if score <= 0:
        return (255, 255, 0)
    return (0, 255, 0)
