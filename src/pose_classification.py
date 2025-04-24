import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

def load_pretrained_model(device):
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)  # Adjust output layer
    # change first layer to take 1 channel
    # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # load fine-tuned weights
    fine_tuned_weights = torch.load('weights/resnet18_multiclass_tuned.pth', map_location=device)
    # fine_tuned_weights = torch.load('weights/resnet18_multiclass_tuned_new.pth', map_location=device)
    model.load_state_dict(fine_tuned_weights)
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return model, preprocess


def run_pose_classifier(model, device, input_image, transform):
    model = model.to(device)
    model.eval()
    # input_image is a np array
    # Convert to tensor
    # image = torch.tensor(input_image)
    
    # apply pytorch transform to the image
    # image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    # image = torch.stack((image, image, image), dim=2)
    # image = image.squeeze(-1)
    # image = image.permute(2, 0, 1)
    # image = transform(image)
    # # add a 1 to the 0th dimension
    # image = image.unsqueeze(0)

    # # ------ checking orientation of image passed to classifier ------
    # # Convert grayscale to 3-channel tensor
    # image = torch.from_numpy(input_image).unsqueeze(0).repeat(3,1,1).float()
    # image = image.unsqueeze(0).to(device)

    # # Visual check
    # np_img = image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # plt.imshow(np_img)
    # plt.title("Image passed to model")
    # plt.show()


    
    image = torch.from_numpy(input_image).unsqueeze(0).repeat(3,1,1).float()
    image = image.unsqueeze(0).to(device)
    # image = image.to(device)

    # feed image into model
    with torch.no_grad():
        outputs = model(image)
    # print(outputs)
    _, predicted = torch.max(outputs.data,1)

    return num_to_rgb(predicted)


def num_to_rgb(numerical_prediction, num_classes=3):
    if numerical_prediction == 0:
        return (0, 255, 0)
    elif numerical_prediction == 1:
        return (255, 255, 0)
    elif numerical_prediction == 2:
        return (255, 0, 0)