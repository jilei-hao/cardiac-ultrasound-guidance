import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

def load_pretrained_model(device):
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1)  # Adjust output layer
    # change first layer to take 1 channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # load fine-tuned weights
    # fine_tuned_weights = torch.load('weights/resnet18_multiclass_tuned.pth', map_location=device)
    fine_tuned_weights = torch.load('weights/9_resnet18_multiclass_tuned_regression_ONLYIMAGES_BEST_new.pth', map_location=device)
    model.load_state_dict(fine_tuned_weights)
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return model, preprocess


def run_pose_classifier(model, device, input_image, landmarks, transform):
    model = model.to(device)
    model.eval()
    # input_image is a np array
    # Convert to tensor
    # image = torch.tensor(input_image)
    
    landmarks_image = plot_landmarks_as_images(landmarks)
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
    # # image = torch.from_numpy(landmarks_image).unsqueeze(0).repeat(3,1,1).float()
    # image = landmarks_image.unsqueeze(0).repeat(3,1,1).float()
    # image = image.unsqueeze(0).to(device)
    # # Visual check
    # np_img = image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # plt.imshow(np_img)
    # plt.title("Image passed to model")
    # plt.show()


    
    # image = torch.from_numpy(input_image).unsqueeze(0).repeat(3,1,1).float()
    image = torch.from_numpy(input_image).unsqueeze(0).float()
    image = image.unsqueeze(0).to(device)
    # image = image.to(device)

    # feed image into model
    with torch.no_grad():
        outputs = model(image)
    # print(outputs)
    # _, predicted = torch.max(outputs.data,1)

    return num_to_rgb(outputs.item())


def num_to_rgb(numerical_prediction, num_classes=3):
    numerical_prediction = _regression_to_num_classes(numerical_prediction)
    if numerical_prediction == -1:
        # print("red")
        return (255, 0 ,0)
    elif numerical_prediction == 0:
        # print("yellow")
        return (255, 255, 0)
    elif numerical_prediction == 1:
        # print("green")
        return (0, 255, 0)
    

# def plot_landmarks_as_images(landmarks):
#     # return a 112x112 image with landmarks as a grayscale image
#     device = landmarks.device
#     image_size = (112, 112)
#     image = torch.zeros(image_size, device=device)
#     landmarks_x = landmarks[:, 0]
#     landmarks_y = landmarks[:, 1]
#     landmarks_ere = landmarks[:, 2]

#     # create coord grid
#     y = torch.arange(0, image_size[0], device=device).unsqueeze(1)
#     x = torch.arange(0, image_size[1], device=device).unsqueeze(0)
#     # plot each landmark
#     for i in range(landmarks_x.shape[0]):
#         sigma = 10 / landmarks_ere[i]
#         blob = torch.exp(-((x - landmarks_x[i])**2 + (y - landmarks_y[i])**2) / (2 * sigma**2))
#         image += blob
#     # normalize image
#     image = image / torch.max(image) * 255
#     return image

def plot_landmarks_as_images(landmarks, image_size=(112, 112)):
    device = landmarks.device
    image = torch.zeros(image_size, device=device)

    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    landmarks_error = landmarks[:, 2]

    # create coordinate grid
    x = torch.arange(image_size[1], device=device)
    y = torch.arange(image_size[0], device=device)
    x, y = torch.meshgrid(x, y, indexing='ij')  # (W, H) -> same as numpy meshgrid

    for i in range(len(landmarks_x)):
        sigma = 3 / landmarks_error[i]
        blob = torch.exp(-((x - landmarks_x[i])**2 + (y - landmarks_y[i])**2) / (2 * sigma**2))
        image += blob

    # Normalize twice to [0, 1]
    image = image / torch.max(image)
    image = image.float()
    image = image / torch.max(image) * 255
    # rotate 90 degrees clockwise
    image = torch.rot90(image, -1, [0, 1])
    # flip horizontally
    image = torch.flip(image, [1])

    return image


def _regression_to_num_classes(regression_val):
    if regression_val <= -1:
        return -1 # red
    elif regression_val <= 0:
        return 0 # yellow
    else:
        return 1 # green