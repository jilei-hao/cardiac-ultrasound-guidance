import numpy as np
import cv2

def draw_circle_on_image(center, radius, image, bgr):
    # Draw a circle on the image
    cv2.circle(image, center, radius, bgr, -1)
    return image

def draw_border_around_image(image, width, bgr):
    # Draw a rectangle around the image
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), bgr, width)
    return image

def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])

def add_rgb_circle_to_result(input_frame, rgb):
    center = (90, 22)
    radius = 5
    return draw_circle_on_image(center, radius, input_frame, rgb)

def add_rgb_border_to_result(input_frame, rgb):
    width = 25
    return draw_border_around_image(input_frame, width, rgb)


def overlay_landmarks(vis_frame, landmarks_tensor, app_settings, ere_threshold=6):
    scale_x = vis_frame.shape[1] / 112
    scale_y = vis_frame.shape[0] / 112

    for lm in landmarks_tensor:
        x, y, ere_val = lm
        if ere_val <= ere_threshold:
            color = app_settings.get_color_for_ere(ere_val, max_ere=ere_threshold)
            scaled_x = int(x.item() * scale_x)
            scaled_y = int(y.item() * scale_y)
            cv2.circle(vis_frame, (scaled_x, scaled_y), 15, color, -1)
            cv2.addWeighted(vis_frame, alpha=0.5, src2=vis_frame, beta=0.5, gamma=0)

    return vis_frame
