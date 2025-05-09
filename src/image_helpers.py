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

    overlay = vis_frame.copy()

    for c, lm in enumerate(landmarks_tensor):
        x, y, ere_val = lm
        if c in [42, 45, 46]:
            ere_val = ere_val / 3
        if ere_val <= ere_threshold:
            color = app_settings.get_color_for_ere(ere_val, max_ere=ere_threshold)
            scaled_x = int(x.item() * scale_x)
            scaled_y = int(y.item() * scale_y)
            cv2.circle(overlay, (scaled_x, scaled_y), app_settings.landmark_radius, color, -1)

    cv2.addWeighted(overlay, app_settings.landmark_alpha, vis_frame, 1 - app_settings.landmark_alpha, 0, vis_frame)

    return vis_frame

def draw_text_on_frame(input_frame, text):
    # Define the font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = input_frame.shape[0] / 400
    font_color = (255, 255, 255)  # White color
    thickness = int(font_scale * 2)

    # Get the text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position to the lower right corner
    text_x = input_frame.shape[1] - text_size[0] - 20  # 20 pixels from the right
    text_y = input_frame.shape[0] - text_size[1] - 20  # 20 pixels from the bottom

    # Draw the text on the frame
    cv2.putText(input_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

    return input_frame
    