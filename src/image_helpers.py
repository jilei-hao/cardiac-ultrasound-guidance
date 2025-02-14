import numpy as np
import cv2

def draw_circle_on_image(center, radius, image, bgr):
    # Draw a circle on the image
    cv2.circle(image, center, radius, bgr, -1)
    return image

def draw_frame_around_image(image, width, bgr):
    # Draw a rectangle around the image
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), bgr, width)
    return image

def test_drawing():
    # Create a blank RGB image
    image = np.zeros((112, 112, 3), dtype=np.uint8)

    # Define the center and radius of the circle
    center = (90, 22)
    radius = 10
    rgb = (255, 0, 0)  # Red color
    bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR for OpenCV

    # Draw the circle
    image = draw_circle_on_image(center, radius, image, bgr)
    cv2.imshow("Image with Circle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def test_frame():
    # Create a blank RGB image
    image = np.zeros((112, 112, 3), dtype=np.uint8)

    # Define the center and radius of the circle
    width = 5
    rgb = (255, 0, 0)  # Red color
    bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR for OpenCV

    # Draw the circle
    image = draw_frame_around_image(image, width, bgr)
    cv2.imshow("Image with Circle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])

def add_rgb_circle_to_result(input_frame, rgb):
    center = (90, 22)
    radius = 5
    return draw_circle_on_image(center, radius, input_frame, rgb)

def add_rgb_frame_to_result(input_frame, rgb):
    width = 5
    return draw_frame_around_image(input_frame, width, rgb)
    

if __name__ == "__main__":
    test_frame()