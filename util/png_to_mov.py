import os
import sys
import cv2
import argparse

def convert_png_to_mov(input_folder, output_file):
    # Get all PNG files in the input folder
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort()  # Sort files to maintain order

    if not png_files:
        print("No PNG files found in the specified folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(input_folder, png_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # 50ms per frame means 20 frames per second (1000ms / 50ms = 20fps)
    video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))

    # Write each image to the video file
    for png_file in png_files:
        image_path = os.path.join(input_folder, png_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PNG images to MP4 video.")
    parser.add_argument("input_folder", type=str, help="Folder containing PNG images")
    parser.add_argument("output_file", type=str, help="Output MP4 file path")

    args = parser.parse_args()

    convert_png_to_mov(args.input_folder, args.output_file)