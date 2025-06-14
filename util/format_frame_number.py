def format_frame_number(frame_number: int) -> str:
    """
    Formats a frame number as a zero-padded string with a length of 4 characters.
    
    Args:
        frame_number (int): The frame number to format.
        
    Returns:
        str: The formatted frame number as a zero-padded string.
    """
    return f"{frame_number:04d}"

def replace_file_frame_number(filename: str) -> str:
    """
    Replaces the frame number in a filename with a zero-padded string.
    
    Args:
        filename (str): The original filename containing a frame number.
        
    Returns:
        str: The filename with the frame number replaced by a zero-padded string.
    """
    parts = filename.split('_')
    if len(parts) > 1 and parts[1].isdigit():
        parts[1] = format_frame_number(int(parts[1]))
        return '_'.join(parts)
    return filename


def main(folder_path: str):
    """
    Formats all frame numbers in filenames within a specified folder.
    
    Args:
        folder_path (str): The path to the folder containing files with frame numbers.
    """
    import os
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            new_filename = replace_file_frame_number(filename)
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            print(f"Renamed '{filename}' to '{new_filename}'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Format frame numbers in filenames.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing files with frame numbers.")
    
    args = parser.parse_args()
    
    main(args.folder_path)