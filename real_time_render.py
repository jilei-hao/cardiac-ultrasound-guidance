import mss
import pygetwindow as gw
import numpy as np
import cv2
import time

from AppKit import NSWorkspace
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly

def get_window_bbox(window_name):
    """Find window position by name on macOS"""
    options = kCGWindowListOptionOnScreenOnly
    window_list = CGWindowListCopyWindowInfo(options, 0)

    for win in window_list:
        if window_name.lower() in win.get("kCGWindowName", "").lower():
            bounds = win["kCGWindowBounds"]
            return {
                "left": int(bounds["X"]),
                "top": int(bounds["Y"]),
                "width": int(bounds["Width"]),
                "height": int(bounds["Height"])
            }

    return None  # Window not found

# Example usage
window_bbox = get_window_bbox("Code")  # Change to your app name
if window_bbox:
    print("Window found:", window_bbox)
else:
    print("Window not found.")

if window_bbox:
    sct = mss.mss()
    
    while True:
        screenshot = sct.grab(bbox)  # Capture the app window
        
        # Convert to numpy array
        img = np.array(screenshot)

        # Convert from BGRA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Display (for testing)
        cv2.imshow("App Capture", img)

        # Stream image data (replace with your processing function)
        # process(img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
else:
    print("Failed to locate the window.")
