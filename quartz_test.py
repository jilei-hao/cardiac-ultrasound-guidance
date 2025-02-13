import Quartz
import numpy as np
import cv2
import time
from PIL import Image

def capture_screen():
    # Get main display size
    display_id = Quartz.CGMainDisplayID()
    width = Quartz.CGDisplayPixelsWide(display_id)
    height = Quartz.CGDisplayPixelsHigh(display_id)

    # Capture screen as a CGImage
    image_ref = Quartz.CGDisplayCreateImage(display_id)

    if image_ref is None:
        print("Failed to capture screen.")
        return None

    # Convert to PIL Image
    image = Image.frombytes(
        "RGB",
        (width, height),
        Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image_ref))
    )

    return np.array(image)

# Stream at high speed
while True:
    start_time = time.time()
    frame = capture_screen()
    
    if frame is None:
        break
    
    # Display using OpenCV (for testing)
    cv2.imshow("Mac Screen Capture", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Calculate FPS
    print(f"FPS: {1 / (time.time() - start_time):.2f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
