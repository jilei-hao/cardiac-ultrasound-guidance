import sys
import time

# Try importing PyQt5/PySide2 if available for event processing
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    HAS_QT = True
except ImportError:
    try:
        from PySide2.QtWidgets import QApplication
        from PySide2.QtCore import QTimer
        HAS_QT = True
    except ImportError:
        HAS_QT = False
        print("Warning: No Qt bindings found. Callbacks may not work properly on Windows.")

import data_access.clarius.pyclariuscast as pyclariuscast
from data_access.clarius import clarius_streamer

def main():
    ip = "192.168.1.1"
    port = 5828
    path = 'research'

    if HAS_QT:
        app = QApplication(sys.argv)
        
        # Configure and start
        clarius_streamer.configure(ip, port, path, 512, 512)
        clarius_streamer.start()
        clarius_streamer.set_print_debug_info(True)
        
        # Timer to check for frames
        def check_frames():
            frame = clarius_streamer.get_frame()
            if frame is not None:
                print(f"Received frame: {frame.timestamp}")
        
        timer = QTimer()
        timer.timeout.connect(check_frames)
        timer.start(100)  # Check every 100ms
        
        # Run Qt event loop
        try:
            sys.exit(app.exec_())
        except KeyboardInterrupt:
            clarius_streamer.stop()
    else:
        # Fallback without Qt
        clarius_streamer.configure(ip, port, path, 512, 512)
        clarius_streamer.start()
        clarius_streamer.set_print_debug_info(True)
        
        print("Running without Qt event loop...")
        try:
            while True:
                time.sleep(0.01)
                frame = clarius_streamer.get_frame()
                if frame is not None:
                    print(f"Received frame: {frame.timestamp}")
        except KeyboardInterrupt:
            clarius_streamer.stop()

if __name__ == "__main__":
    main()