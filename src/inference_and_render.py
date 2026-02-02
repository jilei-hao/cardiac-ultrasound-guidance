try:
    import data_access.clarius.clarius_streamer as ClariusStreamer
except:
    print("No clarius streamer found.")
import data_access.simulator.frame_data_simulator as FrameStreamer

import argparse
import time
import os
import json
from datetime import datetime
import numpy as np
import cv2
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional
import queue
from ui import reticle

from pynput import keyboard

class KeyboardHandler:
    """Handles keyboard input with support for detecting held keys."""

    # Keys reserved for control actions (not available as reference labels)
    _RESERVED_KEYS = {'q', 'n', 'p', '.', ','}

    def __init__(self):
        self._pressed_keys = set()
        self._key_events = []    # Queue of key press events for toggle detection
        self._reference_key = ""    # Most recently pressed alphanumeric reference key

        # Start the keyboard listener
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()

    def _on_press(self, key):
        """Called when a key is pressed."""
        try:
            key_char = key.char if hasattr(key, 'char') else None
        except AttributeError:
            key_char = None

        # Track special keys
        if key == keyboard.Key.space:
            self._pressed_keys.add('space')
        elif key == keyboard.Key.enter:
            self._key_events.append('enter')
        elif key_char:
            ch = key_char.lower()
            self._pressed_keys.add(ch)
            if ch in self._RESERVED_KEYS:
                self._key_events.append(ch)
            elif ch not in self._RESERVED_KEYS:
                self._reference_key = ch

    def _on_release(self, key):
        """Called when a key is released."""
        try:
            key_char = key.char if hasattr(key, 'char') else None
        except AttributeError:
            key_char = None

        # Remove from pressed keys
        if key == keyboard.Key.space:
            self._pressed_keys.discard('space')
        elif key_char:
            ch = key_char.lower()
            self._pressed_keys.discard(ch)
            if ch == self._reference_key:
                self._reference_key = ""

    def is_held(self, key_name: str) -> bool:
        """Check if a key is currently held down."""
        return key_name.lower() in self._pressed_keys

    def get_reference_label(self) -> str:
        """Get the currently held reference label, or "" if none."""
        return self._reference_key

    def get_event(self) -> Optional[str]:
        """Get the next key event (for toggle keys like enter, q)."""
        if self._key_events:
            return self._key_events.pop(0)
        return None

    def stop(self):
        """Stop the keyboard listener."""
        self._listener.stop()


def main():
    parser = argparse.ArgumentParser(description="reads and vizualizes (TODO: inferenced) images from a data source")
    parser.add_argument("--interval", type=int, default=100, help="Interval in ms between frames")
    parser.add_argument('--data_source', type=str, default='frame_data', help='Type of data source')
    parser.add_argument('--data_path', type=str, required=True, help='Path to pngs and csv for saved data')
    options = parser.parse_args()

    # Initialize keyboard handler
    kb_handler = KeyboardHandler()

    print("Controls:")
    print("    [P]            - Play/Pause playback")
    print("    [.] (period)   - If paused, go to next frame")
    print("    [,] (comma)    - If paused, go to previous frame")
    print("    [Q]            - Quit")
    print("")

    dataset = FrameStreamer.FrameDataSimulator(
        os.path.join(options.data_path, 'images'),
        os.path.join(options.data_path, 'frame.meta.csv'),
    )
    dataset.start()

    target_pose = dataset.get_frame(index=0, no_image=True).tracker

    # init vars for playback query of dataset
    relative_time_ns = 0
    last_query_time = time.time_ns()
    pause_playback = True
    current_frame_index = 0

    try:
        while True:
            # Check for key events
            event = kb_handler.get_event()

            if event == 'q':
                print("-- Quitting...")
                break
            
            if event == 'p':
                pause_playback = not pause_playback
                if pause_playback is True:
                    current_frame_index = dataset.get_current_frame_index()
                elif pause_playback is False:
                    last_query_time = time.time_ns()
                print("-- realtime playback is paused:", pause_playback)
            
            if event == '.':
                current_frame_index = current_frame_index + 1
                if current_frame_index >= dataset.get_total_frames():
                    current_frame_index = 0
                # get and save relative_time for that frame
                relative_time_ns = dataset.get_frame(index=current_frame_index, no_image=True).timestamp - dataset.start_timestamp
                last_query_time = time.time_ns()
                print('-- Next frame; index:', current_frame_index)

            if event == ',':
                current_frame_index = current_frame_index - 1
                if current_frame_index < 0:
                    current_frame_index = dataset.get_total_frames() - 1
                # get and save relative_time for that frame
                relative_time_ns = dataset.get_frame(index=current_frame_index, no_image=True).timestamp - dataset.start_timestamp
                last_query_time = time.time_ns()
                print('-- Prev frame; index:', current_frame_index)

            # Fetch frame
            if pause_playback:
                frame = dataset.get_frame(index=current_frame_index)
            else:
                # update relative time
                current_query_time = time.time_ns()
                relative_time_ns += current_query_time - last_query_time
                last_query_time = current_query_time
                # query frame from that timestamp
                frame = dataset.get_frame(relative_time=relative_time_ns)
                current_frame_index = dataset.get_current_frame_index()
            
            # Convert to grayscale numpy array
            if frame.image is not None:
                image = np.array(frame.image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = None


            # TODO: Inference the model here
            pose = frame.tracker
            pose_reticle = reticle.render_pose(target_pose, pose)
            pose_reticle = np.array(pose_reticle)


            # print(relative_time_ns)
            # print(dataset.get_current_frame_index())
            cv2.imshow('output',  image)
            cv2.imshow('reticle', pose_reticle)
            cv2.waitKey(1)

            # Sleep for the specified interval
            time.sleep(options.interval / 1000.0)

    except KeyboardInterrupt:
        print("\n-- Interrupted by user")

    finally:
        # Cleanup
        print("-- Shutting down...")

        # Stop keyboard handler
        kb_handler.stop()

        cv2.destroyAllWindows()

        # Stop Clarius streamer
        # ClariusStreamer.stop()

        print("-- Done.")


if __name__ == "__main__":
    main()
