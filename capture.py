import os
import time
import mss
import cv2
import numpy as np
import pygetwindow as gw
import re

# Directory where screenshots will be saved
# Planets 14-16 Zoom 1
OUTPUT_DIR = "annotation_screenshots/planets/locked/Planets 14-16 Zoom 6"
# Number of screenshots to capture
NUM_SCREENSHOTS = 200
# Interval between screenshots in seconds
INTERVAL_SECONDS = 0.1
# Emulator window title (e.g., "MEmu")
EMULATOR_WINDOW_TITLE = "MEmu"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_latest_screenshot_index(directory):
    """
    Get the latest screenshot index from the filenames in the directory.
    """
    pattern = re.compile(r"screenshot_(\d+)\.png")
    indices = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            indices.append(int(match.group(1)))

    return max(indices) if indices else 0

def get_emulator_window(emulator_title):
    """
    Get the emulator window by its title.
    """
    try:
        windows = gw.getWindowsWithTitle(emulator_title)
        if not windows:
            raise IndexError
        window = windows[0]
        time.sleep(2)  # Wait briefly to ensure the window is ready
        window.restore()  # Restore if minimized
        window.activate()  # Bring it to the foreground
        time.sleep(2)
        window.activate()  # Bring it to the foregro
        time.sleep(2)
        return window
    except IndexError:
        print(f"{emulator_title} window not found.")
        return None

def capture_window(save_path, window):
    """
    Capture the specified window and save it to the specified path.
    """
    if window:
        bbox = (window.left, window.top, window.right, window.bottom)
        with mss.mss() as sct:
            screenshot = sct.grab(bbox)  # Capture the specified window region
            img = np.array(sct.grab(bbox))  # Convert the screenshot to a NumPy array
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR format for OpenCV
            cv2.imwrite(save_path, img)  # Save the screenshot
    else:
        print("Window not found. Unable to capture screenshot.")

def main():
    # Get the emulator window
    window = get_emulator_window(EMULATOR_WINDOW_TITLE)
    if not window:
        print("Exiting because the emulator window was not found.")
        return

    # Determine the starting index based on existing screenshots
    start_index = get_latest_screenshot_index(OUTPUT_DIR) + 1

    print(f"Starting to capture {NUM_SCREENSHOTS} screenshots at {INTERVAL_SECONDS}-second intervals...")
    for i in range(NUM_SCREENSHOTS):
        # Define the path where the screenshot will be saved
        screenshot_path = os.path.join(OUTPUT_DIR, f"screenshot_{start_index + i:04d}.png")
        
        # Capture the window and save the screenshot
        capture_window(screenshot_path, window)
        print(f"Captured screenshot {start_index + i}/{NUM_SCREENSHOTS + start_index - 1}: {screenshot_path}")

        # Wait for the specified interval before capturing the next screenshot
        time.sleep(INTERVAL_SECONDS)

    print("Screenshot capturing complete!")

if __name__ == "__main__":
    main()
