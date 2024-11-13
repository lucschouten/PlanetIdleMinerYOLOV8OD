import pygetwindow as gw
import pyautogui
from PIL import ImageGrab
import numpy as np
import cv2
from ultralytics import YOLO
import pytesseract
import time
import re
from PIL import Image
import matplotlib.pyplot as plt  # Importing matplotlib to display images inlin
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import pytesseract
import re
import matplotlib.pyplot as plt

class PlanetMiner:
    def __init__(self, ):
        # Target dimensions based on specified corners
        self.width = 800
        self.height = 800
        self.window = self.get_window()
        self.set_window_size()
        self.window_properties = self.get_window_properties()
        # self.model = YOLO('Models/planets.pt')
        self.model = YOLO('Models/icons_and_menus.pt')
        self.zoom_set = False
        # self.set_zoom()
        #print model class names
        # print(self.model.names)
        # Load the YOLOv8 model once during initialization
        self.networthmodel = YOLO('Models/networth.pt')
        self.networthmodel_class_names = {0: "Cash", 1: "DarkMatter", 2: "GalaxyCredits", 3: "GalaxyValue"}  # Class names mapping

    def get_window(self, emulator="MEmu"):
        """
        Select the emulator window by its title.
        """
        try:
            windows = gw.getWindowsWithTitle(emulator)
            if not windows:
                raise IndexError
            window = windows[0]
            time.sleep(1)  # Wait briefly to ensure the window is ready
            window.restore()  # Restore if minimized
            window.activate()  # Bring it to the foreground
            return window
        except IndexError:
            print("MEmu window not found.")
            return None

    def set_window_size(self):
        """
        Resize and position the emulator window to the target dimensions.
        """

        if self.window:
            #move to the top left corner
            self.window.moveTo(0, 0)
            current_width = self.window.width
            current_height = self.window.height

            if (self.width != current_width or self.height != current_height):
                self.window.activate()  # Bring the window to the foreground
                self.window.resizeTo(self.width, self.height)  # Set window size
                print(f"Set window size to {self.width}x{self.height}.")
            else:
                print("Window is already at the specified size and position.")
        else:
            print("Cannot set window size because MEmu window was not found.")

    def set_zoom(self, zoom_factor=6):
        "Set the zoom out level of the emulator window."
        if not self.zoom_set and self.window:
            #center mouse position in the middle of the screen
            pyautogui.moveTo(self.window.left + self.window.width//2, self.window.top + self.window.height//2)
            #ZOom in fully first
            #hold ctrl
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(20)  # Zoom in by scrolling up
            #
            pyautogui.keyDown('ctrl')
            for i in range(zoom_factor):
                pyautogui.scroll(-1)  # Zoom out by scrolling up
            # pyautogui.scroll(-zoom_factor)  # Zoom out by scrolling down
            #release ctrl
            pyautogui.keyUp('ctrl')
            #determine the mouse position to zoom out from the center
            self.zoom_set = True    

    def get_window_properties(self):
        """
        Retrieve the corner coordinates of the emulator window.
        """
        if self.window:
            top_left = (self.window.left, self.window.top)
            bottom_right = (self.window.left + self.window.width, self.window.top + self.window.height)
            print("MEmu window corners:", {
                "top_left": top_left,
                "bottom_right": bottom_right
            })

            return {
                "top_left": top_left,
                "bottom_right": bottom_right
            }
        else:
            print("Window properties cannot be retrieved because MEmu window was not found.")
            return None

    def take_screenshot_and_predict(self, filename="predicted_screenshot.png"):
        """
        Takes a screenshot of the MEmu window, performs predictions using YOLO, and saves the annotated image.
        """
        if self.window_properties:
            # Define the bounding box
            top_left = self.window_properties["top_left"]
            bottom_right = self.window_properties["bottom_right"]
            bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

            try:
                # Capture the specified area
                screenshot = ImageGrab.grab(bbox)
                
                # Convert to OpenCV format
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                # Use the YOLO model to predict objects in the frame
                results = self.model.predict(frame, imgsz=640, conf=0.7)

                # Annotate the frame with the prediction results
                annotated_frame = results[0].plot()

                # Save the annotated frame to the specified file
                cv2.imwrite(filename, annotated_frame)
                print(f"Prediction saved as {filename}")
                
            except Exception as e:
                print(f"Failed to take screenshot and predict: {e}")
        else:
            print("Cannot take a screenshot because window properties were not found.")

    def get_positions(self, class_id, model):
        """
        Get the positions (bounding boxes) of the detected objects based on the class ID.
        """
        if self.window_properties:
            # Capture a screenshot of the defined window
            top_left = self.window_properties["top_left"]
            bottom_right = self.window_properties["bottom_right"]
            bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

            try:
                # Capture the specified area
                screenshot = ImageGrab.grab(bbox)
                # Convert to OpenCV format
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                # Use the YOLO model to predict objects in the frame
                results = model.predict(frame, imgsz=640, conf=0.7)
                positions = []
                for det in results[0].boxes:
                    detected_class_id = int(det.cls[0])
                    if detected_class_id == class_id:
                        # Get bounding box coordinates in (x1, y1, x2, y2) format
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        positions.append((x1, y1, x2, y2))
                return positions
            except Exception as e:
                print(f"Failed to get positions: {e}")
                return []
        else:
            print("Cannot get positions because window properties were not found.")
            return []



    def get_networth(self):
        """
        Get the net worth of the player from the screenshot.
        Make sure the values are visible on the screen. If not zoom out (Ctrl + Mouse Wheel) to make them visible.
        """
        if self.window_properties:
            # Capture a screenshot of the defined window
            top_left = self.window_properties["top_left"]
            bottom_right = self.window_properties["bottom_right"]
            bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
            screenshot = ImageGrab.grab(bbox)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Use YOLO model to detect icons in the screenshot
            results = self.networthmodel.predict(frame, imgsz=640, conf=0.3)
            networth_data = {}

            for det in results[0].boxes:
                x, y, w, h = map(int, det.xywh[0])  # Bounding box (center x, center y, width, height)
                class_id = int(det.cls[0])
                icon_name = self.networthmodel_class_names.get(class_id, "Unknown")

                # Define initial crop area closer to the detected icon
                crop_x1 = x + w // 2 + 2    # Reduced offset to start closer to the icon
                crop_y1 = y - h // 2
                initial_crop_width = 100     # Initial width for cropping text area
                crop_y2 = y + h // 2 + 8    # Increase the height by adding 8 pixels to the bottom

                extracted_text = ""
                current_width = initial_crop_width

                # Dynamically expand the width of the crop area to capture text of variable lengths
                while True:
                    crop_x2 = crop_x1 + current_width
                    # Ensure crop region is within image bounds
                    if crop_x2 > frame.shape[1]:
                        crop_x2 = frame.shape[1]

                    # Crop the area containing the text
                    value_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    # Check if the cropped region is valid (non-empty)
                    if value_crop.size == 0:
                        print(f"Skipping OCR for {icon_name}: Empty crop region.")
                        networth_data[icon_name] = "OCR failed or empty region"
                        break

                    # Step 1: Upscale the cropped image to enhance resolution
                    scale_percent = 300  # Scale up by 300%
                    width = int(value_crop.shape[1] * scale_percent / 100)
                    height = int(value_crop.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    upscaled = cv2.resize(value_crop, dim, interpolation=cv2.INTER_CUBIC)

                    # Step 2: Convert to grayscale
                    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

                    # Step 3: Apply a sharpening filter
                    kernel = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
                    sharpened = cv2.filter2D(gray, -1, kernel)

                    # Display the sharpened image for debugging
                    # plt.imshow(sharpened, cmap="gray")
                    # plt.title(f"Sharpened area for {icon_name}")
                    # plt.axis("off")
                    # plt.show()

                    # Convert the sharpened image to a PIL image for OCR
                    value_pil = Image.fromarray(sharpened)

                    # OCR the sharpened grayscale image to extract text
                    raw_text = pytesseract.image_to_string(value_pil, config='--psm 6 --oem 3 -c tessedit_char_whitelist="0123456789Ee+.-"').strip()

                    # Strict cleaning of OCR output to retain only valid characters
                    cleaned_text = re.sub(r'[^0-9Ee+.-]', '', raw_text).strip()  # Retain digits, '.', 'E', '+', and '-'

                    # print(f"Raw OCR result for {icon_name}: '{raw_text}'")
                    # print(f"Cleaned OCR result for {icon_name}: '{cleaned_text}'")

                    # Handle scientific notation if detected
                    if re.match(r'^\d+(\.\d+)?[eE][+-]?\d+$', cleaned_text):
                        try:
                            # Convert scientific notation to an integer or float as a string
                            extracted_text = str(int(float(cleaned_text)))
                            # print(f"Converted scientific notation to integer: {extracted_text}")
                        except ValueError:
                            networth_data[icon_name] = "OCR failed or invalid format"
                            break
                    else:
                        # If no scientific notation, just take cleaned text
                        extracted_text = cleaned_text

                    # Validate the cleaned text to ensure it's a valid number
                    if re.match(r'^\d+$', extracted_text):
                        networth_data[icon_name] = extracted_text
                        break
                    elif current_width >= 160:  # Stop if the crop width gets too large
                        networth_data[icon_name] = "OCR failed or invalid format"
                        break
                    else:
                        current_width += 10  # Expand the crop width incrementally

                # Print the final OCR result for each icon
                # print(f"Final extracted value for {icon_name}: {networth_data[icon_name]}")

            print("Net worth data:", networth_data)
            return networth_data
        else:
            print("Cannot retrieve net worth because window properties were not found.")
            return None

class Planets(PlanetMiner):
    def __init__(self):
        super().__init__()
        self.planet_model = YOLO('Models/planets.pt')
        self.planet_locked_model = YOLO('Models/locked_planets.pt')
        self.class_names = self.planet_model.names  # Get class names mapping from the model
        self.class_names_locked = self.planet_locked_model.names  # Get class names mapping from the model
        print("Planet class names:", self.class_names)
        # print("Locked planet class names:", self.class_names_locked)

    def find_locked_planets(self):
        """
        Find the locked planets on the screen and return their positions.
        """
        # Capture screenshot of the screen or specific region
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)  # Convert to numpy array
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)  # Convert colors for OpenCV
        
        # Predict locked planets on the screenshot
        results = self.planet_locked_model.predict(screenshot)
        
        # Extract positions of locked planets and put them in a dictionary for each planet
        locked_positions = {}
        for box in results[0].boxes:  # Access the first (and only) element in results
            # Each box contains information about position and class id
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract the bounding box coordinates
            class_id = int(box.cls[0])  # Get class ID from box.cls
            class_name = self.class_names_locked[class_id]  # Map class ID to name
            # print(f"Detected locked planet: {class_name} at ({x1}, {y1}) - ({x2}, {y2})")
            
            locked_positions[class_name] = (x1, y1, x2, y2)

        return locked_positions

    def purchase_locked_planet(self, locked_position, planet_name):
        """
        Purchase a locked planet by clicking on the specified position.
        """
        if locked_position:
            if self.can_purchase(planet_name):
                # Click the center of the locked planet to purchase it
                x_center = (locked_position[0] + locked_position[2]) // 2
                y_center = (locked_position[1] + locked_position[3]) // 2
                pyautogui.moveTo(x_center, y_center)
                pyautogui.click(x_center, y_center)
                print("Purchased locked planet.")
        else:
            print("No locked planet position provided. Cannot purchase.")
        return
    
    def can_purchase(self, planet_name):
        """Checks if we can purchase a planet based on the price and our networth cash"""
        planet_purchase_prices = {"Balor_locked": 100, "Drasta_locked": 200, "Anadius_locked": 500, "Dholen_locked": 1250}
        networth = self.get_networth()
        if networth:
            # Get the cash value from the net worth data
            cash = int(networth.get("Cash", 0))
            print("Cash:", cash)
            planet_price = planet_purchase_prices.get(planet_name, 0)
            
            # Check if we can afford the planet
            return cash >= planet_price
        return False


    def purchase_next(self):
        """
        Purchase the next locked planet if available and purchaseable.
        """
        locked_positions = self.find_locked_planets()
        print("Locked planet positions:", locked_positions)
        
        # Define the purchase order for planets (based on price or priority)
        purchase_order = ["Balor_locked", "Drasta_locked", "Anadius_locked", "Dholen_locked"]
        
        # Attempt to purchase the first affordable planet in the purchase order
        for planet_name in purchase_order:
            if planet_name in locked_positions:
                position = locked_positions[planet_name]
                
                # Check if we can afford the planet
                if self.can_purchase(planet_name):
                    self.purchase_locked_planet(position, planet_name)
                    print(f"Purchased {planet_name}")
                else:
                    print(f"Cannot afford {planet_name}.")
                    return False
                
        
        print("Could either purchase all planets or there were no planets it was not able to afford.")
        return True
    
    def find_planet(self, planet_name):
        """
        Use the self.planet_model to find all planets and return the positions of the specified planet_name.

        Parameters:
        - planet_name (str): The name of the planet to find. 
        {0: 'Acheron -8-', 1: 'Anadius -3-', 2: 'Balor -1-', 3: 'Batalla -14-', 4: 'Castellus -17-', 5: 'Dholen -4-', 6: 'Drasta -2-', 7: 'Gorgon -18-', 8: 'Imir -11-', 9: 'Micha -15-', 10: 'Newton -6-', 11: 'Nith -13-', 12: 'No Object -0-', 13: 'Paranitha -19-', 14: 'Pranas -16-', 15: 'Relic -12-', 16: 'Solveig -10-', 17: 'Verr -5-', 18: 'Widow -7-', 19: 'Yangtze -9-'}

        Returns:
        - positions (list of tuples): A list of bounding boxes (x1, y1, x2, y2) for each detected instance of the planet.
        - Returns None if the planet is not found or an error occurs.
        """
        if self.window_properties:
            # Define the bounding box for the emulator window
            top_left = self.window_properties["top_left"]
            bottom_right = self.window_properties["bottom_right"]
            bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

            try:
                # Capture the specified area
                screenshot = ImageGrab.grab(bbox)
                # Convert to OpenCV format
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                # Use the YOLO model to predict objects in the frame
                results = self.planet_model.predict(frame, imgsz=640, conf=0.7)

                positions = []
                for det in results[0].boxes:
                    class_id = int(det.cls[0])
                    class_name = self.class_names.get(class_id, "Unknown")
                    if planet_name in class_name:
                        # Get bounding box coordinates in (x1, y1, x2, y2) format
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        # Adjust coordinates to screen position
                        x1 += self.window.left
                        y1 += self.window.top
                        x2 += self.window.left
                        y2 += self.window.top
                        positions.append((x1, y1, x2, y2))

                if positions:
                    print(f"Found {len(positions)} instance(s) of planet '{planet_name}'.")
                    return positions
                else:
                    print(f"Planet '{planet_name}' not found.")
                    return None
            except Exception as e:
                print(f"Failed to find planet '{planet_name}': {e}")
                return None
        else:
            print("Cannot find planet because window properties were not found.")
            return None
    
    def open_planet_menu(self, planet_name):
        """
        Open the menu for a specific planet by clicking on the planet icon.
        """
        positions = self.find_planet(planet_name)
        if positions:
            # Click the center of the first detected planet
            x_center = (positions[0][0] + positions[0][2]) // 2
            y_center = (positions[0][1] + positions[0][3]) // 2
            pyautogui.click(x_center, y_center)
            print(f"Opened menu for planet '{planet_name}'.")
            return True
        else:
            print(f"Cannot open menu for planet '{planet_name}'.")
            return False
class PlanetMenu(PlanetMiner):
    def __init__(self):
        super().__init__()  # Initialize PlanetMiner to get window properties
        self.model = YOLO('Models/planet_menu.pt')
        self.class_names = self.model.names  # Get class names mapping from the model
        # print(self.class_names)
        print("Model class names:", self.class_names)

        #Class variables
        self.current_menu = None

    def is_open(self):
        """
        Check if the planet menu is open by detecting the 'Exit_menu_button'.
        Returns True if open, False otherwise.
        """
        if self.window_properties:
            # Capture a screenshot of the defined window
            top_left = self.window_properties["top_left"]
            bottom_right = self.window_properties["bottom_right"]
            bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
            try:
                # Capture the specified area
                screenshot = ImageGrab.grab(bbox)
                # Convert to OpenCV format
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                # Use the YOLO model to predict objects in the frame
                results = self.model.predict(frame, imgsz=640, conf=0.9)
                for det in results[0].boxes:
                    class_id = int(det.cls[0])
                    class_name = self.class_names.get(class_id, "Unknown")
                    if class_name == "Exit_menu_button":
                        return True
                return False  # 'Exit_Menu_button' not found
            except Exception as e:
                print(f"Failed to check if menu is open: {e}")
                return False
        else:
            print("Cannot check if menu is open because window properties were not found.")
            return False
    def close_menu(self):
        """
        Check if the planet menu is open by detecting the 'Exit_menu_button'.
        Returns True if open, False otherwise.
        """
        if self.window_properties:
            # Capture a screenshot of the defined window
            top_left = self.window_properties["top_left"]
            bottom_right = self.window_properties["bottom_right"]
            bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
            try:
                # Capture the specified area
                screenshot = ImageGrab.grab(bbox)
                # Convert to OpenCV format
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                # Use the YOLO model to predict objects in the frame
                results = self.model.predict(frame, imgsz=640, conf=0.9)
                for det in results[0].boxes:
                    class_id = int(det.cls[0])
                    class_name = self.class_names.get(class_id, "Unknown")
                    if class_name == "Exit_menu_button":
                        # Click the 'Exit_menu_button' to close the menu
                        x_center, y_center, w, h = map(int, det.xywh[0])
                        x1 = int(x_center - w / 2)
                        y1 = int(y_center - h / 2)
                        # Click the center of the button
                        pyautogui.click(x1 + w // 2, y1 + h // 2)
                        return True
                return False  # 'Exit_Menu_button' not found
            except Exception as e:
                print(f"Failed to check if menu is open: {e}")
                return False
        else:
            print("Cannot check if menu is open because window properties were not found.")
            return False
    def get_menu_data(self):
        """
        Get the data from the planet menu.
        """
        if self.is_open():
            if self.window_properties:
                # Capture a screenshot of the defined window
                top_left = self.window_properties["top_left"]
                bottom_right = self.window_properties["bottom_right"]
                bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                screenshot = ImageGrab.grab(bbox)
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                # Use YOLO model to detect icons in the screenshot
                results = self.model.predict(frame, imgsz=640, conf=0.9)

                # self.get_upgrade_prices(results, frame)
                # self.get_planet_levels(results, frame)
                # self.get_upgrade_multiplier(results, frame)
                # self.get_planet_resources(results, frame)
                return results, frame
    def get_planet_resources(self, results, frame):
        """
        Extract the resource values from the menu by OCR-ing the table under the detected headers.
        """
        header_boxes = []
        for det in results[0].boxes:
            class_id = int(det.cls[0])
            class_name = self.class_names.get(class_id, "Unknown")
            # print(f"Detected class: {class_name}")
            if class_name in ["Resource_text", "Yield_text", "Rate_text", "Mined_text"]:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                header_boxes.append({'class_name': class_name, 'bbox': (x1, y1, x2, y2)})

        if not header_boxes:
            print("No header boxes detected. Please check the class names and confidence threshold.")
            return

        # Step 2: Sort the header_boxes left to right based on x1
        header_boxes.sort(key=lambda x: x['bbox'][0])  # sort by x1

        # Step 3: Define the columns based on the left bound of each box and the right bound of the next box
        columns = []
        num_headers = len(header_boxes)
        for i in range(num_headers):
            current_box = header_boxes[i]['bbox']
            class_name = header_boxes[i]['class_name']
            x1 = current_box[0]-4  # Adjusted left bound for the column
            y1 = current_box[1]
            x2 = current_box[2]-6 # Adjusted right bound for the column
            y2 = current_box[3]
            # For all but the last column, the right bound is the left bound of the next header
            if i < num_headers - 1:
                next_x1 = header_boxes[i+1]['bbox'][0]
                column_x2 = next_x1
            else:
                column_x2 = x2  # For the last column, the right bound is x2 of the last header

            columns.append({'class_name': class_name, 'x1': x1, 'x2': column_x2, 'header_y1': y1, 'header_y2': y2})

        # Step 4: Define the rows based on the first row's actual positions
        # Calculate the height of the header to estimate the row height
        header_height = columns[0]['header_y2'] - columns[0]['header_y1']

        # Estimate row height (you might need to adjust this value based on your image)
        row_height = header_height + 6  # Adjusted height for rows

        # Starting y position for the first row (adjust the offset as needed)
        first_row_y1 = columns[0]['header_y2'] + 10  # Gap between header and first row
        first_row_y2 = first_row_y1 + row_height

        # Collect data
        data = {}
        for row_num in range(3):  # Up to 3 rows
            row_data = {}
            # Calculate y positions for each row
            current_row_y1 = first_row_y1 + row_num * (row_height + 6)  # Adjust gap between rows if needed
            current_row_y2 = current_row_y1 + row_height
            for col in columns:
                col_name = col['class_name']
                x1 = col['x1']
                x2 = col['x2']
                if col_name == "Mined_text":
                    x2 += 30  # Adjust the right bound for the last column
                # Crop the cell
                cell_crop = frame[current_row_y1:current_row_y2, x1:x2]
                
                # Plot the cropped cell image (optional for debugging)
                # plt.figure(figsize=(2, 2))
                # plt.imshow(cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB))
                # plt.title(f"Row {row_num+1}, Column: {col_name}")
                # plt.axis('off')
                # plt.show()

                # Perform OCR on the cell
                cell_text = self.perform_ocr(cell_crop, col_name)
                row_data[col_name] = cell_text
            data[f'row_{row_num+1}'] = row_data

        print("Extracted resource data:", data)
        print("\n")
        return data

    def perform_ocr(self, image_crop, column_name):
        """
        Perform OCR on a given image crop, adjusting settings based on the column name.
        """
        if image_crop.size == 0:
            return "OCR failed or empty region"

        # Step 1: Upscale the cropped image to enhance resolution
        scale_percent = 500  # Scale up by 500%
        width = int(image_crop.shape[1] * scale_percent / 100)
        height = int(image_crop.shape[0] * scale_percent / 100)
        dim = (width, height)
        upscaled = cv2.resize(image_crop, dim, interpolation=cv2.INTER_CUBIC)

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply thresholding to improve contrast
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert the thresholded image to a PIL image for OCR
        pil_img = Image.fromarray(thresh)

        # Adjust Tesseract configuration based on the column
        if column_name == "Resource_text":
            # Only letters and spaces
            ocr_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        elif column_name == "Yield_text":
            # Numbers and %
            ocr_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789%'
        elif column_name == "Rate_text":
            # Numbers, periods, and "/sec"
            ocr_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789./sec'
        elif column_name == "Mined_text":
            # Only numbers
            ocr_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        else:
            # Default configuration
            ocr_config = '--psm 7 --oem 3'

        # Perform OCR using Tesseract
        raw_text = pytesseract.image_to_string(pil_img, config=ocr_config).strip()

        # Clean the OCR output based on the column
        if column_name == "Resource_text":
            # Remove any non-letter characters
            cleaned_text = re.sub(r'[^A-Za-z ]', '', raw_text)
        elif column_name == "Yield_text":
            # Keep numbers and %
            cleaned_text = re.sub(r'[^0-9%]', '', raw_text)
        elif column_name == "Rate_text":
            # Keep numbers, periods, and "/sec"
            cleaned_text = re.sub(r'[^0-9./sec]', '', raw_text)
        elif column_name == "Mined_text":
            # Keep only numbers
            cleaned_text = re.sub(r'[^0-9]', '', raw_text)
        else:
            cleaned_text = raw_text.strip()

        return cleaned_text

    def get_upgrade_prices(self, results, frame):
        """
        Get the upgrade prices from the detected icons.
        """
        upgrade_prices = {}
        purchase_buttons = []
        text_detections = []
        for det in results[0].boxes:
            # Extract bounding box and class
            class_id = int(det.cls[0])
            class_name = self.class_names.get(class_id, "Unknown")
            # For xywh[0], x, y are center coordinates, w, h are width and height
            x_center, y_center, w, h = map(int, det.xywh[0])
            # Convert to x1, y1, x2, y2
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            bbox = (x1, y1, x2, y2)

            # Collect purchase buttons and text detections
            if class_name == "Purchase_button":
                purchase_buttons.append({
                    'class_name': class_name,
                    'bbox': bbox,
                    'y_center': y_center
                })
            elif class_name in ["Mining_rate_text", "Ship_speed_text", "Cargo_text"]:
                text_detections.append({
                    'class_name': class_name,
                    'bbox': bbox,
                    'y_center': y_center
                })
            else:
                # Other classes, ignore or handle if necessary
                pass

        # Sort both lists by the vertical (y) position to align corresponding pairs
        purchase_buttons.sort(key=lambda item: item['y_center'])
        text_detections.sort(key=lambda item: item['y_center'])

        # Match each purchase button with its corresponding text label
        for i in range(min(len(purchase_buttons), len(text_detections))):
            button = purchase_buttons[i]
            text = text_detections[i]

            # Use the purchase button bounding box as the crop region for OCR
            bx1, by1, bx2, by2 = button['bbox']

            # Ensure crop region is within image bounds
            bx1 = max(bx1, 0)
            by1 = max(by1, 0)
            bx2 = min(bx2, frame.shape[1])
            by2 = min(by2, frame.shape[0])

            # Crop the area within the purchase button
            price_crop = frame[by1:by2, bx1:bx2]

            # Check if the cropped region is valid (non-empty)
            if price_crop.size == 0:
                print(f"Skipping OCR for {text['class_name']}: Empty crop region.")
                extracted_price = "OCR failed or empty region"
            else:
                # Step 1: Upscale the cropped image to enhance resolution
                scale_percent = 500  # Scale up by 300%
                width = int(price_crop.shape[1] * scale_percent / 100)
                height = int(price_crop.shape[0] * scale_percent / 100)
                dim = (width, height)
                upscaled = cv2.resize(price_crop, dim, interpolation=cv2.INTER_CUBIC)

                # Step 2: Convert to grayscale
                gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

                # Step 3: Apply a sharpening filter
                kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
                sharpened = cv2.filter2D(gray, -1, kernel)

                # Convert the sharpened image to a PIL image for OCR
                price_pil = Image.fromarray(sharpened)

                # Perform OCR using Tesseract
                raw_text = pytesseract.image_to_string(
                    price_pil,
                    config='--psm 6 --oem 3 -c tessedit_char_whitelist="0123456789Ee+.-"'
                ).strip()

                # Strict cleaning of OCR output to retain only valid characters
                cleaned_text = re.sub(r'[^0-9Ee+.-]', '', raw_text).strip()  # Retain digits, '.', 'E', '+', and '-'
                # print(f"Raw OCR result for {text['class_name']}: '{raw_text}'")
                # Handle scientific notation if detected
                if re.match(r'^\d+(\.\d+)?[eE][+-]?\d+$', cleaned_text):
                    try:
                        # Convert scientific notation to an integer or float as a string
                        # print(f"Cleaned text: {cleaned_text}")
                        extracted_price = str(int(float(cleaned_text)))
                    except ValueError:
                        extracted_price = "OCR failed or invalid format"
                else:
                    # If no scientific notation, just take cleaned text
                    extracted_price = cleaned_text

                # Validate the cleaned text to ensure it's a valid number
                if not re.match(r'^\d+$', extracted_price):
                    extracted_price = "OCR failed or invalid format"

            # Store the extracted price along with the purchase button and text label
            upgrade_prices[text['class_name']] = {
                'purchase_button_bbox': button['bbox'],
                'text_bbox': text['bbox'],
                'price': extracted_price
            }

        print("Upgrade prices and their associated buttons:", upgrade_prices)
        print("\n")
        return upgrade_prices

    def get_upgrade_multiplier(self, results, frame):
        """
        Get the upgrade multiplier from the detected icons.
        """
        multiplier = None

        # Step 1: Find the bounding box for the "Multiplier_button"
        for det in results[0].boxes:
            # Extract bounding box and class
            class_id = int(det.cls[0])
            class_name = self.class_names.get(class_id, "Unknown")

            if class_name == "Multiplier_button":
                # Extract bounding box details
                x_center, y_center, w, h = map(int, det.xywh[0])
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Zoom in by trimming some pixels from the sides of the bounding box
                zoom_factor = 0.15  # Zoom factor (e.g., 20%)
                trim_x = int(w * zoom_factor)
                trim_y = int(h * zoom_factor)

                # Adjust coordinates to zoom in
                x1_zoomed = max(x1 + trim_x, 0)
                y1_zoomed = max(y1 + trim_y, 0)
                x2_zoomed = min(x2 - trim_x, frame.shape[1])
                y2_zoomed = min(y2 - trim_y, frame.shape[0])

                # Crop the area of the Multiplier button with the zoomed-in coordinates
                multiplier_crop = frame[y1_zoomed:y2_zoomed, x1_zoomed:x2_zoomed]

                # Plot the cropped (zoomed-in) region for visual verification
                # plt.figure(figsize=(5, 5))
                # plt.imshow(cv2.cvtColor(multiplier_crop, cv2.COLOR_BGR2RGB))
                # plt.title("Zoomed-In Cropped Multiplier Button Region")
                # plt.axis('off')
                # plt.show()

                # Perform OCR on the cropped region to get the multiplier text
                def ocr_extraction(crop):
                    if crop.size == 0:
                        return "OCR failed or empty region"

                    # Step 1: Upscale the cropped image to enhance resolution
                    scale_percent = 500  # Scale up by 500%
                    width = int(crop.shape[1] * scale_percent / 100)
                    height = int(crop.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    upscaled = cv2.resize(crop, dim, interpolation=cv2.INTER_CUBIC)

                    # Step 2: Convert to grayscale
                    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

                    # Step 3: Apply thresholding to improve contrast
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Convert the thresholded image to a PIL image for OCR
                    info_pil = Image.fromarray(thresh)

                    # Perform OCR using Tesseract
                    allowed_chars = "x0123456789"  # Include characters needed for multipliers
                    ocr_config = f'--psm 6 --oem 3 -c tessedit_char_whitelist="{allowed_chars}"'
                    raw_text = pytesseract.image_to_string(
                        info_pil,
                        config=ocr_config
                    ).strip()
                    # print(f"Raw OCR result for multiplier: '{raw_text}'")

                    return raw_text.strip()

                # Extract OCR text from the cropped region
                raw_multiplier_text = ocr_extraction(multiplier_crop).replace(" ", "").replace("x", "")
                # print(f"Extracted multiplier text: '{raw_multiplier_text}'")
                print("Upgrade multiplier:", raw_multiplier_text)	
                return int(raw_multiplier_text) if raw_multiplier_text.isdigit() else None

                # Since we only need one "Multiplier_button", break once we find it
                break

        # Return the multiplier (could be None if not found or OCR failed)
       
        return multiplier

    def get_planet_levels(self, results, frame):
        """
        Get the planet statistics from the detected icons.
        """
        planet_levels = {}
        purchase_buttons = []
        text_detections = []

        # Step 1: Collect all detected text and buttons
        for det in results[0].boxes:
            # Extract bounding box and class
            class_id = int(det.cls[0])
            class_name = self.class_names.get(class_id, "Unknown")
            # For xywh[0], x, y are center coordinates, w, h are width and height
            x_center, y_center, w, h = map(int, det.xywh[0])
            # Convert to x1, y1, x2, y2
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            bbox = (x1, y1, x2, y2)
            height = h

            # Collect purchase buttons and text detections
            if class_name == "Purchase_button":
                purchase_buttons.append({
                    'class_name': class_name,
                    'bbox': bbox,
                    'y_center': y_center
                })
            elif class_name in ["Mining_rate_text", "Ship_speed_text", "Cargo_text"]:
                text_detections.append({
                    'class_name': class_name,
                    'bbox': bbox,
                    'y_center': y_center,
                    'height': height
                })
            else:
                # Other classes, ignore or handle if necessary
                pass

        # Sort both lists by the vertical (y) position to align corresponding pairs
        purchase_buttons.sort(key=lambda item: item['y_center'])
        text_detections.sort(key=lambda item: item['y_center'])

        # Step 2: Match each text label with its corresponding purchase button
        for i in range(min(len(purchase_buttons), len(text_detections))):
            button = purchase_buttons[i]
            text = text_detections[i]

            # Calculate the bounding box of the information area
            text_bbox = text['bbox']
            button_bbox = button['bbox']
            text_height = text['height']

            # Crop the first line directly below the detected text
            line1_x1 = text_bbox[0]  # Left side of the detected text
            line1_x2 = button_bbox[0]  # Left side of the corresponding button
            line1_y1 = text_bbox[3] + 1  # Just below the bottom of the detected text
            line1_y2 = line1_y1 + text_height  # Height of the first line

            # Ensure crop region is within image bounds
            line1_x1 = max(line1_x1, 0)
            line1_y1 = max(line1_y1, 0)
            line1_x2 = min(line1_x2, frame.shape[1])
            line1_y2 = min(line1_y2, frame.shape[0])

            # Crop the first line of information
            line1_crop = frame[line1_y1:line1_y2, line1_x1:line1_x2]

            # Crop the second line directly below the first line
            line2_y1 = line1_y2   # Just below the first line
            line2_y2 = line2_y1 + text_height  # Height of the second line

            # Ensure crop region is within image bounds
            line2_x1 = line1_x1
            line2_x2 = line1_x2
            line2_y1 = max(line2_y1, 0)
            line2_y2 = min(line2_y2, frame.shape[0])

            # Crop the second line of information
            line2_crop = frame[line2_y1:line2_y2, line2_x1:line2_x2]

            # Define function to process cropped lines and extract text
            def process_crop(crop, class_name):
                if crop.size == 0:
                    return "OCR failed or empty region"

                # Step 1: Upscale the cropped image to enhance resolution
                scale_percent = 500  # Scale up by 300%
                width = int(crop.shape[1] * scale_percent / 100)
                height = int(crop.shape[0] * scale_percent / 100)
                dim = (width, height)
                upscaled = cv2.resize(crop, dim, interpolation=cv2.INTER_CUBIC)

                # Step 2: Convert to grayscale
                gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

                # Step 3: Apply thresholding to improve contrast
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Step 4: Apply morphological operations to clean the text
                kernel = np.ones((2, 2), np.uint8)
                thresh = cv2.erode(thresh, kernel, iterations=1)
                thresh = cv2.dilate(thresh, kernel, iterations=1)

                # Convert the thresholded image to a PIL image for OCR
                info_pil = Image.fromarray(thresh)

                # Perform OCR using Tesseract
                allowed_chars = "Lv.0123456789mkph/sec "  # Include a space at the end
                ocr_config = f'--psm 6 --oem 3 -c tessedit_char_whitelist="{allowed_chars}"'
                raw_text = pytesseract.image_to_string(
                    info_pil,
                    config=ocr_config
                ).strip()

                # Clean the OCR output
                cleaned_text = raw_text.strip()

                # Extract relevant information based on the class label
                return self.parse_planet_info(cleaned_text, label_class_name=class_name)

            # Process the two lines separately
            extracted_info_line1 = process_crop(line1_crop, text['class_name'])
            extracted_info_line2 = process_crop(line2_crop, text['class_name'])

            # Combine extracted information
            combined_info = {}
            if isinstance(extracted_info_line1, dict):
                combined_info.update(extracted_info_line1)
            if isinstance(extracted_info_line2, dict):
                combined_info.update(extracted_info_line2)

            # Store the extracted information along with the relevant bounding boxes
            planet_levels[text['class_name']] = {
                'line1_bbox': (line1_x1, line1_y1, line1_x2, line1_y2),
                'line2_bbox': (line2_x1, line2_y1, line2_x2, line2_y2),
                'text_bbox': text_bbox,
                'extracted_info': combined_info
            }

        print("Planet levels and their extracted information:", planet_levels)
        print("\n")
        
       
        return planet_levels

    def parse_planet_info(self, text, label_class_name=None):
        """
        Parse the extracted text to extract planet statistics.
        The label_class_name parameter is used to customize the parsing based on text type.
        """
        # Initialize a dictionary to hold parsed values
        info = {}

        # Handle Level Information (e.g., "Lv. 21")
        level_match = re.search(r'Lv\.?\s*(\d+)', text)
        if level_match:
            info['Level'] = int(level_match.group(1))

        # Handle Rate Information (e.g., "126.80 / sec")
        rate_match = re.search(r'([\d\.]+)\s*/\s*sec', text)
        if rate_match:
            info['Rate_per_sec'] = float(rate_match.group(1))

        # Handle Speed Information (e.g., "77.96 mkph")
        speed_match = re.search(r'([\d\.]+)\s*mkph', text)
        if speed_match:
            info['Speed_mkph'] = float(speed_match.group(1))

        # Handle Cargo Information
        # If the class label is specifically for Cargo_text, look for a standalone number
        if label_class_name == "Cargo_text":
            cargo_match = re.search(r'(\d+)', text)
            if cargo_match:
                info['Cargo'] = int(cargo_match.group(1))

        # If no known patterns matched, return the raw text
        if not info:
            info['Raw_Text'] = text

        return info

    def get_positions(self, class_id, model):
        """
        Get the positions (bounding boxes) of the detected objects based on the class ID.
        """
        if self.window_properties:
            # Capture a screenshot of the defined window
            top_left = self.window_properties["top_left"]
            bottom_right = self.window_properties["bottom_right"]
            bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

            try:
                # Capture the specified area
                screenshot = ImageGrab.grab(bbox)
                # Convert to OpenCV format
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                # Use the YOLO model to predict objects in the frame
                results = model.predict(frame, imgsz=640, conf=0.7)
                positions = []
                for det in results[0].boxes:
                    detected_class_id = int(det.cls[0])
                    if detected_class_id == class_id:
                        # Get bounding box coordinates in (x1, y1, x2, y2) format
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        positions.append((x1, y1, x2, y2))
                return positions
            except Exception as e:
                print(f"Failed to get positions: {e}")
                return []
        else:
            print("Cannot get positions because window properties were not found.")
            return []

    def can_upgrade(self, upgrade_name):

        
        """Check if we can upgrade a planet based on the price and our networth cash"""
        upgrade_prices = self.get_upgrade_prices()
        networth = self.get_networth()
        if networth:
            # Get the cash value from the net worth data
            cash = int(networth.get("Cash", 0))
            print("Cash:", cash)
            upgrade_price = upgrade_prices.get(upgrade_name, 0)
            
            # Check if we can afford the upgrade
            return cash >= upgrade_price
        return False
    
    def upgrade_planet(self, upgrade_type):
        """
        Attempts to upgrade the specified upgrade_type ('miningspeed', 'shipspeed', 'cargo').

        Steps:
        1. Run get_menu_data()
        2. With results, run get_upgrade_prices(results, frame)
        3. Get net worth using get_networth()
        4. If cash >= upgrade price for upgrade_type, click the purchase button
        """
        # Map the input upgrade_type to the class names used in get_upgrade_prices()
        upgrade_type_mapping = {
            'miningspeed': 'Mining_rate_text',
            'shipspeed': 'Ship_speed_text',
            'cargo': 'Cargo_text'
        }
        # Get the corresponding class name
        upgrade_class_name = upgrade_type_mapping.get(upgrade_type.lower())
        if not upgrade_class_name:
            print(f"Invalid upgrade type: {upgrade_type}")
            return False

        # Step 1: Run get_menu_data()
        menu_data = self.get_menu_data()
        if not menu_data:
            print("Failed to get menu data.")
            return False
        results, frame = menu_data

        # Step 2: With results, run get_upgrade_prices(results, frame)
        upgrade_prices = self.get_upgrade_prices(results, frame)
        if not upgrade_prices:
            print("Failed to get upgrade prices.")
            return False

        # Step 3: Get net worth using get_networth()
        networth = self.get_networth()
        if not networth:
            print("Failed to get net worth.")
            return False
        cash_str = networth.get("Cash", None)
        if not cash_str:
            print("Cash value not found in net worth data.")
            return False
        try:
            cash = float(cash_str)
        except ValueError:
            print(f"Invalid cash value: {cash_str}")
            return False

        # Step 4: Check if cash >= upgrade price
        upgrade_info = upgrade_prices.get(upgrade_class_name)
        if not upgrade_info:
            print(f"No upgrade information found for {upgrade_class_name}")
            return False

        upgrade_price_str = upgrade_info.get('price')
        if not upgrade_price_str:
            print(f"No upgrade price found for {upgrade_class_name}")
            return False

        # Clean the upgrade price string and convert to float
        upgrade_price_str = upgrade_price_str.replace(',', '').replace(' ', '')
        try:
            upgrade_price = float(upgrade_price_str)
        except ValueError:
            print(f"Invalid upgrade price for {upgrade_class_name}: {upgrade_price_str}")
            return False

        if cash >= upgrade_price:
            # Click on the purchase button
            purchase_button_bbox = upgrade_info.get('purchase_button_bbox')
            if purchase_button_bbox:
                x1, y1, x2, y2 = purchase_button_bbox
                # Click the center of the purchase button
                x_click = x1 + (x2 - x1) // 2 + self.window.left
                y_click = y1 + (y2 - y1) // 2 + self.window.top
                pyautogui.click(x_click, y_click)
                print(f"Upgraded {upgrade_type}.")
                return True
            else:
                print(f"No purchase button bounding box found for {upgrade_class_name}")
                return False
        else:
            print(f"Not enough cash to upgrade {upgrade_type}. Cash: {cash}, Upgrade price: {upgrade_price}")
            return False
        

class Methods():
    def __init__(self):
        self.planets = Planets()
        self.menu = PlanetMenu()
        self.menu.set_zoom()
    
    def upgrade_mining(self, planet_name):
        """
        Upgrade the mining speed of the specified planet.
        """
        # Step 1: Open the planet menu
        if not self.menu.is_open():
            self.planets.open_planet_menu(planet_name)
            self.menu.current_menu = planet_name
            time.sleep(1)
        else:
            print("Planet menu is already open.")
            if self.menu.current_menu != planet_name:
                self.menu.close_menu()
                self.planets.open_planet_menu(planet_name)
                self.menu.current_menu = planet_name
        upgraded = self.menu.upgrade_planet("miningspeed")
        print("Upgraded mining speed:", upgraded)
        return upgraded
    def upgrade_shipping_speed(self,planet_name):
        """
        Upgrade the shipping speed of the specified planet.
        """
        # Step 1: Open the planet menu
        if not self.menu.is_open():
            self.planets.open_planet_menu(planet_name)
            self.menu.current_menu = planet_name
            time.sleep(1)
        else:
            print("Planet menu is already open.")
            if self.menu.current_menu != planet_name:
                self.menu.close_menu()
                self.planets.open_planet_menu(planet_name)
                self.menu.current_menu = planet_name

        upgraded = self.menu.upgrade_planet("shipspeed")
        print("Upgraded shipping speed:", upgraded)
        return upgraded   
    def upgrade_cargo(self,planet_name):
        """
        Upgrade the cargo capacity of the specified planet.
        """
        # Step 1: Open the planet menu
        if not self.menu.is_open():
            self.planets.open_planet_menu(planet_name)
            self.menu.current_menu = planet_name
            time.sleep(1)
        else:
            print("Planet menu is already open.")
            if self.menu.current_menu != planet_name:
                self.menu.close_menu()
                self.planets.open_planet_menu(planet_name)
                self.menu.current_menu = planet_name

        upgraded = self.menu.upgrade_planet("cargo")
        print("Upgraded cargo capacity:", upgraded)
        return upgraded   
    def purchase_unlocked(self):
        """
        Purchase the next unlocked planet.
        """
        purchased = self.planets.purchase_next()
        return purchased
        
        
# Example usage
if __name__ == "__main__":

    # planets = Planets()
    # menu = PlanetMenu()
    # methods = Methods()
    # methods.upgrade_mining("Anadius")
    # methods.upgrade_shipping_speed("Anadius")
    # methods.upgrade_cargo("Anadius")
    # methods.upgrade_mining("Drasta")
    # methods.menu.close_menu()

    game = PlanetMiner()
    game.take_screenshot_and_predict("Resultspapa.png")


    # purchased = planets.purchase_next()
    # if purchased:
    #     menu.close_menu()
    # time.sleep(5)
    # menu.get_menu_data()