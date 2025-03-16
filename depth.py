import cv2
from ultralytics import YOLO
import time
import pyttsx3

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()

# Load YOLOv8 model for object detection
yolo_model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model path

# Function to detect persons and calculate bounding box area
def detect_person_and_calculate_area(frame):
    # Perform object detection on the frame
    results = yolo_model(frame)

    # Filter out only the 'person' class (class ID 0)
    persons = results[0].boxes[results[0].boxes.cls == 0]  # Class 0 is 'person' in YOLOv8

    # Get bounding boxes for persons
    boxes = persons.xyxy.cpu().numpy()  # Bounding boxes for persons

    # Calculate the area of each bounding box
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        width = x2 - x1
        height = y2 - y1
        area = width * height  # Area of the bounding box
        areas.append(area)

    return areas

# Open the laptop's built-in camera
cap = cv2.VideoCapture(0)

# Variables to store the last update time and areas
last_update_time = time.time()
last_area = None

# File to log position changes
log_file = "position_changes.txt"
with open(log_file, "w") as f:  # Clear the file at the start
    f.write("Position Changes Log:\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect persons and calculate bounding box areas
    areas = detect_person_and_calculate_area(frame)

    # Get the largest bounding box area (if any)
    current_area = max(areas) if areas else None

    # Display the live feed
    cv2.imshow('Live Feed', frame)

    # Check if there is a significant change in the bounding box area
    if current_area is not None and last_area is not None:
        change_threshold = 0.1  # 10% change threshold
        change = abs(current_area - last_area) / last_area

        if change > change_threshold:  # Significant change detected
            # Determine if the person is closer or farther
            if current_area > last_area:
                position_message = "Person is closer"
                log_message = f"{time.ctime()}: Person is closer."
            else:
                position_message = "Person is farther"
                log_message = f"{time.ctime()}: Person is farther."

            # Log the change to the file (with date and time)
            with open(log_file, "a") as f:
                f.write(f"{log_message}\n")

            # Convert the position message to speech
            tts_engine.say(position_message)
            tts_engine.runAndWait()

            # Update the last area
            last_area = current_area

    # Update the last area (for the first frame or if no change is detected)
    if current_area is not None:
        last_area = current_area

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()