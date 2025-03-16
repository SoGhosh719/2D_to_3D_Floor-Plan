import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import ezdxf
import torch
import time
import pyttsx3
from PIL import Image
import open3d as o3d
from skimage.measure import label
import pytesseract
import networkx as nx
from ultralytics import YOLO

# Initialize Text-to-Speech
tts_engine = pyttsx3.init()

# Load YOLO Model for Door Detection and Person Detection
def load_yolo_model():
    return YOLO("yolov8n.pt")  # Ensure you have this model

# Detect doors using YOLO
def detect_doors_with_yolo(model, image):
    results = model(image)
    doors = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            doors.append((x1, y1, x2, y2))
    return doors

# Preprocess Image for Room Segmentation
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return binary

# Room Segmentation
def segment_rooms(image):
    labeled = label(image > 0)
    room_contours = {label_value: cv2.findContours(np.uint8(labeled == label_value) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                     for label_value in np.unique(labeled) if label_value != 0}
    return room_contours

# Build Graph for Navigation
def build_navigation_graph(doors, rooms):
    G = nx.Graph()
    for room_id, contour in rooms.items():
        room_center = np.mean(contour, axis=0)[0]
        G.add_node(room_id, pos=(room_center[0], room_center[1]))
    for (x1, y1, x2, y2) in doors:
        door_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        nearest_rooms = sorted(rooms.keys(), key=lambda r: np.linalg.norm(np.mean(rooms[r], axis=0)[0] - door_center))[:2]
        if len(nearest_rooms) == 2:
            G.add_edge(nearest_rooms[0], nearest_rooms[1], weight=np.linalg.norm(door_center))
    return G

# OCR for Floor Plan Text Extraction
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(thresh, config='--psm 6')

# 3D Model Generation
def generate_3d_model(doors, filename="floorplan_3d.ply"):
    meshes = [o3d.geometry.TriangleMesh.create_box(width=abs(x2-x1), height=2.1, depth=0.1).translate([(x1+x2)/2, (y1+y2)/2, 1.05]) for (x1, y1, x2, y2) in doors]
    combined_mesh = sum(meshes, start=o3d.geometry.TriangleMesh())
    o3d.io.write_triangle_mesh(filename, combined_mesh)

# Export DXF
def export_dxf(doors, filename="doors.dxf"):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    doc.layers.add("Doors", color=1)
    for (x1, y1, x2, y2) in doors:
        msp.add_circle(((x1+x2)/2, (y1+y2)/2), radius=0.1, dxfattribs={"layer": "Doors"})
    doc.saveas(filename)

# Detect persons and track movement using webcam
def detect_person_movement():
    model = load_yolo_model()
    cap = cv2.VideoCapture(0)

    last_area = None
    log_file = "position_changes.txt"
    with open(log_file, "w") as f:
        f.write("Position Changes Log:\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        areas = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                width = x2 - x1
                height = y2 - y1
                areas.append(width * height)

        current_area = max(areas) if areas else None

        cv2.imshow('Live Feed', frame)

        if current_area is not None and last_area is not None:
            change_threshold = 0.1
            change = abs(current_area - last_area) / last_area

            if change > change_threshold:
                position_message = "Person is closer" if current_area > last_area else "Person is farther"
                with open(log_file, "a") as f:
                    f.write(f"{time.ctime()}: {position_message}\n")

                tts_engine.say(position_message)
                tts_engine.runAndWait()

        if current_area is not None:
            last_area = current_area

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Function
def main():
    print("🏠 Floor Plan Analysis with Door Detection and Movement Tracking")
    
    image_path = "floor_plan.jpg"
    model = load_yolo_model()

    try:
        image = np.array(Image.open(image_path))
        processed_img = preprocess_image(image)

        print("🔍 Detecting doors...")
        detected_doors = detect_doors_with_yolo(model, image)
        cv2.imwrite("detected_doors.jpg", image)

        print("📏 Running room segmentation...")
        segmented_rooms = segment_rooms(processed_img)

        print("📡 Constructing navigation graph...")
        navigation_graph = build_navigation_graph(detected_doors, segmented_rooms)

        print("📝 Extracting floor plan text...")
        extracted_text = extract_text_from_image(image)
        print("Extracted Text:", extracted_text)

        print("📄 Exporting DXF...")
        export_dxf(detected_doors)

        print("📐 Generating 3D model...")
        generate_3d_model(detected_doors)

        print("🚶 Tracking person movement...")
        detect_person_movement()

        print("✅ Process completed.")

    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
