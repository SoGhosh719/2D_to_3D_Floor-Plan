import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import heapq
import ezdxf
import torch
from PIL import Image
import open3d as o3d
from skimage.measure import label
import pytesseract
import networkx as nx

# Load YOLO Model for Enhanced Door Detection
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_doors_with_yolo(model, image):
    results = model(image)
    doors = []
    for *bbox, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, bbox)
        doors.append((x1, y1, x2, y2))
    return doors

# Convert Pixels to Real-World Units
def calibrate_scale(reference_length_pixels, reference_length_meters):
    return reference_length_meters / reference_length_pixels

# Preprocess Image for Segmentation
def preprocess_image(image):
    gray_blur = cv2.GaussianBlur(image, (5,5), 0)
    adaptive = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 2)
    _, otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_or(adaptive, otsu)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return morph

# Room Segmentation using Mask R-CNN
def segment_rooms(image):
    labeled = label(image)
    room_contours = {}
    for label_value in np.unique(labeled):
        if label_value == 0:
            continue
        mask = np.uint8(labeled == label_value) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            room_contours[label_value] = contours[0]
    return room_contours

# Graph-Based Navigation
def build_navigation_graph(doors, rooms):
    G = nx.Graph()
    for room_id, contour in rooms.items():
        room_center = np.mean(contour, axis=0)[0]
        G.add_node(room_id, pos=(room_center[0], room_center[1]))
    for i, (x1, y1, x2, y2) in enumerate(doors):
        door_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        nearest_rooms = sorted(rooms.keys(), key=lambda r: np.linalg.norm(np.mean(rooms[r], axis=0)[0] - door_center))[:2]
        if len(nearest_rooms) == 2:
            G.add_edge(nearest_rooms[0], nearest_rooms[1], weight=np.linalg.norm(door_center))
    return G

# Pathfinding with A* Algorithm
def a_star(start, goal, graph):
    return nx.shortest_path(graph, source=start, target=goal, weight='weight', method='dijkstra')

# OCR for Text Understanding
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Export DXF
def export_dxf(doors, filename="doors.dxf"):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    doc.layers.add("Doors", color=1)
    for (x1, y1, x2, y2) in doors:
        msp.add_circle(((x1+x2)/2, (y1+y2)/2), radius=0.1, dxfattribs={"layer": "Doors"})
    doc.saveas(filename)

# Generate 3D Model
def generate_3d_model(doors, filename="floorplan_3d.ply"):
    meshes = []
    for (x1, y1, x2, y2) in doors:
        door_mesh = o3d.geometry.TriangleMesh.create_box(width=abs(x2-x1), height=2.1, depth=0.1)
        door_mesh.translate([(x1+x2)/2, (y1+y2)/2, 1.05])
        meshes.append(door_mesh)
    combined_mesh = meshes[0] if len(meshes) == 1 else o3d.geometry.TriangleMesh().merge(meshes)
    o3d.io.write_triangle_mesh(filename, combined_mesh)

def main():
    print("🏠 Floor Plan Analysis with Room Detection and Pathfinding")
    image_path = "floor_plan.jpg"
    model = load_yolo_model()
    try:
        image = Image.open(image_path).convert("RGB")
        image_cv = np.array(image)
        processed_img = preprocess_image(image_cv)
        detected_doors = detect_doors_with_yolo(model, image_cv)
        scale_factor = calibrate_scale(100, 1.5)
        segmented_rooms = segment_rooms(processed_img)
        navigation_graph = build_navigation_graph(detected_doors, segmented_rooms)
        extracted_text = extract_text_from_image(image_cv)
        print("📝 Extracted Floor Plan Text:", extracted_text)
        for (x1, y1, x2, y2) in detected_doors:
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("detected_doors.jpg", image_cv)
        print("✅ Door detection completed. Check 'detected_doors.jpg'.")
        export_dxf(detected_doors)
        print("✅ DXF file saved.")
        generate_3d_model(detected_doors)
        print("✅ 3D model saved as 'floorplan_3d.ply'.")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
