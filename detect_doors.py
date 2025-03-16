import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 🖼 Load the Floor Plan ---
image_path = "floor_plan.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Error: Cannot find the image {image_path}")

# --- 🔍 Preprocess Image (Enhance Contrast & Remove Noise) ---
equalized = cv2.equalizeHist(img)  # Improve contrast
_, binary = cv2.threshold(equalized, 200, 255, cv2.THRESH_BINARY_INV)  # Convert to binary

# --- 📌 Find Contours (Potential Doors) ---
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Convert image to color for visualization
door_candidates = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# --- 🚪 Detect Doors Based on Shape & Size ---
potential_doors = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h) if h > 0 else 0  # Calculate aspect ratio

    # Criteria: Thin & rectangular (aspect ratio, size)
    if 0.2 < aspect_ratio < 0.6 and 50 < h < 150:
        potential_doors.append((x, y, w, h))
        cv2.rectangle(door_candidates, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

# --- 💾 Save & Display Results ---
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(door_candidates, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Doors: {len(potential_doors)} Found")
plt.axis("off")
plt.show()

# Save the detected door coordinates to a JSON file
import json
with open("doors.json", "w") as f:
    json.dump(potential_doors, f)

print(f"✅ Door locations saved to doors.json")
