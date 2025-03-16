import cv2
import numpy as np
import open3d as o3d
import argparse

# --- 🛠 Command Line Argument Parsing ---
parser = argparse.ArgumentParser(description="Convert 2D floor plan to 3D model")
parser.add_argument("--input", type=str, default="floor_plan.jpg", help="Input floor plan image")
parser.add_argument("--output", type=str, default="floor_plan_3D", help="Output filename without extension")
args = parser.parse_args()

# --- 🖼 Load and Process the Floor Plan Image ---
img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Error: Cannot find the image {args.input}")

# --- 🔍 Edge Detection (Detect Walls) ---
edges = cv2.Canny(img, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 📌 Create a 3D Model (Point Cloud) ---
points = []
for contour in contours:
    for pt in contour:
        x, y = pt[0]
        points.append([x, y, 0])   # Floor level
        points.append([x, y, 100]) # Ceiling level (100 units height)

# Convert to Open3D PointCloud format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))

# --- 💾 Save as PLY ---
ply_filename = f"{args.output}.ply"
o3d.io.write_point_cloud(ply_filename, pcd)
print(f"✅ 3D Point Cloud saved as {ply_filename}")

# --- 🏗 Convert to 3D Mesh (Delaunay Triangulation) ---
print("🔄 Converting point cloud to mesh...")
pcd.estimate_normals()
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=15.0)
mesh.compute_vertex_normals()

# --- 💾 Save as STL ---
stl_filename = f"{args.output}.stl"
o3d.io.write_triangle_mesh(stl_filename, mesh)
print(f"✅ 3D Mesh saved as {stl_filename}")

# --- 🎥 Visualization ---
print("🔍 Displaying 3D Model...")
o3d.visualization.draw_geometries([mesh], window_name="3D Floor Plan")
