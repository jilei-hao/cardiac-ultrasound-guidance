import numpy as np
import cv2
import struct

def load_stl_vertices(stl_path):
    """Load vertices from STL file."""
    vertices = []
    
    with open(stl_path, 'rb') as f:
        # Skip header (80 bytes)
        f.seek(80)
        
        # Read number of triangles
        num_triangles = struct.unpack('<I', f.read(4))[0]
        
        # Read triangles
        for _ in range(num_triangles):
            # Skip normal vector (12 bytes)
            f.read(12)
            
            # Read 3 vertices (36 bytes total)
            for _ in range(3):
                x, y, z = struct.unpack('<fff', f.read(12))
                vertices.append([x, y, z])
            
            # Skip attribute byte count (2 bytes)
            f.read(2)
    
    return np.array(vertices)


def project_3d_to_2d(vertices, position, size, rotation_angle=0.0):
    """Project 3D vertices to 2D screen coordinates. rotation_angle in radians."""
    if vertices is None or len(vertices) == 0:
        return np.empty((0, 2), dtype=np.int32)

    v = np.asarray(vertices, dtype=np.float32)
    valid_mask = np.isfinite(v).all(axis=1)
    if not np.any(valid_mask):
        return np.empty((0, 2), dtype=np.int32)
    v = v[valid_mask]

    # normalize to unit cube
    min_vals = v.min(axis=0)
    max_vals = v.max(axis=0)
    range_vals = max_vals - min_vals

    center = (min_vals + max_vals) / 2.0
    v = v - center
    max_range = float(np.max(range_vals)) if np.all(range_vals > 0) else 1.0
    v = v / (max_range if max_range > 0 else 1.0)

    # rotate around Z
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    Rz = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]], dtype=np.float32)
    v = v @ Rz.T

    # scale & orthographic project
    v = v * float(size)
    pts = v[:, :2] + np.asarray(position, dtype=np.float32)
    pts = np.clip(pts, 0, 10000)
    return pts.astype(np.int32)


def draw_stl_simplified(image, stl_path, position=(150, 150), size=80, rotation_angle=0,
                        color=(255, 255, 255), thickness=2):
    """Draw STL wireframe."""
    vertices = load_stl_vertices(stl_path)
    projected = project_3d_to_2d(vertices, position, size, rotation_angle)
    
    # Draw triangle edges
    for i in range(0, len(projected), 3):
        if i + 2 < len(projected):
            p0, p1, p2 = projected[i], projected[i+1], projected[i+2]
            cv2.line(image, tuple(p0), tuple(p1), color, thickness)
            cv2.line(image, tuple(p1), tuple(p2), color, thickness)
            cv2.line(image, tuple(p2), tuple(p0), color, thickness)
    
    return image
