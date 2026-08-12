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



def draw_stl_simplified(image, stl_path, position=(150, 150), size=80, rotation_matrix=None,
                        color=(255, 255, 255), thickness=2):
    """Draw STL wireframe with full 3D rotation."""
    vertices = load_stl_vertices(stl_path)
    projected = project_3d_to_2d(vertices, position, size, rotation_matrix)

    # Draw triangle edges
    for i in range(0, len(projected), 3):
        if i + 2 < len(projected):
            p0, p1, p2 = projected[i], projected[i+1], projected[i+2]
            cv2.line(image, tuple(p0), tuple(p1), color, thickness)
            cv2.line(image, tuple(p1), tuple(p2), color, thickness)
            cv2.line(image, tuple(p2), tuple(p0), color, thickness)

    return image
