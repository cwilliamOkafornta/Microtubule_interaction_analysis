import numpy as np
import pandas as pd
import os
import struct
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Try to import CuPy for GPU acceleration with a functional check
try:
    import cupy as cp
    # Perform a minimal functional check to ensure CUDA libraries (like nvrtc) are present
    _ = cp.cuda.runtime.getDeviceCount()
    HAS_GPU = True
except (ImportError, Exception):
    # This catches missing packages AND missing DLLs/drivers
    HAS_GPU = False
    cp = None

class MicrotubuleSpatialGraph:
    def __init__(self):
        self.nodes = {}
        self.segments = []
        self.class_mapping = {
            0: "Unclassified",
            1: "SMTs",
            2: "KMTs_pole1_Inside",
            3: "KMTs_pole1_Outside",
            4: "KMTs_pole2_Inside",
            5: "KMTs_pole2_Outside",
            6: "Mid-MTs",
            7: "Interdigitating-MTs",
            8: "Bridging-MTs"
        }
        self.class_reverse_mapping = {v: k for k, v in self.class_mapping.items()}

    def load_from_am(self, file_path):
        """Strictly parses the AmiraMesh (.am) binary spatial graph."""
        with open(file_path, 'rb') as f:
            content = f.read()
            # Robustly find the header end
            header_end = content.find(b'@1')
            header = content[:header_end].decode('ascii', errors='ignore')
            
            n_vertices = int(header.split("define VERTEX")[1].split()[0])
            n_edges = int(header.split("define EDGE")[1].split()[0])
            n_points = int(header.split("define POINT")[1].split()[0])
            
            def get_data_by_marker(marker_str, count, fmt, size):
                marker = f"\n{marker_str}\n".encode('ascii')
                idx = content.find(marker)
                if idx == -1: return None
                start = idx + len(marker)
                data = struct.unpack('<' + fmt * count, content[start:start + size * count])
                return np.array(data)

            v_coords = get_data_by_marker("@1", n_vertices * 3, "f", 4).reshape(-1, 3)
            edge_conn = get_data_by_marker("@10", n_edges * 2, "i", 4).reshape(-1, 2)
            edge_n_pts = get_data_by_marker("@11", n_edges, "i", 4)
            
            edge_classes = np.zeros(n_edges, dtype=int)
            for i in range(12, 20):
                data = get_data_by_marker(f"@{i}", n_edges, "i", 4)
                if data is not None and np.any(data):
                    edge_classes[data > 0] = i - 11

            p_coords = get_data_by_marker("@20", n_points * 3, "f", 4).reshape(-1, 3)
            
            self.segments = []
            pt_idx = 0
            for i in range(n_edges):
                n_pts = edge_n_pts[i]
                seg_pts = p_coords[pt_idx : pt_idx + n_pts].copy()
                pt_idx += n_pts
                if len(seg_pts) < 2: continue
                self.segments.append({
                    'segment_id': i,
                    'mt_class': int(edge_classes[i]),
                    'node1_id': int(edge_conn[i, 0]),
                    'node2_id': int(edge_conn[i, 1]),
                    'node1_pos': v_coords[edge_conn[i, 0]],
                    'node2_pos': v_coords[edge_conn[i, 1]],
                    'points': seg_pts
                })
            self.nodes = {i: {'pos': v_coords[i]} for i in range(n_vertices)}
            return len(self.segments)

    def get_combined_traces(self, selected_classes=None, subsample=1):
        traces = []
        if selected_classes is None:
            selected_classes = list(self.class_mapping.keys())
        for cls_id in selected_classes:
            cls_segs = [s for s in self.segments if s['mt_class'] == cls_id]
            if not cls_segs: continue
            all_x, all_y, all_z = [], [], []
            for seg in cls_segs:
                pts = seg['points'][::subsample]
                all_x.extend(pts[:, 0]); all_x.append(None)
                all_y.extend(pts[:, 1]); all_y.append(None)
                all_z.extend(pts[:, 2]); all_z.append(None)
            traces.append({
                'name': self.class_mapping[cls_id],
                'x': all_x, 'y': all_y, 'z': all_z,
                'id': cls_id
            })
        return traces

    def load_from_excel(self, file_path):
        xls = pd.ExcelFile(file_path)
        nodes_df = xls.parse("Nodes")
        points_df = xls.parse("Points")
        segments_df = xls.parse("Segments")
        for df in [nodes_df, points_df, segments_df]:
            df.columns = [str(c).strip() for c in df.columns]
        pts_lookup = {int(row['Point ID']): np.array([row['X Coord'], row['Y Coord'], row['Z Coord']], dtype=np.float32)
                      for _, row in points_df.iterrows()}
        self.nodes = {int(row['Node ID']): {'pos': np.array([row['X Coord'], row['Y Coord'], row['Z Coord']], dtype=np.float32),
                                            'class': self._extract_class(row)} for _, row in nodes_df.iterrows()}
        self.segments = []
        for _, row in segments_df.iterrows():
            pt_ids = [int(x) for x in str(row['Point IDs']).replace(';', ',').split(',') if x.strip().isdigit()]
            pts_coords = np.array([pts_lookup[pid] for pid in pt_ids if pid in pts_lookup], dtype=np.float32)
            if len(pts_coords) < 2: continue
            self.segments.append({'segment_id': int(row['Segment ID']), 'mt_class': self._extract_class(row),
                                  'node1_id': int(row['Node ID #1']), 'node2_id': int(row['Node ID #2']), 'points': pts_coords})

    def _extract_class(self, row):
        for class_name, class_id in self.class_reverse_mapping.items():
            if class_name in row and row[class_name] == 1:
                return class_id
        return 1

    def save_as_am(self, file_path, segments=None, surface_verts=None):
        """
        Exports the spatial graph back to an AmiraMesh (.am) binary file.
        Includes surface vertices as isolated points if provided.
        """
        if segments is None:
            segments = self.segments

        # Collect unique vertices from segments
        vertices = []
        vertex_map = {} # (x,y,z) -> index
        
        edges = []
        points = []
        
        for seg in segments:
            # Add vertices
            v1_pos = tuple(seg.get('node1_pos', seg['points'][0]))
            v2_pos = tuple(seg.get('node2_pos', seg['points'][-1]))
            
            if v1_pos not in vertex_map:
                vertex_map[v1_pos] = len(vertices)
                vertices.append(v1_pos)
            if v2_pos not in vertex_map:
                vertex_map[v2_pos] = len(vertices)
                vertices.append(v2_pos)
            
            edges.append({
                'v1': vertex_map[v1_pos],
                'v2': vertex_map[v2_pos],
                'n_pts': len(seg['points']),
                'class': seg['mt_class']
            })
            points.extend(seg['points'])

        # Add surface vertices as isolated points if provided
        if surface_verts is not None:
            # surface_verts is expected to be a list of arrays (e.g., [surf1, surf2])
            for surf in surface_verts:
                for v in surf:
                    v_pos = tuple(v)
                    if v_pos not in vertex_map:
                        vertex_map[v_pos] = len(vertices)
                        vertices.append(v_pos)

        n_vertices = len(vertices)
        n_edges = len(edges)
        n_points = len(points)

        header = f"""# AmiraMesh 3D BINARY 2.0
define VERTEX {n_vertices}
define EDGE {n_edges}
define POINT {n_points}

Parameters {{
    ContentType "SpatialGraph",
    Description "Microtubule Interaction Exports"
}}

VERTEX {{ float[3] Coordinates }} @1
EDGE {{ int[2] EdgeConnectivity }} @10
EDGE {{ int NumPoints }} @11
"""
        # Add class markers to header
        for i in range(1, 9):
             header += f"EDGE {{ int Class_{i} }} @{i+11}\n"
        
        header += f"POINT {{ float[3] Coordinates }} @20\n"
        header += "\n@1\n"

        with open(file_path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(np.array(vertices, dtype=np.float32).tobytes())
            
            f.write(b"\n@10\n")
            edge_conn = []
            for e in edges: edge_conn.extend([e['v1'], e['v2']])
            f.write(np.array(edge_conn, dtype=np.int32).tobytes())
            
            f.write(b"\n@11\n")
            edge_n_pts = [e['n_pts'] for e in edges]
            f.write(np.array(edge_n_pts, dtype=np.int32).tobytes())
            
            # Write class masks
            for cls_id in range(1, 9):
                f.write(f"\n@{cls_id+11}\n".encode('ascii'))
                mask = [(1 if e['class'] == cls_id else 0) for e in edges]
                f.write(np.array(mask, dtype=np.int32).tobytes())
                
            f.write(b"\n@20\n")
            f.write(np.array(points, dtype=np.float32).tobytes())

    def save_points_as_am(self, file_path, points, values=None, label="Proximity"):
        """
        Exports a point cloud (e.g., heatmap) to an AmiraMesh (.am) VertexSet.
        """
        n_points = len(points)
        header = f"""# AmiraMesh 3D BINARY 2.0
define VERTEX {n_points}

Parameters {{
    ContentType "VertexSet",
    Description "Interaction Heatmap Points"
}}

VERTEX {{ float[3] Coordinates }} @1
"""
        if values is not None:
            header += f"VERTEX {{ float {label} }} @2\n"
        
        header += "\n@1\n"
        
        with open(file_path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(np.array(points, dtype=np.float32).tobytes())
            
            if values is not None:
                f.write(b"\n@2\n")
                f.write(np.array(values, dtype=np.float32).tobytes())

def approximate_direction(points):
    """Approximate the direction vector of a MT segment."""
    if len(points) < 2:
        return np.array([0, 0, 0], dtype=np.float32)
    vec = points[-1] - points[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.array([0, 0, 0], dtype=np.float32)

def load_surfaces(surf_path):
    """
    Read an Amira ASCII .surf file and extract vertices for two surfaces.
    Returns: (surf1_verts, surf2_verts, c1, c2)
    """
    vertices = []
    with open(surf_path, 'r', encoding='utf-8', errors='ignore') as f:
        in_vertices = False
        lines_to_read = 0
        for line in f:
            line = line.strip()
            if line.startswith("Vertices"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    lines_to_read = int(parts[1])
                    in_vertices = True
                    continue
            if in_vertices and lines_to_read > 0:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        lines_to_read -= 1
                    except ValueError: pass
                if lines_to_read == 0: break
    vertices = np.array(vertices, dtype=np.float32)
    if len(vertices) == 0: raise ValueError("No vertices found in surface file.")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(vertices)
    s1, s2 = vertices[kmeans.labels_ == 0], vertices[kmeans.labels_ == 1]
    return s1, s2, np.mean(s1, axis=0), np.mean(s2, axis=0)

def compute_advanced_interactions(segments, dist_threshold, ref_class_filter, neighbor_class_filter, c1, c2, use_gpu=False):
    """
    Computes pairwise proximity with dual-class filtering.
    """
    results = []
    n = len(segments)
    
    # Define Spindle Axis
    spindle_vec = c2 - c1
    spindle_len = np.linalg.norm(spindle_vec)
    spindle_unit = spindle_vec / spindle_len if spindle_len > 0 else np.array([0,0,1])
    midpoint = (c1 + c2) / 2.0

    # Indices for reference microtubules
    ref_indices = [i for i, s in enumerate(segments) if s['mt_class'] in ref_class_filter]
    # Indices for candidate neighbor microtubules
    neighbor_indices = [i for i, s in enumerate(segments) if s['mt_class'] in neighbor_class_filter]
    
    for i in ref_indices:
        seg1 = segments[i]
        pts1 = seg1['points']
        
        for j in neighbor_indices:
            if i == j: continue
            seg2 = segments[j]
            pts2 = seg2['points']
            
            # 1. Bounding box pre-filter
            if np.any(pts1.min(axis=0) > pts2.max(axis=0) + dist_threshold) or \
               np.any(pts2.min(axis=0) > pts1.max(axis=0) + dist_threshold):
                continue
            
            # 2. Orientation Check
            dir1 = approximate_direction(pts1)
            dir2 = approximate_direction(pts2)
            angle = np.degrees(np.arccos(np.clip(np.dot(dir1, dir2), -1.0, 1.0)))
            
            orientation = None
            if 0 <= angle <= 30:
                orientation = "Parallel"
            elif 31 <= angle <= 60:
                orientation = "Anti-parallel"
            
            if not orientation: continue

            # 3. Distance and Interaction Length
            if use_gpu and HAS_GPU:
                pts1_gpu = cp.asarray(pts1)
                pts2_gpu = cp.asarray(pts2)
                # Compute distance matrix on GPU
                dists = cp.linalg.norm(pts1_gpu[:, None, :] - pts2_gpu[None, :, :], axis=-1)
                dists_to_seg2_gpu = dists.min(axis=1)
                dists_to_seg2 = cp.asnumpy(dists_to_seg2_gpu)
            else:
                dists_to_seg2 = cdist(pts1, pts2).min(axis=1)
                
            interacting_mask = dists_to_seg2 <= dist_threshold
            
            if not np.any(interacting_mask): continue
            
            mean_dist = np.mean(dists_to_seg2[interacting_mask])
            
            int_length = 0
            for k in range(len(pts1) - 1):
                if interacting_mask[k] and interacting_mask[k+1]:
                    int_length += np.linalg.norm(pts1[k+1] - pts1[k])
            
            if int_length == 0: int_length = 0.1 

            # 4. Location
            int_pts = pts1[interacting_mask]
            mean_int_pt = np.mean(int_pts, axis=0)
            vec_from_mid = mean_int_pt - midpoint
            projection = np.dot(vec_from_mid, spindle_unit)
            
            results.append({
                'Ref_Seg_ID': seg1['segment_id'],
                'Neighbor_Seg_ID': seg2['segment_id'],
                'Class_Ref': seg1['mt_class'],
                'Class_Neighbor': seg2['mt_class'],
                'Orientation': orientation,
                'Angle_deg': round(angle, 2),
                'Mean_Dist_A': round(mean_dist, 2),
                'Int_Length_A': round(int_length, 2),
                'Spindle_Pos_A': round(projection, 2)
            })
            
    return pd.DataFrame(results)
