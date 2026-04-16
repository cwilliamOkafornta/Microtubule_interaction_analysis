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
        """Robustly parses AmiraMesh (.am) both BINARY and ASCII versions."""
        with open(file_path, 'rb') as f:
            content = f.read()
            # Find the magic marker that starts the first data section (@1)
            # We look for \n@1 followed by newline to avoid false positives in binary data
            first_marker_idx = content.find(b'\n@1\n')
            if first_marker_idx == -1:
                first_marker_idx = content.find(b'\n@1\r\n')
            
            if first_marker_idx != -1:
                first_marker_idx += 1 # Skip the leading \n
            else:
                # Fallback: find first @1 anywhere (less robust)
                first_marker_idx = content.find(b'@1')
            
            if first_marker_idx == -1:
                return 0
                
            header = content[:first_marker_idx].decode('ascii', errors='ignore')
            # Check the first line for ASCII or BINARY
            first_line = header.split('\n')[0]
            is_ascii = "ASCII" in first_line
            
            def get_define(name):
                try: return int(header.split(f"define {name}")[1].split()[0])
                except: return 0

            n_vertices = get_define("VERTEX")
            n_edges = get_define("EDGE")
            n_points = get_define("POINT")

            def get_data_by_marker(marker_str, count, fmt, size):
                # Try various marker formats: \n@1\n, \n@1\r\n, or @1 at start
                for m_fmt in [f"\n{marker_str}\n", f"\n{marker_str}\r\n", f"{marker_str}\n", f"{marker_str}\r\n"]:
                    marker = m_fmt.encode('ascii')
                    idx = content.find(marker)
                    if idx != -1:
                        start = idx + len(marker)
                        if is_ascii:
                            end = content.find(b'\n@', start)
                            if end == -1: end = len(content)
                            data_str = content[start:end].decode('ascii', errors='ignore')
                            return data_str.split()
                        else:
                            try:
                                data = struct.unpack('<' + fmt * count, content[start:start + size * count])
                                return np.array(data)
                            except: return None
                return None

            if not is_ascii:
                v_coords = get_data_by_marker("@1", n_vertices * 3, "f", 4).reshape(-1, 3)
                edge_conn = get_data_by_marker("@10", n_edges * 2, "i", 4).reshape(-1, 2)
                edge_n_pts = get_data_by_marker("@11", n_edges, "i", 4)
                
                edge_classes = np.zeros(n_edges, dtype=int)
                for i in range(12, 20):
                    data = get_data_by_marker(f"@{i}", n_edges, "i", 4)
                    if data is not None and np.any(data):
                        edge_classes[data > 0] = i - 11

                p_coords = get_data_by_marker("@20", n_points * 3, "f", 4).reshape(-1, 3)
            else:
                v_data = get_data_by_marker("@1", 0, "", 0)
                v_coords = np.array([float(x) for x in v_data]).reshape(-1, 3) if v_data else np.zeros((0,3))
                
                edge_conn_data = get_data_by_marker("@2", 0, "", 0)
                edge_conn = np.array([int(x) for x in edge_conn_data]).reshape(-1, 2) if edge_conn_data else np.zeros((0,2), dtype=int)
                
                edge_n_pts_data = get_data_by_marker("@3", 0, "", 0)
                edge_n_pts = np.array([int(x) for x in edge_n_pts_data]) if edge_n_pts_data else np.zeros(0, dtype=int)
                
                edge_classes = np.zeros(n_edges, dtype=int)
                for i in range(4, 12):
                    data = get_data_by_marker(f"@{i}", 0, "", 0)
                    if data:
                        cls_vals = np.array([int(x) for x in data])
                        if np.any(cls_vals):
                            edge_classes[cls_vals > 0] = i - 3

                p_data = get_data_by_marker("@12", 0, "", 0)
                p_coords = np.array([float(x) for x in p_data]).reshape(-1, 3) if p_data else np.zeros((0,3))
            
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
                    'node1_pos': v_coords[edge_conn[i, 0]] if edge_conn[i, 0] < len(v_coords) else np.zeros(3),
                    'node2_pos': v_coords[edge_conn[i, 1]] if edge_conn[i, 1] < len(v_coords) else np.zeros(3),
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
        Exports the spatial graph back to an AmiraMesh (.am) ASCII 3.0 file.
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

        header = f"""# AmiraMesh 3D ASCII 3.0

define VERTEX {n_vertices}
define EDGE {n_edges}
define POINT {n_points}

Parameters {{
    Units {{
        Coordinates "Å"
    }}
    SpatialGraphUnitsVertex {{
    }}
    SpatialGraphUnitsEdge {{
    }}
    SpatialGraphUnitsPoint {{
        thickness {{
            Unit -1,
            Dimension -1
        }}
    }}
    ContentType "HxSpatialGraph",
    Description "Microtubule Interaction Exports"
}}

VERTEX {{ float[3] VertexCoordinates }} @1
EDGE {{ int[2] EdgeConnectivity }} @2
EDGE {{ int NumEdgePoints }} @3
"""
        # Add class markers to header
        for i in range(1, 9):
             header += f"EDGE {{ int Class_{i} }} @{i+3}\n"
        
        point_idx = 4 + 8
        header += f"POINT {{ float[3] EdgePointCoordinates }} @{point_idx}\n"
        
        header += "\n# Data section follows\n@1\n"

        with open(file_path, 'w') as f:
            f.write(header)
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            if n_edges > 0:
                f.write("\n@2\n")
                for e in edges:
                    f.write(f"{e['v1']} {e['v2']}\n")
                
                f.write("\n@3\n")
                for e in edges:
                    f.write(f"{e['n_pts']}\n")
                
                # Write class masks
                for cls_id in range(1, 9):
                    f.write(f"\n@{cls_id+3}\n")
                    for e in edges:
                        f.write(f"{1 if e['class'] == cls_id else 0}\n")
                
            if n_points > 0:
                f.write(f"\n@{point_idx}\n")
                for p in points:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    def save_points_as_am(self, file_path, points, values=None, label="Proximity"):
        """
        Exports a point cloud (e.g., heatmap) to an AmiraMesh (.am) ASCII 3.0 file.
        Uses HxSpatialGraph type with 0 edges for better compatibility.
        """
        n_points = len(points)
        header = f"""# AmiraMesh 3D ASCII 3.0

define VERTEX {n_points}
define EDGE 0
define POINT 0

Parameters {{
    Units {{
        Coordinates "Å"
    }}
    SpatialGraphUnitsVertex {{
    }}
    SpatialGraphUnitsEdge {{
    }}
    SpatialGraphUnitsPoint {{
        thickness {{
            Unit -1,
            Dimension -1
        }}
    }}
    ContentType "HxSpatialGraph",
    Description "Microtubule Interaction Heatmap"
}}

VERTEX {{ float[3] VertexCoordinates }} @1
"""
        if values is not None:
            header += f"VERTEX {{ float {label} }} @2\n"
            edge_idx = 3
        else:
            edge_idx = 2
            
        header += f"EDGE {{ int[2] EdgeConnectivity }} @{edge_idx}\n"
        header += f"EDGE {{ int NumEdgePoints }} @{edge_idx+1}\n"
        header += f"POINT {{ float[3] EdgePointCoordinates }} @{edge_idx+2}\n"
        
        header += "\n# Data section follows\n@1\n"
        
        with open(file_path, 'w') as f:
            f.write(header)
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            if values is not None:
                f.write("\n@2\n")
                for v in values:
                    f.write(f"{v:.6f}\n")

    def export_dual_class_heatmaps(self, interaction_df, output_dir, dist_threshold):
        """
        Exports two separate heatmap .AM files, one for each interacting MT class.
        This allows displaying two intensity legends for the two quantified MTs in Amira.
        """
        if interaction_df.empty:
            print("No interactions found to export heatmaps.")
            return []

        seg_map = {s['segment_id']: s for s in self.segments}
        
        # Identify the main classes involved in the interactions
        all_involved_classes = sorted(list(set(interaction_df['Class_Ref'].unique()) | 
                                         set(interaction_df['Class_Neighbor'].unique())))
        
        exported_files = []
        
        for cls_id in all_involved_classes:
            cls_name = self.class_mapping.get(cls_id, f"Class_{cls_id}").replace(" ", "_").replace("-", "_")
            
            # Find all segments of this class that are involved in any interaction
            # (either as reference or as neighbor)
            cls_seg_ids = set(interaction_df[interaction_df['Class_Ref'] == cls_id]['Ref_Seg_ID']) | \
                          set(interaction_df[interaction_df['Class_Neighbor'] == cls_id]['Neighbor_Seg_ID'])
            
            if not cls_seg_ids:
                continue
            
            all_pts = []
            all_dists = []
            
            for sid in cls_seg_ids:
                seg = seg_map[sid]
                # Find all partner segments this specific segment interacted with
                partners = set(interaction_df[interaction_df['Ref_Seg_ID'] == sid]['Neighbor_Seg_ID']) | \
                           set(interaction_df[interaction_df['Neighbor_Seg_ID'] == sid]['Ref_Seg_ID'])
                
                if not partners:
                    continue
                
                # Stack all points from all partners to find minimum distance from each point in 'seg' 
                # to the closest point in ANY partner segment.
                partner_pts = np.vstack([seg_map[pid]['points'] for pid in partners])
                
                # Compute distance from each point in the current segment to the nearest point among all partners
                dists = cdist(seg['points'], partner_pts).min(axis=1)
                mask = dists <= dist_threshold
                
                if np.any(mask):
                    all_pts.append(seg['points'][mask])
                    all_dists.append(dists[mask])
            
            if all_pts:
                final_pts = np.vstack(all_pts)
                final_dists = np.concatenate(all_dists)
                
                file_path = os.path.join(output_dir, f"interaction_heatmap_{cls_name}.am")
                self.save_points_as_am(file_path, final_pts, values=final_dists, label=f"Proximity_{cls_name}")
                exported_files.append(file_path)
                print(f"Exported heatmap for {cls_name}: {len(final_pts)} points -> {file_path}")
        
        return exported_files

    def compute_tortuosity(self, c1, c2, bin_size=1000.0, selected_classes=None, output_dir="output"):
        """
        Calculates the tortuosity of microtubules along the spindle axis defined by c1 and c2.
        Bins the spindle axis and calculates tortuosity for each MT within each bin.
        Exports results to two CSV files.
        """
        if selected_classes is None:
            selected_classes = list(self.class_mapping.keys())
        
        # Define Spindle Axis
        spindle_vec = c2 - c1
        spindle_len = np.linalg.norm(spindle_vec)
        if spindle_len == 0:
            print("Error: Spindle length is zero. Cannot define axis.")
            return
        spindle_unit = spindle_vec / spindle_len
        
        results = []
        
        for seg in self.segments:
            if seg['mt_class'] not in selected_classes:
                continue
            
            pts = seg['points']
            # Project points onto spindle axis relative to c1
            projections = np.dot(pts - c1, spindle_unit)
            
            # Determine bins for each point
            bin_indices = (projections // bin_size).astype(int)
            unique_bins = np.unique(bin_indices)
            
            for b_idx in unique_bins:
                # Find points belonging to this bin
                mask = bin_indices == b_idx
                if np.sum(mask) < 2:
                    continue
                
                bin_pts = pts[mask]
                
                # Calculate chord length: distance between first and last point in bin
                chord = np.linalg.norm(bin_pts[-1] - bin_pts[0])
                
                # Calculate segment length: sum of distances between consecutive points in bin
                seg_len = np.sum(np.linalg.norm(np.diff(bin_pts, axis=0), axis=1))
                
                # Tortuosity calculation
                tortuosity = seg_len / chord if chord != 0 else np.nan
                
                results.append({
                    'Microtubule_ID': seg['segment_id'],
                    'Class_Name': self.class_mapping.get(seg['mt_class'], "Unknown"),
                    'Bin_Position': b_idx * bin_size,
                    'Bin_Number': b_idx,
                    'Tortuosity': tortuosity
                })
        
        df_quant = pd.DataFrame(results)
        if df_quant.empty:
            print("No data found for tortuosity quantification.")
            return

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Export individual quantification
        quant_file = os.path.join(output_dir, "Microtubules_tortuosity_quantification.csv")
        df_quant.to_csv(quant_file, index=False)
        print(f"Exported individual tortuosity to {quant_file}")
        
        # Calculate and export average tortuosity
        df_avg = df_quant.groupby(['Class_Name', 'Bin_Position', 'Bin_Number'])['Tortuosity'].mean().reset_index()
        df_avg.rename(columns={'Tortuosity': 'Mean_Tortuosity'}, inplace=True)
        
        avg_file = os.path.join(output_dir, "Microtubules_average_tortuosity.csv")
        df_avg.to_csv(avg_file, index=False)
        print(f"Exported average tortuosity to {avg_file}")
        
        return df_quant, df_avg

    def export_tortuosity_heatmap(self, df_quant, output_dir, bin_size=1000.0, c1=None, c2=None):
        """
        Exports a 3D heatmap of tortuosity using Plotly (HTML) and Amira (.am).
        Uses sky-blue intensity for tortuosity levels.
        """
        if df_quant.empty:
            print("No tortuosity data to export.")
            return

        import plotly.graph_objects as go
        
        seg_map = {s['segment_id']: s for s in self.segments}
        
        # Define Spindle Axis for projection if needed
        spindle_unit = None
        if c1 is not None and c2 is not None:
            spindle_vec = c2 - c1
            spindle_len = np.linalg.norm(spindle_vec)
            if spindle_len > 0:
                spindle_unit = spindle_vec / spindle_len

        all_pts = []
        all_torts = []

        for _, row in df_quant.iterrows():
            sid = row['Microtubule_ID']
            b_idx = row['Bin_Number']
            tort = row['Tortuosity']
            
            if sid not in seg_map: continue
            seg = seg_map[sid]
            pts = seg['points']
            
            # Use projections to find exact points in this bin
            if spindle_unit is not None:
                projections = np.dot(pts - c1, spindle_unit)
                bin_mask = (projections // bin_size).astype(int) == b_idx
                bin_pts = pts[bin_mask]
                if len(bin_pts) > 0:
                    all_pts.append(bin_pts)
                    all_torts.append(np.full(len(bin_pts), tort))
            else:
                all_pts.append(pts)
                all_torts.append(np.full(len(pts), tort))

        if not all_pts:
            print("No points found for heatmap.")
            return

        final_pts = np.vstack(all_pts)
        final_torts = np.concatenate(all_torts)

        # 1. Export as .AM file
        am_file = os.path.join(output_dir, "microtubule_tortuosity_heatmap.am")
        self.save_points_as_am(am_file, final_pts, values=final_torts, label="Tortuosity")
        print(f"Exported tortuosity heatmap to {am_file}")

        # 2. Export as .HTML file (Plotly)
        skyblue_colorscale = [
            [0.0, 'lightskyblue'],
            [0.5, 'skyblue'],
            [1.0, 'deepskyblue']
        ]

        fig = go.Figure(data=[go.Scatter3d(
            x=final_pts[:, 0], y=final_pts[:, 1], z=final_pts[:, 2],
            mode='markers',
            marker=dict(size=2, color=final_torts, colorscale=skyblue_colorscale,
                        colorbar=dict(title="Tortuosity"), opacity=0.8)
        )])

        fig.update_layout(
            title="Microtubule Tortuosity Heatmap",
            scene=dict(aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        html_file = os.path.join(output_dir, "microtubule_tortuosity_heatmap.html")
        fig.write_html(html_file)
        print(f"Exported interactive tortuosity heatmap to {html_file}")

        return html_file, am_file

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
