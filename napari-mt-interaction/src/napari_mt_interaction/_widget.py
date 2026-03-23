import os
import multiprocessing
import subprocess
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QComboBox, QLineEdit, QCheckBox, QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView, QSpinBox
)
from qtpy.QtCore import Qt
from napari.qt.threading import thread_worker
import plotly.graph_objects as go

# Import core functionalities
from .core import MicrotubuleSpatialGraph, load_surfaces, compute_advanced_interactions

class MTInteractionWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        self.am_file_path = None
        self.surf_file_path = None
        self.output_dir = None
        
        self.graph = None
        self.interaction_df = None
        self.summary_df = None
        self.c1 = None
        self.c2 = None
        self.surf1 = None
        self.surf2 = None

        self._init_ui()
        self._scan_hardware()
        
    def _init_ui(self):
        layout = QVBoxLayout()
        
        # 1. Inputs Group
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout()
        
        self.btn_load_am = QPushButton("Load SpatialGraph (.am)")
        self.btn_load_am.clicked.connect(self._select_am_file)
        self.lbl_am = QLabel("No .am file selected")
        input_layout.addWidget(self.btn_load_am)
        input_layout.addWidget(self.lbl_am)
        
        self.btn_load_surf = QPushButton("Load Chromosome Surfaces (.surf)")
        self.btn_load_surf.clicked.connect(self._select_surf_file)
        self.lbl_surf = QLabel("No .surf file selected")
        input_layout.addWidget(self.btn_load_surf)
        input_layout.addWidget(self.lbl_surf)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 2. Parameters Group
        param_group = QGroupBox("Interaction Parameters")
        param_layout = QVBoxLayout()
        
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Interaction Dist (Å):"))
        self.spin_dist = QSpinBox()
        self.spin_dist.setMaximum(5000)
        self.spin_dist.setValue(500)
        dist_layout.addWidget(self.spin_dist)
        param_layout.addLayout(dist_layout)
        
        # Class selectors
        class_names = [
            (1, "SMTs"), (2, "KMTs_pole1_Inside"), (3, "KMTs_pole1_Outside"),
            (4, "KMTs_pole2_Inside"), (5, "KMTs_pole2_Outside"), (6, "Mid-MTs"),
            (7, "Interdigitating-MTs"), (8, "Bridging-MTs")
        ]
        
        self.list_ref_classes = QListWidget()
        self.list_ref_classes.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_neighbor_classes = QListWidget()
        self.list_neighbor_classes.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        for c_id, c_name in class_names:
            item1 = QListWidgetItem(f"{c_id}: {c_name}")
            item1.setData(Qt.UserRole, c_id)
            self.list_ref_classes.addItem(item1)
            
            item2 = QListWidgetItem(f"{c_id}: {c_name}")
            item2.setData(Qt.UserRole, c_id)
            self.list_neighbor_classes.addItem(item2)
            
            # Default selection (6 and 7 as in Task 3)
            if c_id in [6, 7]:
                item1.setSelected(True)
                item2.setSelected(True)
                
        param_layout.addWidget(QLabel("Reference Classes (Select multiple):"))
        param_layout.addWidget(self.list_ref_classes)
        param_layout.addWidget(QLabel("Neighbor Classes (Select multiple):"))
        param_layout.addWidget(self.list_neighbor_classes)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 3. Hardware Group
        hw_group = QGroupBox("Hardware Selection")
        hw_layout = QVBoxLayout()
        
        self.btn_scan_hw = QPushButton("Scan Hardware")
        self.btn_scan_hw.clicked.connect(self._scan_hardware)
        hw_layout.addWidget(self.btn_scan_hw)
        
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU:"))
        self.combo_cpu = QComboBox()
        cpu_layout.addWidget(self.combo_cpu)
        hw_layout.addLayout(cpu_layout)
        
        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel("GPU:"))
        self.combo_gpu = QComboBox()
        gpu_layout.addWidget(self.combo_gpu)
        hw_layout.addLayout(gpu_layout)
        
        hw_group.setLayout(hw_layout)
        layout.addWidget(hw_group)
        
        # 4. Export Group
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        self.btn_out_dir = QPushButton("Select Output Directory")
        self.btn_out_dir.clicked.connect(self._select_out_dir)
        self.lbl_out_dir = QLabel("No output dir selected")
        export_layout.addWidget(self.btn_out_dir)
        export_layout.addWidget(self.lbl_out_dir)
        
        self.chk_csv = QCheckBox("Export CSV Tables")
        self.chk_csv.setChecked(True)
        self.chk_html = QCheckBox("Export Interactive Heatmap (HTML)")
        self.chk_html.setChecked(True)
        self.chk_png = QCheckBox("Export Heatmap Image (PNG/SVG)")
        self.chk_png.setChecked(True)
        self.chk_am = QCheckBox("Export AmiraMesh Files (.am)")
        self.chk_am.setChecked(True)
        
        export_layout.addWidget(self.chk_csv)
        export_layout.addWidget(self.chk_html)
        export_layout.addWidget(self.chk_png)
        export_layout.addWidget(self.chk_am)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # 5. Run Button
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self._run_analysis)
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(self.btn_run)
        
        self.setLayout(layout)
        
    def _select_am_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select AmiraMesh File", "", "AmiraMesh (*.am)")
        if file_path:
            self.am_file_path = file_path
            self.lbl_am.setText(os.path.basename(file_path))
            
    def _select_surf_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Surfaces File", "", "Surfaces (*.surf)")
        if file_path:
            self.surf_file_path = file_path
            self.lbl_surf.setText(os.path.basename(file_path))
            
    def _select_out_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.lbl_out_dir.setText(dir_path)

    def _scan_hardware(self):
        from .core import HAS_GPU
        self.combo_cpu.clear()
        self.combo_gpu.clear()
        
        # Scan CPU
        cpu_count = multiprocessing.cpu_count()
        self.combo_cpu.addItem(f"CPU ({cpu_count} logical cores)")
        
        # Scan GPU
        if HAS_GPU:
            gpus = []
            try:
                result = subprocess.check_output(['nvidia-smi', '-L'], text=True)
                for line in result.strip().split('\n'):
                    if line: gpus.append(line)
            except Exception:
                try:
                    result = subprocess.check_output(['wmic', 'path', 'win32_VideoController', 'get', 'name'], text=True)
                    lines = result.strip().split('\n')[1:]
                    for line in lines:
                        if line.strip(): gpus.append(line.strip())
                except Exception:
                    pass
            
            if gpus:
                self.combo_gpu.addItems(gpus)
            else:
                self.combo_gpu.addItem("Generic CUDA GPU")
        else:
            self.combo_gpu.addItem("GPU Unavailable (Missing CUDA Libs)")
            self.combo_gpu.setEnabled(False)

    def _run_analysis(self):
        if not self.am_file_path or not self.surf_file_path:
            print("Please select both .am and .surf files.")
            return
            
        dist_threshold = self.spin_dist.value()
        ref_classes = [item.data(Qt.UserRole) for item in self.list_ref_classes.selectedItems()]
        neighbor_classes = [item.data(Qt.UserRole) for item in self.list_neighbor_classes.selectedItems()]
        
        # Check if GPU is selected
        gpu_selection = self.combo_gpu.currentText()
        use_gpu = False
        if gpu_selection and "No discrete GPU" not in gpu_selection:
            use_gpu = True
            
        # Disable button during run
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")
        
        worker = self._compute_worker(
            self.am_file_path, self.surf_file_path, 
            dist_threshold, ref_classes, neighbor_classes, use_gpu
        )
        worker.returned.connect(self._on_analysis_complete)
        worker.start()
        
    @thread_worker
    def _compute_worker(self, am_path, surf_path, dist_threshold, ref_classes, neighbor_classes, use_gpu):
        # 1. Load AM
        print(f"Loading AM: {am_path}")
        graph = MicrotubuleSpatialGraph()
        graph.load_from_am(am_path)
        
        # 2. Load Surfaces
        print(f"Loading Surf: {surf_path}")
        surf1, surf2, c1, c2 = load_surfaces(surf_path)
        
        # 3. Compute Interactions
        print(f"Computing Interactions (GPU: {use_gpu})...")
        interaction_df = compute_advanced_interactions(
            graph.segments, dist_threshold, ref_classes, neighbor_classes, c1, c2, use_gpu=use_gpu
        )
        
        return graph, interaction_df, surf1, surf2, c1, c2, dist_threshold

    def _on_analysis_complete(self, results):
        self.graph, self.interaction_df, self.surf1, self.surf2, self.c1, self.c2, dist_threshold = results
        print(f"Found {len(self.interaction_df)} interactions.")
        
        self._visualize_in_napari()
        
        if self.output_dir:
            self._export_results(dist_threshold)
            
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Analysis")
        print("Analysis Complete!")

    def _visualize_in_napari(self):
        # Colors definition
        colors = {
            1: [0.53, 0.53, 0.53, 1], # "#888888"
            2: [0.12, 0.46, 0.70, 1], # "#1f77b4"
            3: [0.68, 0.78, 0.90, 1], # "#aec7e8"
            4: [1.00, 0.49, 0.05, 1], # "#ff7f0e"
            5: [1.00, 0.73, 0.47, 1], # "#ffbb78"
            6: [0.17, 0.62, 0.17, 1], # "#2ca02c"
            7: [0.83, 0.15, 0.15, 1], # "#d62728"
            8: [0.58, 0.40, 0.74, 1]  # "#9467bd"
        }
        
        paths = []
        path_colors = []
        
        for seg in self.graph.segments:
            paths.append(seg['points'])
            color = colors.get(seg['mt_class'], [1,1,1,1])
            path_colors.append(color)
            
        self.viewer.add_shapes(
            paths,
            shape_type='path',
            edge_width=2,
            edge_color=path_colors,
            name='Microtubules'
        )
        
        # Visualization of Heatmap Points
        if not self.interaction_df.empty:
            from scipy.spatial.distance import cdist
            seg_map = {s['segment_id']: s for s in self.graph.segments}
            all_int_pts = []
            all_int_dists = []
            
            for _, row in self.interaction_df.iterrows():
                s1, s2 = seg_map[row['Ref_Seg_ID']], seg_map[row['Neighbor_Seg_ID']]
                dists = cdist(s1['points'], s2['points']).min(axis=1)
                mask = dists <= self.spin_dist.value()
                if np.any(mask):
                    all_int_pts.append(s1['points'][mask])
                    all_int_dists.append(dists[mask])
                    
            if all_int_pts:
                heatmap_pts = np.vstack(all_int_pts)
                heatmap_dists = np.concatenate(all_int_dists)
                
                self.viewer.add_points(
                    heatmap_pts,
                    features={'distance': heatmap_dists},
                    face_color='distance',
                    face_colormap='hot_r', # reverse hot
                    size=3,
                    name='Interaction Heatmap'
                )

    def _export_results(self, dist_threshold):
        print(f"Exporting to {self.output_dir}...")
        
        if not self.interaction_df.empty:
            self.interaction_df['Class_Ref_Name'] = self.interaction_df['Class_Ref'].map(self.graph.class_mapping)
            self.interaction_df['Class_Neighbor_Name'] = self.interaction_df['Class_Neighbor'].map(self.graph.class_mapping)
            
            self.summary_df = self.interaction_df.groupby(['Class_Ref_Name', 'Class_Neighbor_Name', 'Orientation']).agg({
                'Int_Length_A': 'mean', 'Angle_deg': 'mean', 'Spindle_Pos_A': 'mean', 'Mean_Dist_A': 'mean'
            }).reset_index()

            if self.chk_csv.isChecked():
                self.interaction_df.to_csv(os.path.join(self.output_dir, "microtubule_interaction_summary.csv"), index=False)
                self.summary_df.to_csv(os.path.join(self.output_dir, "mean_interaction_length_by_class.csv"), index=False)
                
            if self.chk_am.isChecked():
                interacting_ids = set(self.interaction_df['Ref_Seg_ID'].unique()) | set(self.interaction_df['Neighbor_Seg_ID'].unique())
                interacting_segments = [s for s in self.graph.segments if s['segment_id'] in interacting_ids]
                
                self.graph.save_as_am(os.path.join(self.output_dir, "interaction_heatmap_segments.am"), segments=interacting_segments)
                self.graph.save_as_am(os.path.join(self.output_dir, "full_spindle_with_surfaces.am"), segments=self.graph.segments, surface_verts=[self.surf1, self.surf2])

            if self.chk_html.isChecked() or self.chk_png.isChecked():
                self._export_plotly(interacting_ids, dist_threshold)

    def _export_plotly(self, interacting_ids, dist_threshold):
        fig_heatmap = go.Figure()
        seg_map = {s['segment_id']: s for s in self.graph.segments}
        
        colors = {
            0: "#FFFFFF", 1: "#888888", 2: "#1f77b4", 3: "#aec7e8",
            4: "#ff7f0e", 5: "#ffbb78", 6: "#2ca02c", 7: "#d62728", 8: "#9467bd"
        }
        
        # We need SELECTED_CLASSES logic from before, just using unique classes from interacting_ids
        involved_classes = list(set([seg_map[sid]['mt_class'] for sid in interacting_ids]))
        
        for cls_id in involved_classes:
            cls_interacting_ids = [sid for sid in interacting_ids if seg_map[sid]['mt_class'] == cls_id]
            if not cls_interacting_ids: continue
            
            x_coords, y_coords, z_coords = [], [], []
            for sid in cls_interacting_ids:
                pts = seg_map[sid]['points']
                x_coords.extend(pts[:, 0]); x_coords.append(None)
                y_coords.extend(pts[:, 1]); y_coords.append(None)
                z_coords.extend(pts[:, 2]); z_coords.append(None)
            
            fig_heatmap.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines', line=dict(color=colors.get(cls_id, "white"), width=1.5),
                opacity=0.4, hoverinfo='none', name=f"Class {cls_id}"
            ))

        from scipy.spatial.distance import cdist
        all_int_pts, all_int_dists = [], []
        for _, row in self.interaction_df.iterrows():
            s1, s2 = seg_map[row['Ref_Seg_ID']], seg_map[row['Neighbor_Seg_ID']]
            dists = cdist(s1['points'], s2['points']).min(axis=1)
            mask = dists <= dist_threshold
            if np.any(mask):
                all_int_pts.append(s1['points'][mask])
                all_int_dists.append(dists[mask])

        if all_int_pts:
            heatmap_pts = np.vstack(all_int_pts)
            heatmap_dists = np.concatenate(all_int_dists)
            
            if len(heatmap_pts) > 50000:
                idx = np.random.choice(len(heatmap_pts), 50000, replace=False)
                heatmap_pts = heatmap_pts[idx]
                heatmap_dists = heatmap_dists[idx]

            fig_heatmap.add_trace(go.Scatter3d(
                x=heatmap_pts[:, 0], y=heatmap_pts[:, 1], z=heatmap_pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=heatmap_dists, colorscale='Hot', reversescale=True,
                            cmin=0, cmax=dist_threshold, showscale=True),
                name='Interaction Proximity'
            ))

        fig_heatmap.update_layout(
            title="3D Interaction Proximity Heatmap",
            scene=dict(aspectmode='data', bgcolor='black',
                       xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), zaxis=dict(showgrid=False)),
            margin=dict(r=0, l=0, b=0, t=40)
        )

        if self.chk_html.isChecked():
            html_path = os.path.join(self.output_dir, "interaction_heatmap.html")
            fig_heatmap.write_html(html_path)
            
        if self.chk_png.isChecked():
            try:
                fig_heatmap.write_image(os.path.join(self.output_dir, "interaction_heatmap.png"), scale=3)
                fig_heatmap.write_image(os.path.join(self.output_dir, "interaction_heatmap.svg"))
            except Exception as e:
                print(f"Image Export Failed: {e}")
