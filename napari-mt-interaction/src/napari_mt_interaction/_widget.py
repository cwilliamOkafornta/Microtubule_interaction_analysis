import os
import multiprocessing
import subprocess
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QComboBox, QLineEdit, QCheckBox, QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView, QSpinBox, QDoubleSpinBox
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
        self.tortuosity_df = None
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
        
        # Proximity Distance
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Proximity Threshold (Å):"))
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(100.0, 5000.0)
        self.spin_dist.setValue(500.0)
        dist_layout.addWidget(self.spin_dist)
        param_layout.addLayout(dist_layout)
        
        # Class Filters
        self.list_ref_classes = QListWidget()
        self.list_ref_classes.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_neighbor_classes = QListWidget()
        self.list_neighbor_classes.setSelectionMode(QAbstractItemView.MultiSelection)
        
        classes = {
            1: "SMTs", 2: "KMTs_pole1_In", 3: "KMTs_pole1_Out",
            4: "KMTs_pole2_In", 5: "KMTs_pole2_Out", 6: "Mid-MTs",
            7: "Interdigitating-MTs", 8: "Bridging-MTs"
        }
        for cid, name in classes.items():
            item1 = QListWidgetItem(name); item1.setData(Qt.UserRole, cid); self.list_ref_classes.addItem(item1)
            item2 = QListWidgetItem(name); item2.setData(Qt.UserRole, cid); self.list_neighbor_classes.addItem(item2)
        
        param_layout.addWidget(QLabel("Reference MT Classes:"))
        param_layout.addWidget(self.list_ref_classes)
        param_layout.addWidget(QLabel("Neighbor MT Classes:"))
        param_layout.addWidget(self.list_neighbor_classes)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 3. Tortuosity Group
        tort_group = QGroupBox("Tortuosity Parameters")
        tort_layout = QVBoxLayout()
        
        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Bin Size (Å):"))
        self.spin_bin = QDoubleSpinBox()
        self.spin_bin.setRange(100.0, 10000.0)
        self.spin_bin.setValue(1000.0)
        bin_layout.addWidget(self.spin_bin)
        tort_layout.addLayout(bin_layout)

        self.list_tort_classes = QListWidget()
        self.list_tort_classes.setSelectionMode(QAbstractItemView.MultiSelection)
        for cid, name in classes.items():
            item = QListWidgetItem(name); item.setData(Qt.UserRole, cid); self.list_tort_classes.addItem(item)
        
        tort_layout.addWidget(QLabel("Target MT Classes:"))
        tort_layout.addWidget(self.list_tort_classes)
        
        tort_group.setLayout(tort_layout)
        layout.addWidget(tort_group)

        # 4. Hardware & Run
        self.combo_gpu = QComboBox()
        self.btn_run = QPushButton("Run Interaction Analysis")
        self.btn_run.clicked.connect(self._run_interaction)
        self.btn_run_tort = QPushButton("Run Tortuosity Analysis")
        self.btn_run_tort.clicked.connect(self._run_tortuosity)
        
        layout.addWidget(QLabel("Hardware Acceleration:"))
        layout.addWidget(self.combo_gpu)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_run_tort)

        # 5. Output
        self.btn_select_out = QPushButton("Set Output Directory")
        self.btn_select_out.clicked.connect(self._select_out_dir)
        self.lbl_out = QLabel("No output directory set")
        layout.addWidget(self.btn_select_out)
        layout.addWidget(self.lbl_out)
        
        self.setLayout(layout)

    def _scan_hardware(self):
        self.combo_gpu.clear()
        try:
            import cupy as cp
            count = cp.cuda.runtime.getDeviceCount()
            for i in range(count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                self.combo_gpu.addItem(f"GPU {i}: {props['name'].decode()}", i)
        except:
            self.combo_gpu.addItem("GPU Unavailable (CPU Only)")
        
        cpu_count = multiprocessing.cpu_count()
        self.combo_gpu.addItem(f"CPU (Multi-core: {cpu_count})", -1)

    def _select_am_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select SpatialGraph File", "", "AmiraMesh (*.am)")
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
            self.lbl_out.setText(dir_path)

    def _run_interaction(self):
        if not self.am_file_path or not self.surf_file_path:
            print("Please select both .am and .surf files.")
            return
        dist_threshold = self.spin_dist.value()
        ref_classes = [item.data(Qt.UserRole) for item in self.list_ref_classes.selectedItems()]
        neighbor_classes = [item.data(Qt.UserRole) for item in self.list_neighbor_classes.selectedItems()]
        gpu_selection = self.combo_gpu.currentText()
        use_gpu = gpu_selection and "GPU Unavailable" not in gpu_selection
        self.btn_run.setEnabled(False); self.btn_run.setText("Running...")
        worker = self._interaction_worker(self.am_file_path, self.surf_file_path, dist_threshold, ref_classes, neighbor_classes, use_gpu)
        worker.returned.connect(self._on_interaction_complete)
        worker.start()

    @thread_worker
    def _interaction_worker(self, am_path, surf_path, dist_threshold, ref_classes, neighbor_classes, use_gpu):
        graph = MicrotubuleSpatialGraph(); graph.load_from_am(am_path)
        surf1, surf2, c1, c2 = load_surfaces(surf_path)
        interaction_df = compute_advanced_interactions(graph.segments, dist_threshold, ref_classes, neighbor_classes, c1, c2, use_gpu=use_gpu)
        return graph, interaction_df, surf1, surf2, c1, c2, dist_threshold

    def _on_interaction_complete(self, results):
        self.graph, self.interaction_df, self.surf1, self.surf2, self.c1, self.c2, dist_threshold = results
        print(f"Interaction analysis complete: {len(self.interaction_df)} pairs.")
        self._visualize_microtubules()
        self._visualize_interaction_heatmap(dist_threshold)
        if self.output_dir: self._export_interaction(dist_threshold)
        self.btn_run.setEnabled(True); self.btn_run.setText("Run Interaction Analysis")

    def _run_tortuosity(self):
        if not self.am_file_path or not self.surf_file_path:
            print("Please select both .am and .surf files.")
            return
        bin_size = self.spin_bin.value()
        tort_classes = [item.data(Qt.UserRole) for item in self.list_tort_classes.selectedItems()]
        self.btn_run_tort.setEnabled(False); self.btn_run_tort.setText("Running...")
        worker = self._tortuosity_worker(self.am_file_path, self.surf_file_path, bin_size, tort_classes)
        worker.returned.connect(self._on_tortuosity_complete)
        worker.start()

    @thread_worker
    def _tortuosity_worker(self, am_path, surf_path, bin_size, tort_classes):
        graph = MicrotubuleSpatialGraph(); graph.load_from_am(am_path)
        surf1, surf2, c1, c2 = load_surfaces(surf_path)
        df_quant, df_avg = graph.compute_tortuosity(c1, c2, bin_size=bin_size, selected_classes=tort_classes, output_dir=None)
        return graph, df_quant, df_avg, surf1, surf2, c1, c2, bin_size

    def _on_tortuosity_complete(self, results):
        self.graph, self.tortuosity_df, self.summary_df, self.surf1, self.surf2, self.c1, self.c2, bin_size = results
        print(f"Tortuosity analysis complete: {len(self.tortuosity_df)} bins quantified.")
        self._visualize_tortuosity_heatmap(bin_size)
        if self.output_dir: self._export_tortuosity(bin_size)
        self.btn_run_tort.setEnabled(True); self.btn_run_tort.setText("Run Tortuosity Analysis")

    def _visualize_microtubules(self):
        if not self.graph: return
        self.viewer.layers.clear()
        colors = {
            1: "gray", 2: "blue", 3: "cyan", 4: "orange", 
            5: "yellow", 6: "green", 7: "red", 8: "magenta"
        }
        for cls_id, name in self.graph.class_mapping.items():
            if cls_id == 0: continue
            cls_segs = [s['points'] for s in self.graph.segments if s['mt_class'] == cls_id]
            if cls_segs:
                self.viewer.add_shapes(cls_segs, shape_type='path', edge_width=50, 
                                       edge_color=colors.get(cls_id, "white"), name=name)
        # Add centroids
        self.viewer.add_points([self.c1, self.c2], size=1000, face_color='yellow', name='Centroids')

    def _visualize_interaction_heatmap(self, threshold):
        # Implementation for real-time heatmap in napari if desired
        pass

    def _visualize_tortuosity_heatmap(self, bin_size):
        pass

    def _export_interaction(self, threshold):
        out_csv = os.path.join(self.output_dir, "interaction_summary.csv")
        self.interaction_df.to_csv(out_csv, index=False)
        self.graph.export_dual_class_heatmaps(self.interaction_df, self.output_dir, threshold)

    def _export_tortuosity(self, bin_size):
        self.graph.export_tortuosity_heatmap(self.tortuosity_df, self.output_dir, bin_size, self.c1, self.c2)

    def _on_export_full(self):
        if not self.graph or not self.output_dir: return
        am_path = os.path.join(self.output_dir, "full_spindle.am")
        self.graph.save_as_am(am_path)
