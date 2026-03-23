# Microtubule Interaction Analysis

An advanced toolkit for quantifying and visualizing pairwise proximity and spatial relationships between microtubule (MT) segments from Amira SpatialGraph data. This project includes an interactive Jupyter Notebook pipeline and a dedicated napari plugin.

## 🚀 Features

### 🔍 Quantification
- **Dual-Class Filtering**: Select specific reference and neighbor MT classes (e.g., Mid-MTs vs. Interdigitating-MTs).
- **Proximity Analysis**: Calculate point-wise distances between segments with configurable thresholds.
- **Spatial Metrics**: Computes Interaction Length, Angle (Parallel vs. Anti-parallel), and Spindle Axis Positioning (projected distance from spindle midpoint).
- **Hardware Acceleration**: Automatic scanning and support for **CPU (multi-core)** and **GPU (NVIDIA/CuPy)** computation.

### 📊 Visualization
- **3D Interaction Heatmap**: Specialized 3D plot showing only interacting microtubules with a color-coded proximity heatmap (Hot colormap: white/yellow for high proximity).
- **Interactive Reports**: Exportable Plotly HTML visualizations for deep exploration without specialized software.
- **Napari Plugin**: A GUI-based workbench for real-time 3D rendering and hardware management.

### 💾 High-Fidelity Exports
- **CSV**: Comprehensive interaction summaries and class-pair statistics.
- **AmiraMesh (.am)**: Coordinate-accurate exports of interacting segments and proximity point clouds (VertexSets) compatible with AMIRA software.
- **High-DPI Images**: Publication-ready PNG and SVG files (1200 DPI scale support).

## 🛠️ Installation & Setup

To ensure all dependencies (including the napari plugin) work correctly, we recommend using the local virtual environment.

### 1. Clone the repository
```bash
git clone https://github.com/cwilliamOkafornta/Microtubule_interaction_analysis.git
cd Microtubule_interaction_analysis
```

### 2. Set up the environment (Windows)
If you don't have a virtual environment yet, create one:
```powershell
python -m venv .
```

Activate the environment:
```powershell
.\Scripts\activate
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Install the napari plugin in editable mode
```powershell
pip install -e ./napari-mt-interaction
```

## 📖 Usage

### Option 1: Jupyter Notebook
Run the complete research pipeline via the interactive notebook:
```powershell
.\Scripts\python.exe -m jupyter notebook MT_Revamp_Task1.ipynb
```

### Option 2: Napari Plugin (GUI)
Launch napari directly from the virtual environment:
```powershell
.\Scripts\python.exe -m napari
```
*Or simply:*
```powershell
.\Scripts\napari.exe
```

Once napari opens:
1. Go to **Plugins > Microtubule Interaction Analyzer**.
2. Select your `.am` (SpatialGraph) and `.surf` (Surfaces) files.
3. Scan hardware and select your preferred GPU/CPU.
4. Click **Run Analysis**.

## 🧪 Testing
The toolkit includes a comprehensive test suite:
```powershell
.\Scripts\pytest tests/
```

## 🏗️ Project Structure
- `MT_Revamp_Task1.ipynb`: Primary research notebook.
- `mt_interaction_core.py`: Core logic for parsing and interaction math.
- `napari-mt-interaction/`: Source code for the napari plugin.
- `tomogram_analysis/`: Directory for input data (.am and .surf).
- `output/`: Generated CSVs, images, and AmiraMesh files.

## ⚖️ License
This project is licensed under the BSD-3-Clause License.
