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

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cwilliamOkafornta/Microtubule_interaction_analysis.git
   cd Microtubule_interaction_analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the napari plugin**:
   ```bash
   pip install -e ./napari-mt-interaction
   ```

## 📖 Usage

### Option 1: Jupyter Notebook
Open `MT_Revamp_Task1.ipynb` to run the complete research pipeline. The notebook is organized into 5 tasks:
- **Task 1-2**: Data loading and initial spindle visualization.
- **Task 3**: Advanced Interaction Analysis computation.
- **Task 4**: 3D Heatmap implementation.
- **Task 5**: Multi-format data exportation.

### Option 2: Napari Plugin (GUI)
1. Launch napari: `napari`
2. Open the plugin: **Plugins > Microtubule Interaction Analyzer**
3. Select your `.am` (SpatialGraph) and `.surf` (Surfaces) files.
4. Scan hardware and select your preferred GPU/CPU.
5. Click **Run Analysis**.

## 🧪 Testing
The toolkit includes a comprehensive test suite to ensure scientific accuracy and coordinate fidelity.

To run the tests:
```bash
pytest tests/
```
The tests cover:
- **Interaction Math**: Validates distance and orientation calculations.
- **IO Fidelity**: Ensures AmiraMesh saving/loading maintains 100% coordinate accuracy.
- **Hardware Robustness**: Verifies graceful fallback to CPU if GPU libraries are missing.

## 🏗️ Project Structure
- `MT_Revamp_Task1.ipynb`: Primary research notebook.
- `mt_interaction_core.py`: Core logic for parsing and interaction math.
- `napari-mt-interaction/`: Source code for the napari plugin.
- `tomogram_analysis/`: Directory for input data (.am and .surf).
- `output/`: Generated CSVs, images, and AmiraMesh files.

## ⚖️ License
This project is licensed under the BSD-3-Clause License.
