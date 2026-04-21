# Microtubule Interaction Analysis

An advanced toolkit for quantifying and visualizing pairwise proximity and spatial relationships between microtubule (MT) segments from Amira SpatialGraph data. This project includes an interactive Jupyter Notebook pipeline and a dedicated napari plugin.

## 🚀 Features

### 🔍 Quantification
- **Dual-Class Filtering**: Select specific reference and neighbor MT classes (e.g., Mid-MTs vs. Interdigitating-MTs).
- **Proximity Analysis**: Calculate point-wise distances between segments with configurable thresholds.
- **Tortuosity Analysis**: Measure the curviness of microtubules along the spindle axis with configurable binning.
- **Spatial Metrics**: Computes Interaction Length, Angle (Parallel vs. Anti-parallel), and Spindle Axis Positioning (projected distance from spindle midpoint).
- **Hardware Acceleration**: Automatic scanning and support for **CPU (multi-core)** and **GPU (NVIDIA/CuPy)** computation.

### 📊 Visualization
- **3D Interaction Heatmap**: Specialized 3D plot showing only interacting microtubules with a color-coded proximity heatmap (Hot colormap: white/yellow for high proximity).
- **3D Tortuosity Heatmap**: Visualization of microtubule curviness using **shades of sky-blue** intensity (Blues colormap).
- **Interactive Reports**: Exportable Plotly HTML visualizations for deep exploration without specialized software.
- **Napari Plugin**: A GUI-based workbench for real-time 3D rendering and hardware management.

### 💾 High-Fidelity Exports
- **CSV**: Comprehensive interaction summaries, tortuosity quantifications, and class-pair statistics.
- **AmiraMesh (.am)**: Coordinate-accurate exports of interacting segments and proximity/tortuosity point clouds compatible with AMIRA software.
- **High-DPI Images**: Publication-ready PNG and SVG files (1200 DPI scale support).

## 📐 Mathematical Formulas

### 1. Interaction Proximity
Distance between point $P_i$ on microtubule $A$ and microtubule $B$ is defined as:
$$d(P_i, B) = \min_{Q_j \in B} \|P_i - Q_j\|$$
An interaction is recorded if $d(P_i, B) \le \text{Threshold}$.

### 2. Tortuosity
Tortuosity ($\tau$) is calculated for each microtubule segment within a binned position along the spindle axis:
$$\tau = \frac{L}{C}$$
Where:
- $L$ = **Arc Length**: The sum of Euclidean distances between all consecutive points of the microtubule segment within the bin.
  $$L = \sum_{k=1}^{n-1} \|P_{k+1} - P_k\|$$
- $C$ = **Chord Length**: The straight-line distance between the first and last points of the microtubule segment within the bin.
  $$C = \|P_n - P_1\|$$

The spindle axis is defined as the vector connecting the centroids of the two chromosome surfaces ($C_1$ and $C_2$).

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
.\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install the napari plugin in editable mode
```powershell
.\Scripts\python.exe -m pip install -e ./napari-mt-interaction
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
3. **Optional**: Select a Spindle Poles `.am` file (e.g., `T0596_08Ana_p1p2_zexpanded.am`). If provided, the positions of the spindle poles will be used as the spindle axis for tortuosity and interaction projections, instead of the geometric center of the chromosomes.
4. Configure **Interaction** or **Tortuosity** parameters.
5. Scan hardware and select your preferred GPU/CPU.
6. Click **Run Analysis**.

## 🛡️ Software Audit & Quality
As of April 2026, this software has undergone a comprehensive internal review for:
- **Accuracy**: High-precision spindle pole axis definition implemented specifically for tortuosity quantification.
- **Robustness**: Maintain original chromosome-based axis for interaction proximity analysis.
- **Integrity**: Dedicated `load_pole_centroids` logic ensures zero impact on primary microtubule data structures.
- **Review**: This software is prepared for review by **CODEX** and **CLAUDE**.

## 🧪 Testing & Validation
The toolkit includes a comprehensive test suite for the core logic and plugin integration. 

### 1. Run core tests
```powershell
.\Scripts\python.exe -m pytest tests/
```

### 2. Validate Plugin Manifest
Ensure the napari plugin is correctly defined:
```powershell
.\Scripts\python.exe -m npe2 validate ./napari-mt-interaction/src/napari_mt_interaction/napari.yaml
```

### 3. Verify GUI Initialization
You can quickly check if the plugin's widget can be instantiated:
```powershell
.\Scripts\python.exe -c "import napari; from napari_mt_interaction._widget import MTInteractionWidget; v = napari.Viewer(show=False); w = MTInteractionWidget(v); print('Plugin ready!')"
```

*Note: As of April 2026, the local environment paths have been automatically updated to reflect the new repository location.*

## 🏗️ Project Structure
- `MT_Revamp_Task1.ipynb`: Primary research notebook.
- `mt_interaction_core.py`: Core logic for parsing and interaction math.
- `napari-mt-interaction/`: Source code for the napari plugin.
- `tomogram_analysis/`: Directory for input data (.am and .surf).
- `output/`: Generated CSVs, images, and AmiraMesh files.

## ⚖️ License
This project is licensed under the BSD-3-Clause License.
