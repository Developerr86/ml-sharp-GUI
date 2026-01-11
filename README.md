# Sharp Monocular View Synthesis in Less Than a Second

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy,
Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

![](data/teaser.jpg)

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

## Getting started

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

## Using the GUI (Desktop Application)

SHARP now includes a desktop GUI application built with PyQt6 for interactive 3D Gaussian Splatting visualization.

### Prerequisites (Windows)

The GUI requires additional setup on Windows for CUDA-accelerated rendering:

1. **CUDA Toolkit** (Required for GPU rendering)
   - Download and install [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
   - Match the version to your PyTorch installation (check with `pip show torch`)
   - Verify installation: `nvcc --version`

2. **Visual Studio Build Tools** (Required for compiling CUDA extensions)
   - Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - During installation, select **"Desktop development with C++"**
   - Ensure these components are checked:
     - MSVC v143 (or latest)
     - Windows 10/11 SDK
     - C++ CMake tools for Windows

3. **PyTorch with CUDA**
   ```bash
   # Uninstall CPU version if present
   pip uninstall torch torchvision -y
   
   # Install CUDA version (matching your CUDA Toolkit)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

4. **PyQt6**
   ```bash
   pip install PyQt6
   ```

5. **Reinstall gsplat** (after CUDA Toolkit and Build Tools are installed)
   ```bash
   pip uninstall gsplat -y
   pip install gsplat
   ```

### Running the GUI

**Option 1: Using Developer Command Prompt (Recommended for Windows)**
1. Open **"x64 Native Tools Command Prompt for VS"** from Start Menu
2. Navigate to the project directory
3. Activate your conda environment: `conda activate sharp`
4. Run: `python src/sharp_gui.py`

**Option 2: Regular Command Prompt**
```bash
python src/sharp_gui.py
```

### GUI Features

- **Load Image**: Open JPG/PNG/HEIC images for processing
- **Generate 3D Model**: Run SHARP inference to create 3D Gaussians (downloads model checkpoint automatically on first run)
- **Interactive Viewer**: 
  - Left-click + drag to rotate camera
  - Scroll wheel to zoom in/out
  - Real-time rendering using gsplat
- **Controls**:
  - Camera Radius slider: Adjust viewing distance
  - Splat Scale slider: Control Gaussian splat sizes
- **File Operations**:
  - Save PLY: Export generated 3D Gaussians
  - Load PLY: Import previously saved models

### Troubleshooting

**"CUDA not available" error:**
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Should return `True`. If `False`, reinstall PyTorch with CUDA support.

**"gsplat: No CUDA toolkit found" error:**
- Install CUDA Toolkit matching your PyTorch version
- Reinstall gsplat after CUDA Toolkit installation

**"cannot find 'cl'" error:**
- Install Visual Studio Build Tools with C++ development tools
- Use Developer Command Prompt for VS, or
- Run `vcvars64.bat` before launching Python

**First run is slow:**
- gsplat compiles CUDA kernels on first use (takes a few minutes)
- Subsequent runs will be much faster

## Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

Alternatively, you can download the model directly:

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

To use a manually downloaded checkpoint, specify it with the `-c` flag:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

The results will be 3D gaussian splats (3DGS) in the output folder. The 3DGS `.ply` files are compatible to various public 3DGS renderers. We follow the OpenCV coordinate convention (x right, y down, z forward). The 3DGS scene center is roughly at (0, 0, +z). When dealing with 3rdparty renderers, please scale and rotate to re-center the scene accordingly.

### Rendering trajectories (CUDA GPU only)

Additionally you can render videos with a camera trajectory. While the gaussians prediction works for all CPU, CUDA, and MPS, rendering videos via the `--render` option currently requires a CUDA GPU. The gsplat renderer takes a while to initialize at the first launch.

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# Or from the intermediate gaussians:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

## Evaluation

Please refer to the paper for both quantitative and qualitative evaluations.
Additionally, please check out this [qualitative examples page](https://apple.github.io/ml-sharp/) containing several video comparisons against related work.

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
