# Requirements

This directory contains Python package requirements for the ART-VS project.

## Files

### `requirements.txt`
**Purpose**: Basic simulation dependencies

**Installation**:
```bash
pip install -r requirements.txt
```

**Includes**:
- `opencv-python`: Computer vision library
- `numpy`: Numerical computing
- `scipy`: Scientific computing
- `PyYAML`: YAML file parsing

### `requirements_vision.txt`
**Purpose**: Full vision dependencies including deep learning frameworks

**Installation**:
```bash
pip install -r requirements_vision.txt
```

**Includes**:
All packages from `requirements.txt` plus:
- `torch`: PyTorch deep learning framework
- `torchvision`: PyTorch vision utilities
- `transformers`: HuggingFace transformers
- `timm`: PyTorch Image Models
- `einops`: Tensor operations
- `scikit-learn`: Machine learning utilities
- `Pillow`: Image processing
- `pandas`: Data manipulation
- `ultralytics`: YOLO models
- `supervision`: Computer vision utilities
- `torchmetrics`: Metrics for PyTorch
- `tabulate`: Table formatting

## Installation Guide

### For Simulation Only
If you only need to run Gazebo simulation without visual servoing:
```bash
pip install -r requirements.txt
```

### For Full Visual Servoing (Recommended)
For running visual servoing with Vision Transformers:
```bash
pip install -r requirements_vision.txt
```

### GPU-Specific PyTorch Installation

The vision requirements include PyTorch, but you may need to install the correct version for your GPU:

**For CUDA 11.8** (RTX 40XX series):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.8** (RTX 50XX series):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For CPU only**:
```bash
pip install torch torchvision torchaudio
```

## Docker Containers

If you're using the provided Docker containers, all requirements are pre-installed:

- **Simulation container**: Basic requirements only
- **Vision containers (40XX/50XX)**: Full vision requirements with GPU-optimized PyTorch

## Verification

After installation, verify key packages:

```bash
# Check PyTorch and CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check OpenCV
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## Updating Requirements

When adding new dependencies:

1. Add to appropriate requirements file:
   - Basic packages → `requirements.txt`
   - Vision/DL packages → `requirements_vision.txt`

2. Document why the package is needed

3. Pin versions for reproducibility (e.g., `numpy==1.24.0`)

## Common Issues

### CUDA Version Mismatch
**Symptom**: PyTorch doesn't detect GPU
**Solution**: Reinstall PyTorch with correct CUDA version (see GPU-Specific Installation above)

### Package Conflicts
**Symptom**: Installation fails with dependency conflicts
**Solution**: Use virtual environment or conda environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_vision.txt
```

### Missing ROS Packages
**Symptom**: `rospkg` or `catkin-pkg` not found
**Solution**: These are installed separately via RoboStack in Docker containers:
```bash
mamba install rospkg catkin-pkg -c conda-forge
```

## Notes

- Requirements files do not include ROS packages (handled by RoboStack in Docker)
- PyTorch version in Docker containers is optimized for specific GPU architectures
- All package versions are tested and compatible with Ubuntu 20.04
