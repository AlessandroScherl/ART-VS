# Docker Environment

This directory contains Docker configurations for building and running the ART-VS simulation and vision environments.

## Available Dockerfiles

### `Dockerfile.simulation`
- **Purpose**: Basic simulation environment without GPU requirements
- **Use Case**: Running Gazebo simulation only, no visual servoing
- **Build Script**: `buildandrun_simulation.sh`
- **Container Name**: `viso_sim`

### `Dockerfile.vision_50XX`
- **Purpose**: Vision environment for NVIDIA RTX 50XX series GPUs (e.g., RTX 5090)
- **CUDA Version**: 12.8
- **PyTorch**: CUDA 12.8 compatible
- **Build Script**: `buildandrun_vision_50XX.sh`
- **Container Name**: `viso_vision_desktop`

### `Dockerfile.vision_40XX`
- **Purpose**: Vision environment for NVIDIA RTX 40XX series GPUs (e.g., RTX 4070 Mobile)
- **CUDA Version**: 11.8
- **PyTorch**: CUDA 11.8 compatible
- **Build Script**: `buildandrun_vision_40XX.sh`
- **Container Name**: `viso_vision_laptop`

## Quick Start

### For RTX 50XX (Desktop)
```bash
cd docker
./buildandrun_vision_50XX.sh
```

### For RTX 40XX (Laptop)
```bash
cd docker
./buildandrun_vision_40XX.sh
```

### For Simulation Only
```bash
cd docker
./buildandrun_simulation.sh
```

## What's Installed

All vision containers include:
- **ROS1 Noetic** (via RoboStack)
- **Gazebo** physics simulator
- **PyTorch** (GPU-compatible version)
- **Vision Libraries**: torchvision, transformers, timm, einops
- **Computer Vision**: opencv-python, scikit-learn, PIL
- **Utilities**: scipy, numpy, matplotlib, tqdm, pandas

## Container Details

### Workspace Mounting
- **Host Path**: `../catkin_ws`
- **Container Path**: `/root/vision_ws/src/`
- Changes made in the container are reflected on the host

### GPU Access
- Uses `--runtime=nvidia` and `--gpus all`
- NVIDIA driver capabilities: `compute,utility,display`
- X11 forwarding enabled for Gazebo GUI

### Activation
Inside the container, activate the ROS environment:
```bash
mamba activate ros_env
```

## Troubleshooting

### Build Fails
- Ensure Docker is installed and running
- Check NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`

### GPU Not Detected
- Verify NVIDIA drivers: `nvidia-smi`
- Check Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`

### X11 Forwarding Issues
- Run `xhost +local:docker` before starting container
- Check `DISPLAY` environment variable is set

## Which image for which GPU?

The two vision images are **identical except for the PyTorch CUDA wheel** (`cu118` vs
`cu128`), but they are **not interchangeable across GPU architectures**:

| GPU | Architecture | Use image | Why |
|---|---|---|---|
| RTX 30XX / 40XX (e.g. 4070, 4090) | Ampere / Ada (sm_86/sm_89) | `vision_40XX` (CUDA 11.8) **or** `vision_50XX` | both `cu118` and `cu128` provide kernels for these |
| RTX 50XX (e.g. 5090) | Blackwell (sm_120) | **`vision_50XX` (CUDA 12.8) only** | Blackwell needs `cu128`; the `cu118` image will fail with *"no kernel image available for execution on the device"* |

In short: a **4090 works with either** image (40XX is sufficient); a **5090 must use the
50XX image**. The 50XX image is effectively a superset and is the safe default on any
recent NVIDIA GPU.

**Host requirements:** an NVIDIA driver new enough for your card (≈ R520+ for Ada, R570+
for Blackwell) and the `nvidia-container-toolkit` (`--gpus all` / `--runtime=nvidia`).

## Notes

- Both images use the Ubuntu 20.04 base; CUDA comes from the pip PyTorch wheels (no system CUDA needed)
- RoboStack provides ROS 1 Noetic without requiring Ubuntu 20.04 natively
- The PyTorch version is intentionally unpinned (latest on the chosen CUDA index); pin it
  if you need bit-exact reproducibility
