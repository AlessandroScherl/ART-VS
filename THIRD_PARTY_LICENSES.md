# Third-Party Components & Licenses

ART-VS itself is released under the **MIT license** (see [`LICENSE`](LICENSE)). It bundles,
adapts, or downloads several third-party components, each governed by its own
license. This file documents them. **Please verify each upstream license before
redistribution** — licenses can change over time.

## Bundled in this repository

| Component | Location | Origin | License |
|---|---|---|---|
| RealSense camera URDF/description | `catkin_ws/ibvs/realsense2_description/` | Intel® RealSense ROS | Apache-2.0 |
| RealSense Gazebo plugin | `catkin_ws/ibvs/realsense_gazebo_plugin/` | Intel® RealSense Gazebo plugin | Apache-2.0 (BSD-3 portions) |
| DINOv2 feature-extraction wrapper | `catkin_ws/ibvs/src/visual_servoing/features/dinov2_legacy_wrapper.py` | Adapted from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 |
| LightTrack wrapper (glue only) | `docker/lighttrack_wrapper.py` | Thin interface to [researchmm/LightTrack](https://github.com/researchmm/LightTrack); the tracker itself is **not bundled** — it is cloned from upstream at Docker build time | per upstream LightTrack |

> The Hollywood-poster evaluation texture (`catkin_ws/ibvs/models/viso/`) reproduces
> the planar benchmark of the ViT-VS setup. Confirm you have the right to redistribute
> this image before publishing; if in doubt, replace it with a freely-licensed poster.

## Downloaded at build/run time (NOT bundled)

| Component | How it is obtained | License (verify upstream) |
|---|---|---|
| **PyTorch** (`torch`, `torchvision`, `torchaudio`) | `pip` (CUDA-specific index) | BSD-3-Clause |
| **DINOv2 / DINOv3** backbones | `torch.hub` / HuggingFace at runtime | Apache-2.0 (Meta AI) |
| **AM-RADIO / C-RADIOv3** backbone | HuggingFace at runtime | NVIDIA license — verify |
| **Ultralytics** (YOLO-World ROI detector) | `pip install ultralytics` | **AGPL-3.0** — note the network-copyleft terms |
| **LangSAM** (language-guided SAM) | `pip install git+https://github.com/luca-medeiros/lang-segment-anything.git` | Apache-2.0 |
| **LightTrack** weights/code | cloned from [researchmm/LightTrack](https://github.com/researchmm/LightTrack) (+ released snapshot weights) | per upstream |
| **MoveIt2 / MoveIt Servo**, **ur_robot_driver**, **pymoveit2**, **realsense2_camera** (real robot only) | ROS2 Jazzy apt packages | BSD-3-Clause / Apache-2.0 — verify per package |
| `transformers`, `timm`, `einops`, `supervision`, etc. | `pip` | Apache-2.0 / MIT |

## Notes

- The simulation benchmark (Hollywood poster) requires **none** of the ROI components
  (Ultralytics / LangSAM / LightTrack). Those are only used by the optional
  language-guided ROI mode for cluttered / category-level real-robot targets.
- **Ultralytics is AGPL-3.0.** If you redistribute or offer a network service built on
  it, review the AGPL obligations.
