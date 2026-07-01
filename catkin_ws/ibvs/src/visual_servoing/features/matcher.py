import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)


def create_patch_mask(mask_path, vit_input_size=224, patch_size=14):
    """
    Create a patch-based mask from a pixel-based mask image.

    Args:
        mask_path (str): Path to the binary mask image (white = include, black = exclude)
        vit_input_size (int): Size to resize image to before patch processing
        patch_size (int): Size of each patch

    Returns:
        torch.Tensor: Binary mask tensor of shape (1, num_patches*num_patches) where
                     True indicates patches to include and False indicates patches to exclude
    """
    # Load mask image
    mask_img = Image.open(mask_path).convert('L')
    
    # Resize to DINO input size
    mask_img = mask_img.resize((vit_input_size, vit_input_size), Image.LANCZOS)
    
    # Convert to numpy array and binarize
    mask_array = np.array(mask_img) > 128
    
    # Calculate number of patches
    num_patches = vit_input_size // patch_size
    if num_patches == 0:
        logger.warning(f"Invalid patch configuration: vit_input_size={vit_input_size}, patch_size={patch_size}")
        return torch.ones((1, 1), dtype=torch.bool)  # Return minimal valid mask
        
    patch_mask = np.zeros((num_patches, num_patches), dtype=bool)
    
    # Process each patch
    for i in range(num_patches):
        for j in range(num_patches):
            patch = mask_array[
                    i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size
                    ]
            patch_mask[i, j] = np.mean(patch) > 0.5
    
    return torch.from_numpy(patch_mask.reshape(1, -1))


def create_patch_mask_from_pil(mask_pil, vit_input_size=224, patch_size=14):
    """
    Create a patch-based mask from a PIL mask image (no file I/O).

    Args:
        mask_pil: PIL Image object of the mask
        vit_input_size (int): Size to resize image to before patch processing
        patch_size (int): Size of each patch

    Returns:
        torch.Tensor: Binary mask tensor of shape (1, num_patches*num_patches)
    """
    # Resize to VIT input size
    mask_img = mask_pil.resize((vit_input_size, vit_input_size), Image.LANCZOS)
    
    # Convert to numpy array and binarize
    mask_array = np.array(mask_img) > 128
    
    # Calculate number of patches
    num_patches = vit_input_size // patch_size
    if num_patches == 0:
        logger.warning(f"Invalid patch configuration: vit_input_size={vit_input_size}, patch_size={patch_size}")
        return torch.ones((1, 1), dtype=torch.bool)  # Return minimal valid mask
        
    patch_mask = np.zeros((num_patches, num_patches), dtype=bool)
    
    # Process each patch
    for i in range(num_patches):
        for j in range(num_patches):
            patch = mask_array[
                    i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size
                    ]
            patch_mask[i, j] = np.mean(patch) > 0.5
    
    return torch.from_numpy(patch_mask.reshape(1, -1))


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between all possible pairs in two sets of vectors."""
    # Force float32 to avoid BFloat16 issues on RTX 5090/Blackwell GPUs
    # F.normalize() doesn't support BFloat16, which may be the default dtype on newer GPUs
    # ALSO: torch.einsum() may output bfloat16 due to global autocast on RTX 5090
    x = x.float()
    y = y.float()
    # Normalize along the feature dimension (last dimension)
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)

    # Compute all pairwise dot products
    # Assuming shapes: x: [B, 1, t_x, d], y: [B, 1, t_y, d]
    # Force float32 output - einsum may return bfloat16 on RTX 5090 due to autocast
    result = torch.einsum('bixd,biyd->bixy', x_norm, y_norm)
    return result.float()


def _to_cartesian(coords, shape):
    """Takes raveled coordinates and returns them in a cartesian coordinate frame"""
    if torch.is_tensor(coords):
        coords = coords.long()

    # Calculate rows and columns for all indices
    width = shape[1]
    rows = coords // width
    cols = coords % width

    # Stack coordinates as [x, y] format (cols, rows)
    result = torch.stack([cols, rows], dim=-1)
    return result


def find_correspondences_batch(descriptors1, descriptors2, num_pairs=18, distance_threshold=1, 
                               mask_path=None, mask_tensor=None, vit_input_size=224, patch_size=14):
    """Find correspondences between two images using their descriptors.
    
    Args:
        mask_path: Path to mask file (for backward compatibility)
        mask_tensor: Pre-computed mask tensor to avoid file I/O
    """
    # Profile correspondence finding
    import time
    corresp_start = time.perf_counter()
    
    B, _, t_m_1, d_h = descriptors1.size()
    num_patches = (int(np.sqrt(t_m_1)), int(np.sqrt(t_m_1)))
    
    # Use provided mask tensor if available, otherwise load from path
    if mask_tensor is not None:
        patch_mask = mask_tensor.to(descriptors1.device)
        if patch_mask.shape[1] != t_m_1:
            # In tiled mode, each tile might have different patch counts
            # Just use no mask for tiles instead of warning
            logger.debug(f"Mask tensor size mismatch: {patch_mask.shape[1]} vs {t_m_1} - disabling mask for this tile")
            patch_mask = torch.ones((1, t_m_1), dtype=torch.bool, device=descriptors1.device)
    elif mask_path and os.path.exists(mask_path):
        # Calculate actual patch size from descriptor dimensions
        actual_num_patches = int(np.sqrt(t_m_1))
        if actual_num_patches == 0:
            logger.warning(f"Invalid number of patches: {actual_num_patches} (from t_m_1={t_m_1})")
            patch_mask = torch.ones((1, t_m_1), dtype=torch.bool, device=descriptors1.device)
        else:
            actual_patch_size = vit_input_size // actual_num_patches
            
            patch_mask = create_patch_mask(
                mask_path,
                vit_input_size=vit_input_size,
                patch_size=actual_patch_size
            ).to(descriptors1.device)
        
        # Verify mask size matches descriptor size
        if patch_mask.shape[1] != t_m_1:
            logger.warning(f"Mask size mismatch: {patch_mask.shape[1]} vs {t_m_1}")
            # Fallback to no mask if size mismatch
            patch_mask = torch.ones((1, t_m_1), dtype=torch.bool, device=descriptors1.device)
        else:
            valid_patches = patch_mask.sum().item()
    else:
        if mask_path:
            logger.warning(f"Mask path provided but file not found: {mask_path}")
        patch_mask = torch.ones((1, t_m_1), dtype=torch.bool, device=descriptors1.device)

    # Calculate similarities
    sim_start = time.perf_counter()
    similarities = chunk_cosine_sim(descriptors1, descriptors2)
    sim_time = (time.perf_counter() - sim_start) * 1000

    # Find nearest neighbors
    nn_start = time.perf_counter()
    sim_1, nn_1 = torch.max(similarities, dim=-1)
    sim_2, nn_2 = torch.max(similarities, dim=-2)
    nn_time = (time.perf_counter() - nn_start) * 1000

    # Always use the standard matching logic (removed same-image detection)
    nn_1, nn_2 = nn_1[:, 0, :], nn_2[:, 0, :]
    cyclical_idxs = torch.gather(nn_2, dim=-1, index=nn_1)

    # Create image indices
    image_idxs = torch.arange(t_m_1, device=descriptors1.device)[None, :].repeat(B, 1)

    # Convert to cartesian coordinates
    coord_start = time.perf_counter()
    cyclical_idxs_ij = _to_cartesian(cyclical_idxs, shape=num_patches)
    image_idxs_ij = _to_cartesian(image_idxs, shape=num_patches)
    coord_time = (time.perf_counter() - coord_start) * 1000

    # Calculate distances using vectorized operations
    dist_start = time.perf_counter()
    # More efficient distance calculation - ensure float type for norm
    diff = (cyclical_idxs_ij - image_idxs_ij).float()
    cyclical_dists = -torch.norm(diff, p=2, dim=-1)
    dist_time = (time.perf_counter() - dist_start) * 1000

    # Normalize distances
    cyclical_dists_norm = cyclical_dists - cyclical_dists.min(1, keepdim=True)[0]
    max_dist = cyclical_dists_norm.max(1, keepdim=True)[0]
    # Avoid division by zero when all distances are the same
    if (max_dist < 1e-6).any():
        # When all distances are the same, set all to 1.0 (maximum normalized value)
        cyclical_dists_norm = torch.ones_like(cyclical_dists_norm)
    else:
        cyclical_dists_norm /= (max_dist + 1e-8)  # Add small epsilon

    # Sort values and get selected points
    sorted_vals, selected_points_image_1 = cyclical_dists_norm.sort(dim=-1, descending=True)

    # Apply distance threshold (with small epsilon for floating point comparison)
    # When threshold is 1.0, we want the maximum values, but due to floating point
    # the max might be 0.9999999 instead of exactly 1.0
    effective_threshold = distance_threshold - 1e-6 if distance_threshold >= 0.99 else distance_threshold
    threshold_mask = sorted_vals >= effective_threshold

    # Apply patch mask - only keep points that are in valid patches
    mask_values = patch_mask[0][selected_points_image_1]
    combined_mask = threshold_mask & mask_values

    # Apply combined mask to get final filtered points
    filtered_points = selected_points_image_1[combined_mask]

    # Select points
    num_available = filtered_points.numel()
    num_to_select = min(num_pairs, num_available)

    # Debug logging when no points available (only log if this is unexpected)
    if num_available == 0 and num_pairs > 0:
        # This can happen when mask excludes all good correspondences
        logger.debug(f"[MATCHER] No valid correspondences found after filtering")

    if num_to_select > 0:
        perm = torch.randperm(num_available, device=descriptors1.device)
        selected_indices = perm[:num_to_select]
        selected_points_image_1 = filtered_points[selected_indices].unsqueeze(0)

        # Get corresponding points in image 2
        selected_points_image_2 = torch.gather(nn_1, dim=-1, index=selected_points_image_1)

        # Get similarity scores
        sim_selected_12 = torch.gather(sim_1[:, 0, :], dim=-1, index=selected_points_image_1)

        # Convert to coordinates
        points1 = _to_cartesian(selected_points_image_1[0], num_patches)
        points2 = _to_cartesian(selected_points_image_2[0], num_patches)

        total_time = (time.perf_counter() - corresp_start) * 1000
        if total_time > 5:  # Only log if taking more than 5ms
            logger.debug(f"[CORRESP] Total: {total_time:.1f}ms (sim: {sim_time:.1f}, nn: {nn_time:.1f}, coord: {coord_time:.1f}, dist: {dist_time:.1f})")
        return points1, points2, sim_selected_12[0]
    else:
        return None, None, None


def scale_points_from_patch(points, vit_image_size=518, num_patches=37):
    """Scale points from patch coordinates to pixel coordinates"""
    points = (points + 0.5) / num_patches * vit_image_size
    return points
