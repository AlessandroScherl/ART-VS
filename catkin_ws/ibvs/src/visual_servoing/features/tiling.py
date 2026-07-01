import math
import numpy as np
import torch
from PIL import Image


class TilingConfig:
    """Configuration for the tiling approach"""
    def __init__(self, slice_number=4, input_resolution=896, vit_input_size=224):
        """
        Args:
            slice_number: Total number of tiles (must be a perfect square: 4, 9, 16, 25, etc.)
            input_resolution: Resolution to resize the full image to
            vit_input_size: Input size for ViT models (224, 308, 518, etc.)
        """
        # Validate slice_number is a perfect square
        grid_size = int(math.sqrt(slice_number))
        if grid_size * grid_size != slice_number:
            raise ValueError(f"slice_number must be a perfect square (4, 9, 16, 25...), got {slice_number}")
        
        # Validate input_resolution is divisible by grid_size
        if input_resolution % grid_size != 0:
            raise ValueError(f"input_resolution ({input_resolution}) must be divisible by sqrt(slice_number) ({grid_size})")
        
        self.slice_number = slice_number
        self.grid_size = grid_size
        self.input_resolution = input_resolution
        self.slice_resolution = input_resolution // grid_size
        self.vit_input_size = vit_input_size
        self.patch_size = 14  # DINOv2 patch size
        
        # Calculate effective resolutions
        self.patches_per_vit_input = vit_input_size // self.patch_size
        self.effective_patches = grid_size * self.patches_per_vit_input


class TilingHandler:
    """Handles image tiling operations for feature extraction."""
    
    def __init__(self, tiling_config):
        self.config = tiling_config
    
    def create_tiles(self, image):
        """
        Create tiles from an image based on the tiling configuration.
        
        Args:
            image: PIL Image
        
        Returns:
            tiles: List of PIL Images
            tile_coords: List of (y, x) coordinates for each tile's top-left corner
            image_resized: The resized full image
        """
        # Resize image to target resolution
        image_resized = image.resize((self.config.input_resolution, self.config.input_resolution))
        
        # Convert to numpy for easier slicing
        img_array = np.array(image_resized)
        
        tiles = []
        tile_coords = []
        
        # Create grid
        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                y_start = i * self.config.slice_resolution
                x_start = j * self.config.slice_resolution
                y_end = y_start + self.config.slice_resolution
                x_end = x_start + self.config.slice_resolution
                
                tile = img_array[y_start:y_end, x_start:x_end]
                tiles.append(Image.fromarray(tile))
                tile_coords.append((y_start, x_start))
        
        return tiles, tile_coords, image_resized
    
    def transform_patch_to_global(self, patch_coord, tile_idx, tile_coords):
        """
        Transform patch coordinate within a tile to global image coordinate.
        
        Args:
            patch_coord: (y, x) coordinate in patch space within tile
            tile_idx: Index of the tile
            tile_coords: List of tile positions in the full image
        
        Returns:
            (global_y, global_x): Coordinates in the full image
        """
        tile_y, tile_x = tile_coords[tile_idx]
        
        # Scale factor from DINOv2 input back to original tile size
        scale = self.config.slice_resolution / self.config.vit_input_size
        
        # Convert patch coord to pixel coord within DINOv2 input
        local_pixel_y = patch_coord[0] * self.config.patch_size + self.config.patch_size // 2
        local_pixel_x = patch_coord[1] * self.config.patch_size + self.config.patch_size // 2
        
        # Scale up to original tile size
        scaled_pixel_y = local_pixel_y * scale
        scaled_pixel_x = local_pixel_x * scale
        
        # Convert to global coordinates
        global_y = tile_y + scaled_pixel_y
        global_x = tile_x + scaled_pixel_x
        
        return torch.tensor([global_y, global_x], device=patch_coord.device)
    
    def calculate_uv_tiled(self, goal_features, current_features, scale_to_original):
        """Calculate feature points from tiled approach and scale them to the real image resolution."""
        num_pairs = len(goal_features)
        
        s_uv_star = np.zeros([num_pairs, 2], dtype=int)
        s_uv = np.zeros([num_pairs, 2], dtype=int)
        
        for count in range(num_pairs):
            # Features are already in tiling resolution coordinates
            # Scale to original image resolution
            s_uv_star[count, 0] = round(goal_features[count][1] * scale_to_original[0])
            s_uv_star[count, 1] = round(goal_features[count][0] * scale_to_original[1])
            s_uv[count, 0] = round(current_features[count][1] * scale_to_original[0])
            s_uv[count, 1] = round(current_features[count][0] * scale_to_original[1])
        
        return s_uv_star, s_uv
