"""
Legacy DINOv2 wrapper for exact compatibility with vitvs_v2.py
This uses torch.hub instead of HuggingFace transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import types
import math
import logging

logger = logging.getLogger(__name__)

class DINOv2LegacyExtractor:
    """Legacy DINOv2 extractor matching vitvs_v2.py implementation exactly"""
    
    def __init__(self, model_name='dinov2-small', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Map model names to torch.hub names
        model_map = {
            'dinov2-small': 'dinov2_vits14',
            'dinov2-base': 'dinov2_vitb14',
            'dinov2-large': 'dinov2_vitl14',
            'dinov2-giant': 'dinov2_vitg14'
        }
        
        hub_model_name = model_map.get(model_name, 'dinov2_vits14')
        
        # Load model from torch.hub (same as legacy)
        logger.info(f"Loading DINOv2 from torch.hub: {hub_model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', hub_model_name)
        self.model = self.model.to(self.device).eval()
        
        # Get patch size and stride (should be 14 for vits14)
        self.patch_size = self.model.patch_embed.patch_size
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]
        self.stride = self.model.patch_embed.proj.stride
        if isinstance(self.stride, tuple):
            self.stride = self.stride[0]
            
        logger.info(f"DINOv2 Legacy: patch_size={self.patch_size}, stride={self.stride}")
        
        # Setup preprocessing (same as legacy)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Feature extraction setup
        self._feats = []
        self.hook_handlers = []
        self.num_patches = None
        
        # Config for compatibility
        self.config = {
            'has_cls_token': True,
            'feature_dim': 384 if 'small' in model_name else 768,
            'patch_size': self.patch_size,
            'architecture': 'vit'
        }
    
    def preprocess_pil(self, pil_image):
        """Preprocess PIL image same as legacy"""
        return self.transform(pil_image).unsqueeze(0).to(self.device)
    
    def _get_hook(self, facet):
        """Get hook for extracting features from specific facet"""
        if facet == 'token':
            def _hook(module, input, output):
                self._feats.append(output)
            return _hook
        else:
            # For key/query/value extraction
            if facet == 'query':
                facet_idx = 0
            elif facet == 'key':
                facet_idx = 1
            elif facet == 'value':
                facet_idx = 2
            else:
                raise TypeError(f"{facet} is not a supported facet.")
                
            def _inner_hook(module, input, output):
                input = input[0]
                B, N, C = input.shape
                qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                self._feats.append(qkv[facet_idx])
            return _inner_hook
    
    def _register_hooks(self, layers, facet):
        """Register hooks on transformer blocks"""
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
    
    def _unregister_hooks(self):
        """Unregister all hooks"""
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []
    
    def _extract_features(self, batch, layers=[11], facet='token'):
        """Extract features using hooks (same as legacy)"""
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.num_patches = (H // self.patch_size, W // self.patch_size)
        return self._feats
    
    def _log_bin(self, x, hierarchy=1):
        """Log binning implementation from legacy"""
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy
        
        # Permute and reshape for binning
        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        
        sub_desc_dim = bin_x.shape[1]
        
        avg_pools = []
        for k in range(0, hierarchy):
            win_size = 3 ** k
            avg_pool = nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))
        
        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][:, :, i, j]
                            else:
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][:, :, temp_i, temp_j]
                            part_idx += 1
        
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        return bin_x
    
    def extract_descriptors(self, batch, layer=11, facet='token', bin=False, include_cls=False, **kwargs):
        """Extract descriptors matching legacy implementation exactly"""
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        
        if facet == 'token':
            x.unsqueeze_(dim=1)  # Bx1xtxd
            
        if not include_cls:
            x = x[:, :, 1:, :]  # Remove CLS token
            
        if not bin:
            # Without binning
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
        else:
            # With binning
            desc = self._log_bin(x)
            
        return desc
    
    def get_feature_info(self):
        """Get feature information for compatibility"""
        return {
            'feature_dim': self.config['feature_dim'],
            'patch_size': self.patch_size,
            'num_patches': self.num_patches,
            'has_cls_token': True
        }
    
    def get_patch_size(self):
        """Get patch size"""
        return self.patch_size


def integrate_legacy_dinov2(multi_backbone_extractor):
    """
    Monkey-patch the MultiBackboneExtractor to use legacy DINOv2 for dinov2-small
    
    Usage in multi_backbone_extractor.py:
    
    # In __init__ method, replace the dinov2 initialization with:
    
    if 'dinov2-small' in model_name:
        # Use legacy implementation for exact compatibility
        from dinov2_legacy_wrapper import DINOv2LegacyExtractor
        self.model = None  # We'll use the legacy extractor directly
        self.processor = None
        self.legacy_extractor = DINOv2LegacyExtractor('dinov2-small', device)
        logger.info("Using LEGACY DINOv2 implementation for exact compatibility")
    elif 'dinov2' in model_name:
        # Other DINOv2 models use HuggingFace
        self.model = Dinov2Model.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
    
    # Then in preprocess_pil:
    if hasattr(self, 'legacy_extractor'):
        return self.legacy_extractor.preprocess_pil(pil_image)
    
    # And in extract_descriptors:
    if hasattr(self, 'legacy_extractor'):
        return self.legacy_extractor.extract_descriptors(
            image_tensor, layer=layer, facet=facet, 
            bin=bin, include_cls=include_cls
        )
    """
    pass