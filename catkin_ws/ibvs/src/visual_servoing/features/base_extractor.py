from abc import ABC, abstractmethod
import torch
from PIL import Image


class BaseFeatureExtractor(ABC):
    """Abstract base class for visual feature extractors."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for the model."""
        pass
    
    @abstractmethod
    def extract_descriptors(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract feature descriptors from an image tensor."""
        pass
    
    @abstractmethod
    def get_descriptor_dim(self) -> int:
        """Get the dimension of the feature descriptors."""
        pass
    
    @abstractmethod
    def get_patch_size(self) -> int:
        """Get the patch size used by the model."""
        pass
