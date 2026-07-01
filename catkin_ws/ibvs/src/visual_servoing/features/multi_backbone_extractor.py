import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from transformers import (
    AutoModel, AutoImageProcessor, AutoProcessor,
    Dinov2Model, ViTMAEModel, SiglipModel, 
    ConvNextV2Model, SwinModel
)
import timm
import logging
from typing import Dict, Tuple, Optional
from features.base_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class MultiBackboneExtractor:
    """
    Unified feature extractor for multiple vision backbones.
    Handles different architectures and extracts spatial features suitable for correspondence matching.
    """
    
    SUPPORTED_MODELS = {
        # Self-supervised models (good for correspondence)
        'dinov2-base': 'facebook/dinov2-base',
        'dinov2-small': 'facebook/dinov2-small',
        'dinov2-large': 'facebook/dinov2-large',
        
        # DINOv3 models (next-generation self-supervised with register tokens)
        'dinov3-small': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'dinov3-base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'dinov3-large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'mae-base': 'facebook/vit-mae-base',
        'mae-large': 'facebook/vit-mae-large',
        # I-JEPA models (self-supervised without masking - excellent for correspondence)
        'ijepa-vit-b': 'ijepa-vit-b',  # I-JEPA ViT-B/16 
        'ijepa-vit-l': 'ijepa-vit-l',  # I-JEPA ViT-L/16
        'ijepa-vit-h': 'ijepa-vit-h',  # I-JEPA ViT-H/14
        
        # Vision-language models (challenging for correspondence)
        'eva-clip-8b': 'BAAI/EVA-CLIP-8B',
        'siglip-base': 'google/siglip-base-patch16-224',
        'siglip-large': 'google/siglip-large-patch16-256',
        'siglip-so400m': 'google/siglip-so400m-patch14-384',  # SigLIP v2 400M params
        'siglip-base-patch16-384': 'google/siglip-base-patch16-384',  # Higher res SigLIP
        'siglip-base-patch16-512': 'google/siglip-base-patch16-512',  # Even higher res
        'clip-base': 'openai/clip-vit-base-patch32',
        'clip-large': 'openai/clip-vit-large-patch14',
        
        # Supervised models
        'convnext-v2-base': 'facebook/convnextv2-base-22k-224',
        'swin-base': 'microsoft/swin-base-patch4-window7-224',
        
        # Vision-language models (advanced)
        'owlvit-base': 'google/owlvit-base-patch32',
        'owlvit-large': 'google/owlvit-large-patch14',
        
        # Efficient models (AM-RADIO variants)
        'am-radio': 'am-radio',  # NVlabs/RADIO (auto-select)
        'am-radio-base': 'am-radio-base',  # C-RADIOv3-B (ViT-B/16)
        'am-radio-large': 'am-radio-large',  # C-RADIOv3-L (ViT-L/16)
        
        # Classical computer vision methods
        'classical-sift': 'classical-sift',  # SIFT (Scale-Invariant Feature Transform)
        'classical-orb': 'classical-orb',    # ORB (Oriented FAST and Rotated BRIEF)
        'classical-akaze': 'classical-akaze', # AKAZE (Accelerated-KAZE)
    }
    
    def __init__(self, model_name: str = 'dinov2-base', device: str = 'cuda', input_size: int = 224):
        """
        Initialize the feature extractor with specified backbone.
        
        Args:
            model_name: Name of the model from SUPPORTED_MODELS
            device: Device to run the model on
            input_size: Input image size (for DINOv3 torch.hub models)
        """
        self.device = device
        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS.get(model_name)
        self.input_size = input_size
        
        print(f"[INIT DEBUG] MultiBackboneExtractor.__init__ called")
        print(f"[INIT DEBUG] model_name: {model_name}")
        print(f"[INIT DEBUG] input_size: {input_size}")
        print(f"[INIT DEBUG] model_id: {self.model_id}")
        
        if 'classical-' in model_name:
            self._init_classical()
        elif 'am-radio' in model_name:
            self._init_am_radio()
        elif 'owlvit' in model_name:
            self._init_owlvit()
        elif 'ijepa' in model_name:
            self._init_ijepa()
        else:
            print(f"[INIT DEBUG] Calling _init_transformers_model() for {model_name}")
            self._init_transformers_model()
            
        # Model-specific configurations
        print(f"[INIT DEBUG] Calling _setup_model_config() for {model_name}")
        self._setup_model_config()
        
    def _init_am_radio(self):
        """Initialize AM-RADIO model from NVlabs/RADIO."""
        try:
            # Map model names to specific variants
            variant_map = {
                'am-radio-base': "c-radio_v3-b",   # C-RADIOv3-B model (ViT-B/16)
                'am-radio-large': "c-radio_v3-l",  # C-RADIOv3-L model (ViT-L/16)
                'am-radio': "c-radio_v3-b"         # Default to base
            }
            
            # Get the specific variant to load
            variant = variant_map.get(self.model_name, "c-radio_v3-b")
            
            # Try to load the specific variant
            try:
                self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', 
                                          version=variant, progress=True, 
                                          skip_validation=True)
                self.model = self.model.to(self.device).eval()
                self.radio_version = variant
                print(f"Successfully loaded AM-RADIO variant: {variant}")
            except Exception as e:
                # If specific variant fails, try fallback variants
                print(f"Failed to load {variant}: {e}")
                fallback_variants = ["c-radio_v3-b", "c-radio_v3-l", "c-radio_v3-h", "e-radio_v2"]
                model_loaded = False
                
                for fallback in fallback_variants:
                    if fallback == variant:  # Skip already tried variant
                        continue
                    try:
                        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', 
                                                  version=fallback, progress=True, 
                                                  skip_validation=True)
                        self.model = self.model.to(self.device).eval()
                        self.radio_version = fallback
                        print(f"Successfully loaded AM-RADIO fallback variant: {fallback}")
                        model_loaded = True
                        break
                    except Exception as fallback_error:
                        print(f"Failed to load fallback {fallback}: {fallback_error}")
                        continue
                
                if not model_loaded:
                    raise RuntimeError("Could not load any AM-RADIO variant")
            
            # Create AM-RADIO specific processor
            self.processor = self._create_am_radio_processor()
            
        except Exception as e:
            print(f"Error loading AM-RADIO: {e}")
            raise
    
    def _init_owlvit(self):
        """Initialize OWL-ViT model."""
        try:
            from transformers import OwlViTModel, OwlViTProcessor
            self.model = OwlViTModel.from_pretrained(self.model_id).to(self.device).eval()
            self.processor = OwlViTProcessor.from_pretrained(self.model_id)
            print(f"Successfully loaded OWL-ViT: {self.model_name}")
        except Exception as e:
            print(f"Error loading OWL-ViT: {e}")
            raise
    
    def _init_ijepa(self):
        """Initialize I-JEPA model."""
        try:
            # I-JEPA requires special loading from torch hub
            print(f"Loading I-JEPA model: {self.model_name}")
            
            # Map model names to I-JEPA variants
            model_variant_map = {
                'ijepa-vit-b': 'vit_base',
                'ijepa-vit-l': 'vit_large', 
                'ijepa-vit-h': 'vit_huge'
            }
            
            variant = model_variant_map.get(self.model_name, 'vit_base')
            
            # Try to load from torch hub
            try:
                self.model = torch.hub.load('facebookresearch/ijepa', variant, pretrained=True)
                self.model = self.model.to(self.device).eval()
                print(f"Successfully loaded I-JEPA from torch hub: {variant}")
            except:
                # Fallback: I-JEPA might need manual loading
                print("I-JEPA not available via torch hub, trying manual initialization...")
                print("WARNING: Loading I-JEPA architecture without pretrained weights!")
                print("This will likely result in poor performance for visual servoing.")
                
                # Create a ViT model with I-JEPA architecture
                import timm
                if 'vit-b' in self.model_name or 'base' in self.model_name:
                    # Try to load with ImageNet pretrained weights as a better starting point
                    try:
                        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
                        print("Created ViT-B/16 with ImageNet pretrained weights (not I-JEPA weights)")
                    except:
                        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
                        print("Created ViT-B/16 without pretrained weights")
                elif 'vit-l' in self.model_name or 'large' in self.model_name:
                    try:
                        self.model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=0)
                        print("Created ViT-L/16 with ImageNet pretrained weights (not I-JEPA weights)")
                    except:
                        self.model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0)
                        print("Created ViT-L/16 without pretrained weights")
                elif 'vit-h' in self.model_name or 'huge' in self.model_name:
                    try:
                        self.model = timm.create_model('vit_huge_patch14_224', pretrained=True, num_classes=0)
                        print("Created ViT-H/14 with ImageNet pretrained weights (not I-JEPA weights)")
                    except:
                        self.model = timm.create_model('vit_huge_patch14_224', pretrained=False, num_classes=0)
                        print("Created ViT-H/14 without pretrained weights")
                else:
                    # Default to base
                    self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
                    print(f"Created default ViT-B/16 for I-JEPA model: {self.model_name}")
                    
                self.model = self.model.to(self.device).eval()
                print(f"Model embed_dim: {getattr(self.model, 'embed_dim', 'unknown')}")
            
            # Create processor (standard ViT preprocessing)
            from transformers import AutoImageProcessor
            try:
                self.processor = AutoImageProcessor.from_pretrained('facebook/vit-base-patch16-224')
            except:
                # Fallback to manual preprocessing
                from torchvision import transforms
                self.processor = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print("Using manual preprocessing for I-JEPA")
                
        except Exception as e:
            print(f"Error loading I-JEPA: {e}")
            raise
    
    def _init_classical(self):
        """Initialize classical feature detector (SIFT, ORB, AKAZE)."""
        try:
            from features.classical_extractor import ClassicalFeatureExtractor
            
            # Extract method name (e.g., 'sift' from 'classical-sift')
            method = self.model_name.split('-')[1]
            
            # Create classical extractor instance
            self.classical_extractor = ClassicalFeatureExtractor(method=method, device=self.device)
            
            # For compatibility, set model and processor to None
            self.model = None
            self.processor = None
            
            print(f"Successfully loaded classical {method.upper()} detector")
            
        except Exception as e:
            print(f"Error loading classical detector: {e}")
            raise
    
    def _init_transformers_model(self):
        """Initialize model from transformers library."""
        try:
            # Load model based on architecture
            if 'dinov3' in self.model_name:
                # DINOv3 with configurable input size using HuggingFace model + custom processor
                print(f"Loading DINOv3 model: {self.model_id}")
                try:
                    # Load HuggingFace model (try local cache first to avoid slow hub checks)
                    try:
                        self.model = AutoModel.from_pretrained(self.model_id, local_files_only=True).to(self.device).eval()
                        base_processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=True)
                    except OSError:
                        # Not cached yet — download from hub
                        print(f"  Downloading {self.model_id} from HuggingFace Hub (first time only)...")
                        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
                        base_processor = AutoImageProcessor.from_pretrained(self.model_id)

                    # Create custom processor with configurable input size
                    
                    # Override the size with our configured input_size
                    base_processor.size = {"height": self.input_size, "width": self.input_size}
                    base_processor.crop_size = {"height": self.input_size, "width": self.input_size}
                    self.processor = base_processor
                    
                    print(f"✓ DINOv3 loaded with HuggingFace model + custom processor")
                    # print(f"[DINOv3 DEBUG] Configured input_size: {self.input_size}")
                    # print(f"[DINOv3 DEBUG] Processor size: {self.processor.size}")
                    # print(f"[DINOv3 DEBUG] Expected patches: {(self.input_size // 16) * (self.input_size // 16)}")
                except Exception as e:
                    print(f"HuggingFace DINOv3 failed: {e}")
                    print("Falling back to torch.hub loading...")
                    # Map model names for torch.hub
                    hub_model_map = {
                        'dinov3-small': 'dinov3_vits16',
                        'dinov3-base': 'dinov3_vitb16',
                        'dinov3-large': 'dinov3_vitl16',
                    }
                    hub_model_name = hub_model_map.get(self.model_name, 'dinov3_vits16')
                    
                    try:
                        # Load using torch.hub
                        self.model = torch.hub.load('facebookresearch/dinov3', hub_model_name).to(self.device).eval()

                        # Create a compatible processor using torchvision transforms
                        from torchvision import transforms
                        self.processor = None  # Will use custom transform

                        # Use configurable input size
                        # Standard approach: resize to input_size + margin, then center crop to exact input_size
                        resize_size = self.input_size + 32  # Add margin for center crop
                        self.transform = transforms.Compose([
                            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(self.input_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        print(f"✓ DINOv3 loaded with torch.hub fallback")
                    except Exception as hub_error:
                        # Check if it's an HTTP 403 error
                        import urllib.error
                        error_msg = str(hub_error)
                        if "403" in error_msg or "Forbidden" in error_msg or isinstance(hub_error, urllib.error.HTTPError):
                            raise RuntimeError(
                                "\n" + "="*80 + "\n"
                                "ERROR: DINOv3 model weights are currently unavailable (HTTP 403 Forbidden).\n"
                                "Facebook/Meta appears to have restricted access to the model weights.\n\n"
                                "The weights were hosted at:\n"
                                "  https://dl.fbaipublicfiles.com/dinov3/\n\n"
                                "This is a known issue as of Feb 2025. Possible solutions:\n"
                                "1. Skip DINOv3 tests temporarily (comment out in test script)\n"
                                "2. Use DINOv2 models instead which are still available\n"
                                "3. Check if you have cached weights from previous runs\n"
                                "4. Contact Meta AI research for alternative download links\n"
                                + "="*80
                            ) from hub_error
                        else:
                            raise RuntimeError(f"Error loading DINOv3 via torch.hub: {hub_error}") from hub_error
                    # print(f"[DINOv3 DEBUG] Configured input_size: {self.input_size}")
                    # print(f"[DINOv3 DEBUG] Transform resize_size: {resize_size}")
                    # print(f"[DINOv3 DEBUG] Expected patches: {(self.input_size // 16) * (self.input_size // 16)}")
            elif 'dinov2' in self.model_name:
                # Use modern HuggingFace implementation for all DINOv2 models
                print(f"Loading DINOv2 model with HuggingFace: {self.model_id}")
                self.model = Dinov2Model.from_pretrained(self.model_id).to(self.device).eval()
                base_processor = AutoImageProcessor.from_pretrained(self.model_id)
                
                # Override processor size with configured input_size (same as DINOv3)
                base_processor.size = {"height": self.input_size, "width": self.input_size}
                base_processor.crop_size = {"height": self.input_size, "width": self.input_size}
                self.processor = base_processor
                
                print(f"✓ DINOv2 loaded with HuggingFace model + custom processor")
                print(f"[DINOv2 DEBUG] Configured input_size: {self.input_size}")
                print(f"[DINOv2 DEBUG] Expected patches: {(self.input_size // 14) * (self.input_size // 14)}")
            elif 'mae' in self.model_name:
                self.model = ViTMAEModel.from_pretrained(self.model_id).to(self.device).eval()
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            elif 'siglip' in self.model_name:
                self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_id)
            elif 'clip' in self.model_name and 'eva' not in self.model_name:
                # Standard OpenAI CLIP
                from transformers import CLIPModel, CLIPProcessor
                self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
                self.processor = CLIPProcessor.from_pretrained(self.model_id)
            elif 'eva-clip' in self.model_name:
                # EVA-CLIP might need special handling
                self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            elif 'convnext' in self.model_name:
                self.model = ConvNextV2Model.from_pretrained(self.model_id).to(self.device).eval()
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            elif 'swin' in self.model_name:
                self.model = SwinModel.from_pretrained(self.model_id).to(self.device).eval()
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            else:
                # Fallback to AutoModel
                self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            raise
    
    def _create_basic_processor(self, image_size: int):
        """Create a basic image processor for models without dedicated processors."""
        class BasicProcessor:
            def __init__(self, size):
                self.size = size
                self.transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            def __call__(self, images, return_tensors="pt"):
                if isinstance(images, Image.Image):
                    images = [images]
                processed = torch.stack([self.transform(img) for img in images])
                return {'pixel_values': processed}
        
        return BasicProcessor(image_size)
    
    def _create_am_radio_processor(self):
        """Create AM-RADIO specific processor."""
        class AMRadioProcessor:
            def __init__(self):
                pass
            
            def __call__(self, images, return_tensors="pt"):
                if isinstance(images, Image.Image):
                    images = [images]
                
                processed_tensors = []
                for img in images:
                    # Convert PIL to tensor and normalize to [0, 1]
                    x = transforms.functional.pil_to_tensor(img).to(dtype=torch.float32)
                    x = x.div_(255.0)  # AM-RADIO expects values between 0 and 1
                    processed_tensors.append(x)
                
                processed = torch.stack(processed_tensors)
                return {'pixel_values': processed}
        
        return AMRadioProcessor()
    
    def _setup_model_config(self):
        """Setup model-specific configurations."""
        self.config = {
            'has_cls_token': True,
            'feature_dim': 768,  # Default, will be updated
            'patch_size': 16,
            'architecture': 'vit'  # vit, cnn, hybrid
        }
        
        # Model-specific adjustments
        if 'classical-' in self.model_name:
            # Classical feature detectors configuration
            self.config.update({
                'has_cls_token': False,
                'feature_dim': self.classical_extractor.get_descriptor_dim(),
                'patch_size': 1,  # Classical methods don't have patches
                'architecture': 'classical'
            })
        elif 'am-radio' in self.model_name:
            # AM-RADIO model configurations
            feature_dim_map = {
                "c-radio_v3-b": 768,   # ViT-B/16
                "c-radio_v3-l": 1024,  # ViT-L/16  
                "c-radio_v3-h": 1280,  # ViT-H/16
                "e-radio_v2": 1024     # E-RADIO
            }
            radio_version = getattr(self, 'radio_version', 'c-radio_v3-b')
            self.config.update({
                'has_cls_token': False,  # AM-RADIO doesn't use CLS token in spatial features
                'feature_dim': feature_dim_map.get(radio_version, 768),
                'patch_size': 16,  # Most AM-RADIO variants use 16
                'architecture': 'vit'
            })
        elif 'owlvit' in self.model_name:
            self.config.update({
                'has_cls_token': True,
                'feature_dim': self.model.config.vision_config.hidden_size,
                'patch_size': self.model.config.vision_config.patch_size,
                'architecture': 'vit'
            })
        elif 'ijepa' in self.model_name:
            # I-JEPA configuration
            feature_dims = {
                'ijepa-vit-b': 768,   # ViT-B/16
                'ijepa-vit-l': 1024,  # ViT-L/16
                'ijepa-vit-h': 1280   # ViT-H/14
            }
            patch_sizes = {
                'ijepa-vit-b': 16,
                'ijepa-vit-l': 16,
                'ijepa-vit-h': 14
            }
            self.config.update({
                'has_cls_token': True,
                'feature_dim': feature_dims.get(self.model_name, 768),
                'patch_size': patch_sizes.get(self.model_name, 16),
                'architecture': 'vit'
            })
            # Store patch size for later use
            self.patch_size = self.config['patch_size']
        elif 'dinov3' in self.model_name:
            # DINOv3 has CLS token + 4 register tokens + patch tokens
            if hasattr(self.model, 'config'):
                # Loaded via AutoModel
                hidden_size = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 384
            else:
                # Loaded via torch.hub - infer from model architecture
                hidden_size_map = {
                    'dinov3-small': 384,
                    'dinov3-base': 768,
                    'dinov3-large': 1024,
                }
                hidden_size = hidden_size_map.get(self.model_name, 384)
            
            self.config.update({
                'has_cls_token': True,
                'has_register_tokens': True,  # DINOv3 specific
                'num_register_tokens': 4,     # Hardcoded for all DINOv3 models
                'feature_dim': hidden_size,
                'patch_size': 16,  # All DINOv3 models use patch size 16
                'architecture': 'vit'
            })
        elif 'dinov2' in self.model_name:
            if hasattr(self, 'legacy_extractor'):
                # Use legacy extractor configuration
                self.config.update(self.legacy_extractor.config)
            else:
                self.config.update({
                    'has_cls_token': True,
                    'feature_dim': self.model.config.hidden_size,
                    'architecture': 'vit'
                })
        elif 'mae' in self.model_name:
            self.config.update({
                'has_cls_token': True,
                'feature_dim': self.model.config.hidden_size,
                'architecture': 'vit'
            })
        elif 'siglip' in self.model_name:
            self.config.update({
                'has_cls_token': False,  # SigLIP doesn't use CLS token
                'feature_dim': self.model.config.vision_config.hidden_size if hasattr(self.model.config, 'vision_config') else 768,
                'architecture': 'vit'
            })
        elif 'clip' in self.model_name and 'eva' not in self.model_name:
            # Standard CLIP configuration
            self.config.update({
                'has_cls_token': True,
                'feature_dim': self.model.config.vision_config.hidden_size,
                'patch_size': self.model.config.vision_config.patch_size,
                'architecture': 'vit'
            })
        elif 'convnext' in self.model_name:
            self.config.update({
                'has_cls_token': False,
                'feature_dim': self.model.config.hidden_sizes[-1],
                'architecture': 'cnn'
            })
        elif 'swin' in self.model_name:
            self.config.update({
                'has_cls_token': False,
                'feature_dim': self.model.config.hidden_size,
                'architecture': 'hybrid'
            })
    
    def preprocess_pil(self, pil_image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for the model."""
        # Check for legacy DINOv2 extractor
        if hasattr(self, 'legacy_extractor'):
            return self.legacy_extractor.preprocess_pil(pil_image)
            
        # Check if processor is None (classical methods)
        if self.processor is None:
            # Classical methods shouldn't use this, but return a dummy tensor to avoid crashes
            # The rotation finding should be skipped for classical methods anyway
            logger.warning(f"preprocess_pil called on classical method {self.model_name} - returning dummy tensor")
            # Return a dummy tensor
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            return dummy
            
        if 'dinov3' in self.model_name and self.processor is None:
            # DINOv3 loaded via torch.hub - use custom transform
            if hasattr(self, 'transform'):
                # print(f"[DINOv3 DEBUG] Input PIL image size: {pil_image.size}")
                processed = self.transform(pil_image)
                # print(f"[DINOv3 DEBUG] After transform shape: {processed.shape}")
                if len(processed.shape) == 3:  # [C, H, W]
                    processed = processed.unsqueeze(0)  # [1, C, H, W]
                # print(f"[DINOv3 DEBUG] Final preprocessed shape: {processed.shape}")
                return processed.to(self.device)
            else:
                # Fallback - this shouldn't happen
                logger.warning("DINOv3 torch.hub model missing transform")
                dummy = torch.zeros(1, 3, 224, 224).to(self.device)
                return dummy
        elif 'ijepa' in self.model_name and hasattr(self.processor, '__call__') and not hasattr(self.processor, 'preprocess'):
            # Handle manual transforms for I-JEPA
            # The transforms expect PIL image, not tensor
            processed = self.processor(pil_image)
            # If the result is already a tensor, just add batch dimension
            if isinstance(processed, torch.Tensor):
                if len(processed.shape) == 3:  # [C, H, W]
                    processed = processed.unsqueeze(0)  # [1, C, H, W]
                return processed.to(self.device)
            else:
                # Should not reach here with proper transforms
                return torch.tensor(processed).unsqueeze(0).to(self.device)
        else:
            # Standard preprocessing
            inputs = self.processor(images=pil_image, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
    
    def extract_descriptors(self, image_tensor: torch.Tensor, 
                          bin: bool = False, 
                          hierarchy: int = 1) -> torch.Tensor:
        """
        Extract spatial descriptors from the image.
        
        Args:
            image_tensor: Preprocessed image tensor
            bin: Whether to apply feature binning
            hierarchy: Hierarchy level for extraction
            
        Returns:
            Descriptors tensor of shape [B, 1, num_patches, feature_dim]
        """
        
        # Legacy DINOv2 extractor no longer used - all models use HuggingFace
        # (Keeping check for backward compatibility, but should never trigger)
        if hasattr(self, 'legacy_extractor'):
            print("[WARNING] Legacy extractor found but should not be used")
            return self.legacy_extractor.extract_descriptors(
                image_tensor, layer=11, facet='token', 
                bin=bin, include_cls=False
            )
            
        # Check for classical methods
        if self.config.get('architecture') == 'classical':
            # Return dummy features for classical methods
            logger.warning("extract_descriptors called on classical method - returning dummy features")
            B = image_tensor.shape[0] if image_tensor is not None else 1
            features = torch.randn(B, 1, 256, 128).to(self.device)
            return F.normalize(features, p=2, dim=-1)
        
        with torch.no_grad():
            features = self._extract_features(image_tensor)
            
            # Handle different feature shapes
            if self.config['architecture'] == 'vit':
                # ViT-style: [B, num_patches, feature_dim]
                # Handle special tokens (CLS and/or register tokens)
                if self.config.get('has_register_tokens', False):
                    # DINOv3: Skip CLS (index 0) and register tokens (indices 1-4)
                    num_register_tokens = self.config.get('num_register_tokens', 4)
                    # if 'dinov3' in self.model_name:
                    #     print(f"[DINOv3] Raw features shape: {features.shape}")
                    #     print(f"[DINOv3] Removing CLS + {num_register_tokens} register tokens")
                    features = features[:, 1 + num_register_tokens:, :]  # Skip CLS + registers
                    # if 'dinov3' in self.model_name:
                    #     print(f"[DINOv3] Patch features shape: {features.shape}")
                elif self.config.get('has_cls_token', False):
                    # DINOv2 and others: Skip only CLS token
                    features = features[:, 1:, :]  # Remove CLS token
            elif self.config['architecture'] == 'cnn':
                # CNN-style: [B, C, H, W] -> [B, H*W, C]
                B, C, H, W = features.shape
                features = features.flatten(2).transpose(1, 2)
            elif self.config['architecture'] == 'hybrid':
                # Swin-style: might already be in the right format
                if len(features.shape) == 4:
                    B, H, W, C = features.shape
                    features = features.reshape(B, H*W, C)
            
            # Store spatial dimensions for binning
            B, num_patches, feature_dim = features.shape
            spatial_size = int(num_patches ** 0.5)  # Assume square patch grid
            self.num_patches = (spatial_size, spatial_size)
            
            # Debug output for DINOv3
            if 'dinov3' in self.model_name:
                # print(f"[DINOv3 DEBUG] Features extracted: B={B}, num_patches={num_patches}, feature_dim={feature_dim}")
                # print(f"[DINOv3 DEBUG] Spatial size: {spatial_size}x{spatial_size}")
                # print(f"[DINOv3 DEBUG] Expected vs actual patches: expected={(self.input_size // 16)**2}, actual={num_patches}")
                pass
            
            # Debug info for I-JEPA
            if 'ijepa' in self.model_name:
                print(f"[I-JEPA Debug] Model name: {self.model_name}")
                print(f"[I-JEPA Debug] Features shape: {features.shape}")
                print(f"[I-JEPA Debug] Spatial size: {spatial_size}, num_patches: {self.num_patches}")
                print(f"[I-JEPA Debug] Expected patches for input size {getattr(self, 'last_input_size', 'unknown')}")
                print(f"[I-JEPA Debug] Config feature_dim: {self.config['feature_dim']}")
                print(f"[I-JEPA Debug] Config patch_size: {self.config['patch_size']}")
                if hasattr(self.model, 'embed_dim'):
                    print(f"[I-JEPA Debug] Model embed_dim: {self.model.embed_dim}")
                if hasattr(self.model, 'patch_embed'):
                    print(f"[I-JEPA Debug] Model patch_size: {getattr(self.model.patch_embed, 'patch_size', 'unknown')}")
            
            # Apply feature binning if requested
            if bin:
                # For I-JEPA, we need to handle potentially non-square patch grids
                if 'ijepa' in self.model_name and num_patches != spatial_size * spatial_size:
                    # I-JEPA might have irregular patch counts, try to find closest square
                    import math
                    # Try common patch grid sizes
                    for grid_size in [14, 15, 16, 24, 28, 32]:
                        if grid_size * grid_size == num_patches:
                            spatial_size = grid_size
                            self.num_patches = (spatial_size, spatial_size)
                            break
                        elif grid_size * grid_size == num_patches + 1:  # Account for removed CLS token
                            spatial_size = grid_size
                            self.num_patches = (spatial_size, spatial_size)
                            break
                    else:
                        # If no perfect square found, use closest
                        logger.warning(f"I-JEPA: Non-square patch count {num_patches}, using approximate grid {spatial_size}x{spatial_size}")

                # Force float32 BEFORE binning to avoid BFloat16 issues in _log_bin
                features = features.float()

                # Reshape for binning: [B, 1, num_patches, feature_dim]
                features_for_bin = features.unsqueeze(1)
                features = self._log_bin(features_for_bin, hierarchy)
            else:
                # Add dimension for compatibility: [B, 1, num_patches, feature_dim]
                features = features.unsqueeze(1)
            
            # Force float32 for RTX 5090/Blackwell GPU compatibility
            # F.normalize() doesn't support BFloat16 which may be the default dtype on newer GPUs
            features = features.float()
            # Apply L2 normalization for better correspondence matching
            features = F.normalize(features, p=2, dim=-1)
            
        return features
    
    def _extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features based on model architecture.
        
        Returns:
            Feature tensor (shape depends on architecture)
        """
        if 'am-radio' in self.model_name:
            # AM-RADIO from NVlabs/RADIO
            import torch.nn.functional as F
            
            # Get nearest supported resolution
            nearest_res = self.model.get_nearest_supported_resolution(*image_tensor.shape[-2:])
            x = F.interpolate(image_tensor, nearest_res, mode='bilinear', align_corners=False)
            
            # Set optimal window size for E-RADIO variants
            if hasattr(self, 'radio_version') and "e-radio" in self.radio_version:
                self.model.model.set_optimal_window_size(x.shape[2:])
            
            # AM-RADIO returns (summary, spatial_features)
            _, spatial_features = self.model(x, feature_fmt='NCHW')  # Get NCHW format
            
            # Convert NCHW to sequence format [B, H*W, C]
            B, C, H, W = spatial_features.shape
            features = spatial_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            return features
        
        elif 'owlvit' in self.model_name:
            # OWL-ViT vision encoder
            outputs = self.model.vision_model(image_tensor, output_hidden_states=True)
            return outputs.last_hidden_state
        
        elif 'ijepa' in self.model_name:
            # I-JEPA feature extraction
            # Store input size for debugging
            self.last_input_size = image_tensor.shape[-2:]
            
            if hasattr(self.model, 'forward_features'):
                # If using timm model
                features = self.model.forward_features(image_tensor)
                
                # Debug: Check feature dimensions
                if features.shape[-1] != self.config['feature_dim']:
                    print(f"[I-JEPA Warning] Expected feature dim {self.config['feature_dim']}, got {features.shape[-1]}")
                    print(f"[I-JEPA Warning] Model: {self.model_name}, Features shape: {features.shape}")
                
                # Remove CLS token if present
                if self.config.get('has_cls_token', True) and features.shape[1] > 1:
                    # Check if first token looks like CLS (should have different statistics)
                    if features.shape[1] == 256:  # 16x16 grid
                        features = features[:, 1:, :]  # Remove CLS token -> 255 patches
                    elif features.shape[1] == 196:  # 14x14 grid
                        features = features[:, 1:, :]  # Remove CLS token -> 195 patches
                return features
            else:
                # Standard forward pass
                outputs = self.model(image_tensor)
                if hasattr(outputs, 'last_hidden_state'):
                    return outputs.last_hidden_state
                else:
                    return outputs
        
        elif 'dinov3' in self.model_name:
            # DINOv3 forward pass
            if self.processor is None:
                # Using torch.hub model - returns tensor directly
                return self.model(image_tensor)
            else:
                # Using AutoModel - returns BaseModelOutputWithPooling
                outputs = self.model(image_tensor)
                if hasattr(outputs, 'last_hidden_state'):
                    return outputs.last_hidden_state
                else:
                    # Fallback for raw tensor output
                    return outputs
        
        elif 'dinov2' in self.model_name:
            outputs = self.model(image_tensor, output_hidden_states=True)
            return outputs.last_hidden_state
        
        elif 'mae' in self.model_name:
            outputs = self.model(image_tensor, output_hidden_states=True)
            return outputs.last_hidden_state
        
        elif 'siglip' in self.model_name:
            # SigLIP might have vision_model attribute
            if hasattr(self.model, 'vision_model'):
                outputs = self.model.vision_model(image_tensor, output_hidden_states=True)
            else:
                outputs = self.model(image_tensor, output_hidden_states=True)
            return outputs.last_hidden_state
        
        elif 'clip' in self.model_name and 'eva' not in self.model_name:
            # Standard CLIP handling
            outputs = self.model.vision_model(image_tensor, output_hidden_states=True)
            return outputs.last_hidden_state
            
        elif 'eva-clip' in self.model_name:
            # EVA-CLIP handling
            if hasattr(self.model, 'encode_image'):
                # Some CLIP models have this method
                features = self.model.encode_image(image_tensor, return_all_features=True)
                return features
            else:
                outputs = self.model.vision_model(image_tensor, output_hidden_states=True)
                return outputs.last_hidden_state
        
        elif 'convnext' in self.model_name:
            outputs = self.model(image_tensor, output_hidden_states=True)
            # ConvNeXt returns features in [B, C, H, W] format
            return outputs.last_hidden_state
        
        elif 'swin' in self.model_name:
            outputs = self.model(image_tensor, output_hidden_states=True)
            return outputs.last_hidden_state
        
        else:
            # Generic fallback
            outputs = self.model(image_tensor, output_hidden_states=True)
            return outputs.last_hidden_state
    
    def _log_bin(self, x: torch.Tensor, hierarchy: int = 1) -> torch.Tensor:
        """
        Create a log-binned descriptor for AM-RADIO and other models.
        Adapted from DINOv2 implementation for multi-backbone use.
        
        Args:
            x: tensor of features. Has shape [B, 1, num_patches, feature_dim].
            hierarchy: how many bin hierarchies to use.
            
        Returns:
            Binned features of shape [B, 1, num_patches, binned_feature_dim]
        """
        
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy
        # One-time print so we can confirm the hierarchy actually in use at runtime
        # (printed once to avoid per-iteration I/O that can cause Servo velocity gaps).
        if not getattr(self, '_logged_hierarchy', False):
            print(f"[BINNING] _log_bin active: hierarchy={hierarchy}, num_bins={num_bins}, "
                  f"input_feature_dim={x.shape[-1]} -> binned_dim={x.shape[-1] * num_bins}")
            self._logged_hierarchy = True

        # Remove the singleton dimension and rearrange: [B, num_patches, feature_dim] -> [B, feature_dim, H, W]
        bin_x = x.squeeze(1)  # [B, num_patches, feature_dim]
        bin_x = bin_x.permute(0, 2, 1)  # [B, feature_dim, num_patches]
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Now: [B, feature_dim, H, W]
        
        sub_desc_dim = bin_x.shape[1]  # feature_dim
        
        avg_pools = []
        # Compute bins of all sizes for all spatial locations
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**k x 3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))
        
        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        
        for y in range(self.num_patches[0]):
            for x_coord in range(self.num_patches[1]):
                part_idx = 0
                # Fill all bins for a spatial location (y, x_coord)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x_coord - kernel_size, x_coord + kernel_size + 1, kernel_size):
                            if i == y and j == x_coord and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x_coord] = avg_pools[k][:, :, i, j]
                            else:  # Handle padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x_coord] = avg_pools[k][:, :, temp_i, temp_j]
                            part_idx += 1
        
        # Reshape back to sequence format: [B, 1, num_patches, binned_feature_dim]
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        return bin_x
    
    def get_feature_info(self) -> Dict:
        """Get information about the extracted features."""
        return {
            'model_name': self.model_name,
            'feature_dim': self.config['feature_dim'],
            'has_cls_token': self.config['has_cls_token'],
            'architecture': self.config['architecture']
        }


class MultiBackboneViTExtractor(BaseFeatureExtractor):
    """
    Wrapper for MultiBackboneExtractor that conforms to BaseFeatureExtractor interface.
    This allows seamless integration with the existing visual servoing system.
    """
    
    def __init__(self, model_name: str = 'am-radio', device: str = 'cuda', input_size: int = 224):
        print(f"[WRAPPER DEBUG] MultiBackboneViTExtractor.__init__ called")
        print(f"[WRAPPER DEBUG] model_name: {model_name}")
        print(f"[WRAPPER DEBUG] input_size: {input_size}")
        super().__init__(device)
        self.extractor = MultiBackboneExtractor(model_name, device, input_size)
        self.model_name = model_name
        
    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for the model."""
        return self.extractor.preprocess_pil(image)
    
    def extract_descriptors(self, image_tensor: torch.Tensor, 
                          layer: int = 11, facet: str = 'key',
                          bin: bool = False, include_cls: bool = False,
                          hierarchy: int = 1, **kwargs) -> torch.Tensor:
        """
        Extract feature descriptors from an image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor
            layer: Layer index (kept for compatibility, not used)
            facet: Facet type (kept for compatibility, not used) 
            bin: Whether to apply binning
            include_cls: Whether to include CLS token (handled internally)
            hierarchy: Hierarchy level for binning
            **kwargs: Additional arguments
            
        Returns:
            Feature descriptors tensor
        """
        return self.extractor.extract_descriptors(image_tensor, bin=bin, hierarchy=hierarchy)
    
    def get_descriptor_dim(self) -> int:
        """Get the dimension of the feature descriptors."""
        return self.extractor.config['feature_dim']
    
    def get_patch_size(self) -> int:
        """Get the patch size used by the model."""
        # Check for legacy DINOv2 extractor
        if hasattr(self.extractor, 'legacy_extractor'):
            return self.extractor.legacy_extractor.get_patch_size()
        # For I-JEPA, try to get from the actual model
        if 'ijepa' in self.extractor.model_name:
            if hasattr(self.extractor.model, 'patch_embed') and hasattr(self.extractor.model.patch_embed, 'patch_size'):
                patch_size = self.extractor.model.patch_embed.patch_size
                if isinstance(patch_size, tuple):
                    return patch_size[0]
                else:
                    return patch_size
        return self.extractor.config.get('patch_size', 16)
    
    def get_supported_models(self) -> Dict[str, str]:
        """Get dictionary of supported models."""
        return MultiBackboneExtractor.SUPPORTED_MODELS
    
    def get_feature_info(self) -> Dict:
        """Get detailed feature information."""
        return self.extractor.get_feature_info()
