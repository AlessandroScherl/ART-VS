#!/usr/bin/env python3
"""
Generate perturbed versions of the Hollywood poster model (viso) for robustness testing.
This script creates 500 variations with visual perturbations:
- Random erasing (simulating occlusions)
- Color jittering (brightness/contrast variations)
- Gaussian noise
"""

import os
import shutil
import xml.etree.ElementTree as ET
import torch
from torchvision.transforms import Compose, RandomErasing, ColorJitter
import cv2
import numpy as np
import argparse

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

class TorchAugmentor(object):
    def __init__(self, composer):
        self.composer = composer

    def __call__(self, I):
        Itorch = torch.from_numpy(I).float().clone() / 255
        Itorch = Itorch.permute(2, 0, 1).unsqueeze(0)
        Itorch = self.composer(Itorch[0])
        Ip = Itorch.permute(1, 2, 0).unsqueeze(0) * 255
        return Ip.cpu().numpy()

def create_perturbed_model(source_model_path, dest_model_path, perturbed_image, model_number):
    """Create a perturbed model with modified texture."""
    # Copy the entire model folder
    shutil.copytree(source_model_path, dest_model_path, dirs_exist_ok=True)

    # Replace resized.png in materials/textures and meshes
    for subfolder in ['materials/textures', 'meshes']:
        image_path = os.path.join(dest_model_path, subfolder, 'resized.png')
        if os.path.exists(os.path.dirname(image_path)):
            cv2.imwrite(image_path, cv2.cvtColor(perturbed_image.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Update model.sdf
    sdf_path = os.path.join(dest_model_path, 'model.sdf')
    if os.path.exists(sdf_path):
        tree = ET.parse(sdf_path)
        root = tree.getroot()

        # Update the mesh URI and model name in model.sdf
        mesh_uri = root.find(".//uri")
        if mesh_uri is not None:
            new_uri = f"model://viso{model_number}/meshes/resized.dae"
            mesh_uri.text = new_uri

        model_elem = root.find("model")
        if model_elem is not None:
            model_elem.set('name', f'resized{model_number}')

        tree.write(sdf_path)

    # Update model.config
    config_path = os.path.join(dest_model_path, 'model.config')
    if os.path.exists(config_path):
        tree = ET.parse(config_path)
        root = tree.getroot()

        # Update the model name in model.config
        name_elem = root.find('name')
        if name_elem is not None:
            name_elem.text = f'viso{model_number}'

        tree.write(config_path)

def main():
    parser = argparse.ArgumentParser(description='Generate perturbed Hollywood poster models')
    parser.add_argument('--num-models', type=int, default=500, 
                       help='Number of perturbed models to generate (default: 500)')
    parser.add_argument('--seed', type=int, default=489, 
                       help='Random seed for reproducibility (default: 489)')
    args = parser.parse_args()

    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Define the augmentation pipeline
    augmentation = Compose([
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ColorJitter(brightness=0.6, contrast=0.4),
        AddGaussianNoise(0.0, 0.05),
    ])

    # Create the augmentor
    augmentor = TorchAugmentor(augmentation)

    # Set paths - relative to current working directory
    source_model_path = 'models/viso'
    original_image_path = os.path.join(source_model_path, 'materials', 'textures', 'resized.png')

    # Check if source model exists
    if not os.path.exists(source_model_path):
        print(f"Error: Source model not found at {source_model_path}")
        print("Make sure you're running this script from the catkin_ws/ibvs directory")
        return

    # Load the original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Error: Unable to load image {original_image_path}")
        return

    print(f"Loaded original Hollywood poster image from {original_image_path}")
    print(f"Image shape: {original_image.shape}")

    # Ensure the image is in RGB format
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Generate perturbed models
    print(f"\nGenerating {args.num_models} perturbed models...")
    for i in range(1, args.num_models + 1):
        dest_model_path = f'models/viso{i}'

        # Generate perturbed image
        perturbed_image = augmentor(image_rgb)[0]

        # Create perturbed model
        create_perturbed_model(source_model_path, dest_model_path, perturbed_image, i)

        if i % 10 == 0:
            print(f"Created {i}/{args.num_models} perturbed models...")

    print(f"\n✅ Successfully created {args.num_models} perturbed Hollywood models!")
    print("Models are saved in: models/viso1 through models/viso" + str(args.num_models))

if __name__ == "__main__":
    main()
