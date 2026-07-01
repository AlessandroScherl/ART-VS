import torch
from torchvision.transforms import Compose, RandomErasing, ColorJitter
import cv2
import numpy as np
import argparse
import os

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class TorchAugmentor(object):
    def __init__(self, composer):
        self.composer = composer

    def __call__(self, I):
        Itorch = torch.from_numpy(I).float().clone() / 255
        Itorch = Itorch.permute(2, 0, 1).unsqueeze(0)
        Itorch = self.composer(Itorch[0])
        Ip = Itorch.permute(1, 2, 0).unsqueeze(0) * 255
        return Ip.cpu().numpy()

def main(input_image, num_samples):
    # Set the random seed
    seed = 489
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define the augmentation pipeline
    augmentation = Compose([
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ColorJitter(brightness=0.6, contrast=0.4),
        AddGaussianNoise(0.0, 0.05),
    ])

    # Create the augmentor
    augmentor = TorchAugmentor(augmentation)

    # Load the input image
    image = cv2.imread(input_image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Unable to load image {input_image}")
        return

    # Ensure the image is in RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate and save perturbed images
    base_name = os.path.splitext(input_image)[0]
    for i in range(num_samples):
        perturbed_image = augmentor(image_rgb)[0]
        output_filename = f"{base_name}_p_{i+1}.png"
        cv2.imwrite(output_filename, cv2.cvtColor(perturbed_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"Saved {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate perturbed images from an input image.")
    parser.add_argument("input_image", help="Path to the input image file")
    parser.add_argument("-num_samples", type=int, default=10, help="Number of perturbed samples to generate")
    args = parser.parse_args()

    main(args.input_image, args.num_samples)
