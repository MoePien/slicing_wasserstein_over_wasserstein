import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.linalg import sqrtm

def generate_perlin_images_vectorized(N, D, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=0.):
    """
    Generates N random images using Perlin noise with a vectorized,
    pure NumPy implementation for increased speed.

    Args:
        N (int): The number of images to generate.
        D (int): The height and width of each square image.
        scale (float): Determines the 'zoom' level of the noise.
        octaves (int): The number of layers of noise to combine.
        persistence (float): Controls how much each octave contributes.
        lacunarity (float): The frequency multiplier for each octave.

    Returns:
        numpy.ndarray: An array of shape (N, D, D) containing the generated images.
                       Pixel values are floats between 0 and 1.
    """
    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    all_images = np.zeros((N, D, D))
    
    for i in range(N):
        image = np.zeros((D, D))
        amplitude = 1.0
        frequency = 1.0 / scale
        
        # A unique seed for each image to ensure different noise patterns
        unique_seed = int(1e5 * N*D*octaves*(i+1)*scale*octaves*persistence*lacunarity + seed)
        unique_seed = unique_seed % (2**32 -1)
        np.random.seed(unique_seed)
        
        # Generate random gradients for a grid larger than the image
        gradients_x = np.random.uniform(-1, 1, size=(D + 2, D + 2))
        gradients_y = np.random.uniform(-1, 1, size=(D + 2, D + 2))

        # Vectorized coordinate grid
        x, y = np.meshgrid(np.arange(D) / scale, np.arange(D) / scale)
        
        for _ in range(octaves):
            # Calculate grid coordinates and local fractional parts
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            
            x1, y1 = x0 + 1, y0 + 1

            # Get the random gradients for all four corners of each cell
            g00_x, g00_y = gradients_x[x0, y0], gradients_y[x0, y0]
            g10_x, g10_y = gradients_x[x1, y0], gradients_y[x1, y0]
            g01_x, g01_y = gradients_x[x0, y1], gradients_y[x0, y1]
            g11_x, g11_y = gradients_x[x1, y1], gradients_y[x1, y1]

            # Calculate the distance vectors
            dx, dy = x - x0, y - y0
            
            # Calculate the dot products
            n00 = g00_x * dx + g00_y * dy
            n10 = g10_x * (dx - 1) + g10_y * dy
            n01 = g01_x * dx + g01_y * (dy - 1)
            n11 = g11_x * (dx - 1) + g11_y * (dy - 1)

            # Apply fade function for smooth interpolation
            sx = fade(dx)
            sy = fade(dy)

            # Bilinear interpolation
            ix0 = n00 * (1 - sx) + n10 * sx
            ix1 = n01 * (1 - sx) + n11 * sx
            
            # Add to the image, scaling by amplitude
            image += (ix0 * (1 - sy) + ix1 * sy) * amplitude
            
            # Update for next octave
            x *= lacunarity
            y *= lacunarity
            amplitude *= persistence
        
        # Normalize the final image to a [0, 1] range
        image_min, image_max = np.min(image), np.max(image)
        if image_max != image_min:
            all_images[i] = (image - image_min) / (image_max - image_min)
        else:
            all_images[i] = np.zeros_like(image)
    
    return all_images


def extract_patches_non_overlapping(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Extracts all non-overlapping patches from a batch of images.

    Args:
        images (torch.Tensor): A tensor of shape (N, D, D) containing N images.
        patch_size (int): The height and width of the square patches.

    Returns:
        torch.Tensor: A tensor of shape (N, M, P * P) where P is the patch size
                      and M is the total number of patches per image.
    """
    if images.dim() != 3 or images.shape[1] != images.shape[2]:
        raise ValueError("Input tensor must be of shape (N, D, D).")

    N, D, _ = images.shape
    P = patch_size

    if D % P != 0:
        raise ValueError("Image dimensions must be divisible by the patch size.")

    # Reshape images to (N, 1, D, D) to be compatible with unfold
    images_reshaped = images.unsqueeze(1)

    # Use unfold to extract patches as columns
    patches_tensor = torch.nn.functional.unfold(
        images_reshaped, kernel_size=P, stride=P
    )

    # The shape of patches_tensor is now (N, P*P, M), where M is the number of patches
    # Transpose the tensor to get the desired shape (N, M, P*P)
    patches_tensor = patches_tensor.permute(0, 2, 1)

    return patches_tensor

def extract_patches(images: torch.Tensor, patch_size: int, stride: int = 1) -> torch.Tensor:
    """
    Extracts all (possibly overlapping) patches from a batch of images.

    Args:
        images (torch.Tensor): A tensor of shape (N, D, D) containing N images.
        patch_size (int): The height and width of the square patches.
        stride (int): The stride between patches (default=1 gives full overlap coverage).

    Returns:
        torch.Tensor: A tensor of shape (N, M, P * P) where P is the patch size
                      and M is the total number of patches per image.
    """
    if images.dim() != 3 or images.shape[1] != images.shape[2]:
        raise ValueError("Input tensor must be of shape (N, D, D).")

    N, D, _ = images.shape
    P = patch_size

    images_reshaped = images.unsqueeze(1)  # (N,1,D,D)

    patches_tensor = torch.nn.functional.unfold(
        images_reshaped, kernel_size=P, stride=stride
    )  # (N, P*P, M)

    return patches_tensor.permute(0, 2, 1)  # (N, M, P*P)

class PerlinPatches(Dataset):
    """
    Custom PyTorch Dataset for loading Perlin Patches.
    Data is stored internally as a list of PyTorch tensors.
    """
    def __init__(self, P, N, D, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=0, device="cpu"):
        self.device = device
        self.P = P
        self.N = N
        self.D = D
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.images_np = generate_perlin_images_vectorized(N, D, scale=scale, 
                                                        octaves=octaves, persistence=persistence, 
                                                        lacunarity=lacunarity, seed=seed)
        self.images_torch = torch.from_numpy(self.images_np).to(device)
        self.patches = extract_patches(self.images_torch, P).to(device)

        self.classes = [i for i in range(N)]
        self.targets = torch.tensor(self.classes) 
        self.data =  self.patches.to(device)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __visualize__(self, figsize_scale=3, savename=False):
        num_images = self.images_np.shape[0]
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * figsize_scale, figsize_scale))
        for i in range(num_images):
            ax = axes[i]
            ax.imshow(self.images_np[i], cmap='gray')
            #ax.set_title(f'Image {i+1}')
            ax.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if savename:
            plt.savefig(savename)
        plt.show()

def calculate_kid(images1, images2, device, n_subsets=100, subset_size=1000):
    """
    Calculates the Kernel Inception Distance (KID) between two batches of images.

    Args:
        images1 (torch.Tensor): A tensor of shape (N, D, D) containing the first set of grayscale images.
        images2 (torch.Tensor): A tensor of shape (N, D, D) containing the second set of grayscale images.
        device (str): The device to use for calculations ('cpu' or 'cuda').
        n_subsets (int): The number of subsets to use for the KID calculation.
        subset_size (int): The size of each subset.

    Returns:
        float: The calculated KID score.
    """
    
    # --- 1. Feature Extraction using InceptionV3 ---
    def get_inception_features(images_tensor, model):
        # Pre-process images for InceptionV3: 299x299, 3-channel, normalized
        transform = transforms.Compose([
            transforms.Resize(299, antialias=True),
            transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        images_tensor = images_tensor.unsqueeze(1) # Add channel dim: (N, 1, D, D)
        images_preprocessed = transform(images_tensor).to(device)
        
        with torch.no_grad():
            features = model(images_preprocessed)
            # InceptionV3 output is a tuple, we need the first element (the features)
            if isinstance(features, tuple):
                features = features[0]
        return features.cpu().numpy()

    # Load the InceptionV3 model from torch.hub
    try:
        inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.DEFAULT',
                                        verbose=False)
    except AttributeError:
        # Fallback for older PyTorch/Torchvision versions
        inception_model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True,
                                         verbose=False)
    
    # We only need the features, so we can use a linear layer as the last layer
    inception_model.fc = torch.nn.Identity()
    inception_model = inception_model.to(device).eval()

    features1 = get_inception_features(images1, inception_model)
    features2 = get_inception_features(images2, inception_model)

    # --- 2. KID Calculation (using MMD with a polynomial kernel) ---
    def polynomial_kernel(x, y, degree=3, gamma=None, c=1):
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        return (gamma * (x @ y.T) + c) ** degree

    def mmd_squared(features1, features2):
        k_xx = polynomial_kernel(features1, features1).mean()
        k_yy = polynomial_kernel(features2, features2).mean()
        k_xy = polynomial_kernel(features1, features2).mean()
        return k_xx + k_yy - 2 * k_xy

    # Calculate KID by averaging MMD over multiple subsets
    kid_scores = []
    min_size = min(len(features1), len(features2))
    
    if min_size < subset_size:
        # print(f"Warning: Not enough samples for a subset size of {subset_size}. Using {min_size} instead.")
        subset_size = min_size

    for _ in range(n_subsets):
        # Randomly sample subsets without replacement
        subset1_indices = np.random.choice(len(features1), subset_size, replace=False)
        subset2_indices = np.random.choice(len(features2), subset_size, replace=False)

        subset1 = features1[subset1_indices, :]
        subset2 = features2[subset2_indices, :]

        kid_scores.append(mmd_squared(subset1, subset2))

    return np.mean(kid_scores)

def calc_KID_between_subsets(dataset, dataset2):
    mu_im1 = dataset.images_torch
    mu_im2 = dataset2.images_torch
    return calculate_kid(mu_im1.float(), mu_im2.float(), mu_im2.device)