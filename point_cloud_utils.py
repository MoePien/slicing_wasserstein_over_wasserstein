import ot
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
import glob
from sliced_hierarchical_OT import * 
from functional_swd import * 

class ModelNet10(Dataset):
    """
    Custom PyTorch Dataset for loading ModelNet10 point clouds from OFF files.
    Data is stored internally as a list of PyTorch tensors.
    """
    def __init__(self, root_dir, selected_class=None, max_shapes=None, n_points=50, std=0., split="train", seed=42, verbose=False):
        self.root_dir = root_dir
        self.n_points = n_points
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.data = []  # Store as a list of PyTorch tensors
        self.targets = []  # Store as a list of PyTorch tensors
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if verbose:
            print(f"Loading ModelNet10 dataset (Class: {selected_class}, Max Shapes: {max_shapes})...")

        target_classes = [selected_class] if selected_class else self.classes

        for class_name in target_classes:
            if class_name not in self.class_to_idx:
                print(f"Warning: Class '{class_name}' not found in dataset. Skipping.")
                continue

            class_path = os.path.join(self.root_dir, class_name, split)
            off_files = sorted(glob.glob(os.path.join(class_path, "*.off")))
            random.shuffle(off_files)
            if max_shapes is not None:
                off_files = off_files[:max_shapes]

            for filename in off_files:
                seed = int(abs(hash(filename))/1e11)
                np.random.seed(seed)
                torch.manual_seed(seed)
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    if "OFF" not in lines[0].strip():
                        continue

                    header_line = lines[1].strip().split()
                    n_vertices = int(header_line[0])

                    vertices = np.array([list(map(float, line.strip().split())) for line in lines[2:2+n_vertices]])

                    if vertices.size > 0:
                        vertices = vertices - np.mean(vertices, axis=0)
                        max_val = np.max(np.abs(vertices))
                        if max_val > 1e-6:
                            vertices = vertices / max_val
                        else:
                            vertices = vertices * 0
                    else:
                        vertices = np.empty((0, 3))

                    rng = np.random.default_rng(seed=seed)
                    if vertices.shape[0] < self.n_points:
                        sample_indices = rng.choice(vertices.shape[0], self.n_points, replace=True)
                    else:
                        sample_indices = rng.choice(vertices.shape[0], self.n_points, replace=False)
                    sampled_points = vertices[sample_indices, :]
                    # Convert to PyTorch tensors and append
                    sampled_points = torch.from_numpy(sampled_points).float()
                    sampled_points += std * torch.randn_like(sampled_points)
                    self.data.append(100 * sampled_points.unsqueeze(0))
                    self.targets.append(torch.tensor(self.class_to_idx[class_name]))
        
        if not self.data:
            raise ValueError(f"No shapes loaded for class '{selected_class}' with max_shapes={max_shapes}. Check path and parameters.")
        self.classes = [i for i, _ in enumerate(off_files)]
        self.targets = torch.tensor(self.classes) 
        self.data = torch.concat(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def one_nna_via_W2(mu_0, mu_1):
    """
    Computes 1-Nearest Neighbor Accuracy (1-NNA) for Wasserstein.
    mu_0: list of M tensors, where each tensor is of shape (N, D)
    mu_1: list of M tensors, where each tensor is of shape (N, D)
    
    The cost matrix for the 1-NNA is C_ij = W_2^2(mu_0^i, mu_1^j).
    
    Returns: 1-NNA via Wasserstein
    """
    
    N = len(mu_0) + len(mu_1)
    cost_matrix = torch.zeros(N, N, device=mu_0[0].device)
    mu_all_ls = mu_0 + mu_1
    label_ls = [0 for mu_sub in mu_0] + [1 for mu_sub in mu_1]
    for i in range(N):
        for j in range(i +1, N):
            w2 = compute_wasserstein_distance(mu_all_ls[i], mu_all_ls[j])
            cost_matrix[i, j] = w2
            cost_matrix[j, i] = w2
    labels = torch.tensor(label_ls).to(cost_matrix.device)

    nna = one_nna(cost_matrix, labels)
    return nna
    
def one_nna(distance_matrix: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute 1-Nearest Neighbor Accuracy (1-NNA).

    Args:
        distance_matrix (torch.Tensor): Pairwise distance matrix of shape (N, N).
        labels (torch.Tensor): Tensor of shape (N,) with binary labels {0,1},
                               0 = real, 1 = fake.

    Returns:
        float: 1-NNA score in [0,1].
    """
    N = distance_matrix.shape[0]
    assert distance_matrix.shape == (N, N), "distance_matrix must be square"
    assert labels.shape[0] == N, "labels must match number of samples"

    # Mask out diagonal (distance to self)

    masking = (1 + torch.eye(N, device=distance_matrix.device) * distance_matrix.max())
    masked_dist = distance_matrix + masking

    # Nearest neighbor indices
    nn_indices = masked_dist.argmin(dim=1)

    # Predicted label = label of nearest neighbor
    pred_labels = labels[nn_indices]

    # Correct classification = same label as neighbor
    correct = (pred_labels == labels).sum()

    return correct / N


def calc_OT_NNA_between_subsets(sub1, sub2, dim=3, length_scale=10., outer_pnum=100, grid_n=10, inner_pnum=100):
    data1_ls = []
    
    for label in sub1.classes:
        data1_ls.append(sub1.data[sub1.targets == label].reshape(-1, dim).float())
    
    data2_ls = []
    
    for label in sub2.classes:
        data2_ls.append(sub2.data[sub2.targets == label].reshape(-1, dim).float())
    return one_nna_via_W2(data1_ls, data2_ls)