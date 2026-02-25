import torch
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
import ot  # Import the POT library for optimal transport
import numpy as np
import copy
from scipy.spatial.distance import cdist, pdist
from scipy.interpolate import interp1d
from geomloss import SamplesLoss
from functional_swd import *




# --- 2. Function to compute a single pairwise Wasserstein distance ---
def compute_wasserstein_distance(x, y, ot_func=None, blur=.01):
    """
    Computes the squared Wasserstein-2 distance between two point clouds.
    x: torch.Tensor of shape (N, D)
    y: torch.Tensor of shape (N, D)
    Returns: scalar tensor (W_2^2)
    """
    # The SamplesLoss returns the squared distance W_p^p.
    if ot_func is None:
        ot_func = SamplesLoss("sinkhorn", p=2, blur=blur, debias=False)
    return ot_func(x, y)


# --- 3. Function to compute the hierarchical Wasserstein distance ---
def compute_hierarchical_wasserstein(mu_0, mu_1, blur=.01):
    """
    Computes the hierarchical Wasserstein distance (Wasserstein-over-Wasserstein).
    mu_0: list of M tensors, where each tensor is of shape (N, D)
    mu_1: list of M tensors, where each tensor is of shape (N, D)
    
    The cost matrix for the outer problem is C_ij = W_2^2(mu_0^i, mu_1^j).
    
    Returns: scalar tensor (hierarchical W_2^2)
    """
    # Create the cost matrix C where C_ij = W_2^2(mu_0^i, mu_1^j)
    # The shape will be (M, M)
    M = len(mu_0)
    N = len(mu_1)
    cost_matrix = torch.zeros(M, N, device=mu_0[0].device)
    ot_func = SamplesLoss("sinkhorn", p=2, blur=blur, debias=False)
    for i in range(M):
        for j in range(N):
            cost_matrix[i, j] = compute_wasserstein_distance(mu_0[i], mu_1[j], ot_func)
            
    # Solve the outer optimal transport problem using POT with the pre-computed cost matrix.
    # We assume uniform distributions over the M point clouds, so we use a simple
    # vector of ones normalized to sum to 1.
    a = torch.ones(M, dtype=torch.float32) / M
    b = torch.ones(N, dtype=torch.float32) / N
    
    # We use a regularization parameter (reg) similar to the blur in geomloss.
    # pot.sinkhorn2 returns the regularized Wasserstein distance squared.
    hierarchical_distance = ot.emd2(
        a.cpu().numpy(),
        b.cpu().numpy(),
        cost_matrix.cpu().numpy(),
        # reg=0.001  # A small regularization value for the outer problem
    )
    
    return torch.tensor(hierarchical_distance)

# --- 3. Function to compute the hierarchical Wasserstein distance ---
def compute_hierarchical_sliced_wasserstein(mu_0, mu_1, outer_pnum=100, grid_n=10, inner_pnum=100, length_scale=1.):
    """
    Computes a sliced hierarchical Wasserstein distance (Wasserstein-over-Wasserstein).
    mu_0: list of M tensors, where each tensor is of shape (N, D)
    mu_1: list of M tensors, where each tensor is of shape (N, D)
    
    Returns: scalar tensor (hierarchical sliced W_2^2)
    """
    sliced_dist = 0.
    
    for piter in range(outer_pnum):
        theta = torch.randn(mu_0[0].shape[-1]).to(mu_0[0].device)
        mu_0_proj, grid = project_point_clouds_1d(mu_0, theta, grid_n=grid_n)
        mu_1_proj, grid = project_point_clouds_1d(mu_1, theta, grid_n=grid_n)
        sliced_dist += calc_functional_SWD_from_discrete_data(mu_0_proj, mu_1_proj, grid=grid, 
                                                       n_projections=inner_pnum, length_scale=length_scale)
    return sliced_dist/outer_pnum
    
# --- 4. Function to project clustered point clouds onto a 1D line ---
def project_point_clouds_1d(point_clouds, theta, grid_n=100):
    """
    Projects a list of point clouds onto a 1D line defined by a unit vector theta.
    
    Args:
        point_clouds (list of torch.Tensor): A list of point clouds, where each tensor
                                             is of shape (N, D).
        theta (torch.Tensor): A unit vector of shape (D,).
    
    Returns:
        torch.Tensor: A new tensor of projected point clouds, where the shape
                      is (M, N, 1).
    """
    # Ensure theta is a unit vector
    theta = theta / torch.norm(theta)
    
    projected_clouds = []
    for cloud in point_clouds:
        # Perform the dot product for each point in the cloud.
        # The result is a 1D tensor of shape (N, 1).
        projected_cloud = torch.matmul(cloud, theta.unsqueeze(1))
        
        # Sort the projected points
        sorted_cloud, _ = torch.sort(projected_cloud, dim=0)
        interp_cloud, grid = estimate_quantile_values(sorted_cloud.permute(1, 0), grid_n=grid_n)
        interp_cloud = interp_cloud.permute(1, 0)
        projected_clouds.append(interp_cloud)
        
    return torch.stack(projected_clouds, dim=0)[:, :, 0], grid


def calc_SHOT_between_subsets(sub1, sub2, dim=3, length_scale=10., outer_pnum=100, grid_n=10, inner_pnum=100):
    data1_ls = []

    for label in sub1.classes:
        data1_ls.append(sub1.data[sub1.targets == label].reshape(-1, dim).float())
    
    data2_ls = []
    
    for label in sub2.classes:
        data2_ls.append(sub2.data[sub2.targets == label].reshape(-1, dim).float())
    return compute_hierarchical_sliced_wasserstein(data1_ls, data2_ls, length_scale=length_scale,
                                            outer_pnum=outer_pnum, grid_n=grid_n, inner_pnum=inner_pnum)

def calc_HOT_between_subsets(sub1, sub2, dim=3, length_scale=10., outer_pnum=100, grid_n=10, inner_pnum=100):
    data1_ls = []
    
    for label in sub1.classes:
        data1_ls.append(sub1.data[sub1.targets == label].reshape(-1, dim).float())
    
    data2_ls = []
    
    for label in sub2.classes:
        data2_ls.append(sub2.data[sub2.targets == label].reshape(-1, dim).float())
    return compute_hierarchical_wasserstein(data1_ls, data2_ls)
