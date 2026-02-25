import torch
import numpy as np
import ot
import torch
import numpy as np
import copy
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.kernels import VertexHistogram
from grakel.kernels import RandomWalk
from grakel.kernels import GraphletSampling
from scipy.spatial.distance import cdist, pdist
import numpy as np
from scipy.interpolate import interp1d
import torch
from scipy.linalg import sqrtm
import torch
from geomloss import SamplesLoss

def to_float64(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    elif isinstance(x, torch.Tensor):
        return x.double()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
        
def convert_graph(graph):
    """
    Converts a graph represented as [Distance Matrix, Weights, Features] to GraKeL's Graph format.
    """
    D, W, F = graph
    # Convert tensors to numpy arrays
    D = D.numpy() if isinstance(D, torch.Tensor) else D
    W = W.numpy() if isinstance(W, torch.Tensor) else W
    F = F.numpy() if isinstance(F, torch.Tensor) else F
    # Create edge list with weights
    edges = []
    num_nodes = D.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if D[i, j] > 0:
                edges.append((i, j, D[i, j]))
    # Create node labels from features
    node_labels = {i: tuple(F[i]) for i in range(num_nodes)}
    return Graph(edges, node_labels=node_labels)



def calc_SLB(graph0, graph1, alpha=None):
    global_dist_0 = graph0[0].flatten()
    global_dist_1 = graph1[0].flatten()
    mass0 = graph0[1].repeat(graph0[0].shape[0])
    mass0 = mass0/mass0.sum()
    mass1 = graph1[1].repeat(graph1[0].shape[0])
    mass1 = mass1/mass1.sum()
    return ot.emd2_1d(global_dist_0, global_dist_1, mass0, mass1)


def tlb_process(graph0, graph1, alpha=.5):
    gdim0 = graph0[0].shape[0]
    gdim1 = graph1[0].shape[0]
    mindim = min([gdim0, gdim1])
    g0_sort, _ = graph0[0].sort(dim=-1)
    g0_sort = np.sqrt(((1-alpha)/mindim)) * g0_sort
    g1_sort, _ = graph1[0].sort(dim=-1)
    g1_sort = np.sqrt(((1-alpha)/mindim)) *  g1_sort
    g0feat = np.sqrt(alpha) * graph0[2]
    g1feat = np.sqrt(alpha) * graph1[2]
    return gdim0, gdim1, mindim, g0feat, g0_sort, g1feat, g1_sort
    
def calc_FTLB(graph0, graph1, alpha=.5, eps=0, plan=False):
    gdim0, gdim1, mindim, g0feat, g0_sort, g1feat, g1_sort = tlb_process(graph0, graph1, alpha=alpha)
    graph0_cat = torch.concatenate([g0_sort, g0feat], dim=-1)
    graph1_cat = torch.concatenate([g1_sort, g1feat], dim=-1)
    M = ot.dist(graph0_cat, graph1_cat)
    if plan:
        return ot.emd(graph0[1], graph1[1], M, numItermax=int(1e6))
    elif eps == 0:
            return ot.emd2(graph0[1], graph1[1], M, numItermax=int(1e6))
    else:
        return ot.bregman.sinkhorn_epsilon_scaling(graph0[1], graph1[1], M, reg=eps, numItermax=int(1e6))

def calc_FTLB_general(graph0, graph1, alpha=.5, plan=False):    
    ot_cost = torch.zeros(graph0[0].shape[0], graph1[0].shape[0])
    gdim0, gdim1, mindim, g0feat, g0_sort, g1feat, g1_sort = tlb_process(graph0, graph1) #, alpha=alpha)

    for d0 in range(graph0[0].shape[0]):
        for d1 in range(graph1[0].shape[0]):
            ot_cost[d0, d1] = optimal_transport_1d_presorted(g0_sort[d0], g1_sort[d1])
    ot_cost = (1-alpha) * ot_cost
    feature_cost = alpha * ot.dist(graph0[2], graph1[2])
    M = ot_cost + feature_cost
    if plan:
        return ot.emd(graph0[1], graph1[1], M)
    else:
        return ot.emd2(graph0[1], graph1[1], M)

def optimal_transport_1d_presorted(x, y):
    """
    Calculates the 1-Wasserstein distance (Earth Mover's Distance)
    between two 1D empirical distributions, given their presorted samples.

    This function assumes uniform weights for each sample in x and y.

    Args:
        x (numpy.ndarray or torch.Tensor): A 1D array/tensor of shape (N,) or (N,1)
                                           representing the first distribution's samples,
                                           must be presorted in non-decreasing order.
        y (numpy.ndarray or torch.Tensor): A 1D array/tensor of shape (M,) or (M,1)
                                           representing the second distribution's samples,
                                           must be presorted in non-decreasing order.

    Returns:
        float: The 1-Wasserstein distance between the two distributions.
               Returns 0.0 if either input array is empty.
    """
    # Ensure inputs are NumPy arrays and flatten them if necessary
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    x = x.flatten()
    y = y.flatten()

    n = len(x)
    m = len(y)

    if n == 0 or m == 0:
        return 0.0

    i = 0  # Pointer for x
    j = 0  # Pointer for y

    current_mass_x = 0.0
    current_mass_y = 0.0
    total_cost = 0.0

    # Initialize previous_pos to the first relevant point.
    # This correctly sets up the start of the integral.
    previous_pos = min(x[0], y[0])

    while i < n or j < m:
        next_pos = float('inf')

        # Determine the next critical point where a CDF changes
        if i < n:
            next_pos = min(next_pos, x[i])
        if j < m:
            next_pos = min(next_pos, y[j])

        # Add the area of the current "rectangle" to the total cost
        # The height is the absolute difference of current CDF values
        # The width is the distance between previous and next critical points
        total_cost += abs(current_mass_x - current_mass_y) * (next_pos - previous_pos)

        # Update masses and advance pointers for the next iteration
        # Important: Use 'if' statements for both to handle cases where x[i] == y[j]
        # (i.e., both distributions have a mass point at the same location)
        if i < n and x[i] == next_pos:
            current_mass_x += 1.0 / n
            i += 1
        if j < m and y[j] == next_pos:
            current_mass_y += 1.0 / m
            j += 1

        previous_pos = next_pos

    return total_cost

def calc_SFTLB(graph0, graph1, alpha=.5, subsample_num=10, n_projections=100, subsample_dim=False):
    gdim0, gdim1, mindim, g0feat, g0_sort, g1feat, g1_sort = tlb_process(graph0, graph1, alpha=alpha)
    if subsample_dim:
        mindim = subsample_dim
    dist = 0.
    for i in range(subsample_num):
        perm = torch.randperm(gdim0)[:mindim].sort().values
        g0_subsort = g0_sort[perm, :]#.sort(dim=-1)
        g0_subsort = g0_subsort[:, perm]
        g0feat_sub = g0feat[perm, :]
        perm = torch.randperm(gdim1)[:mindim].sort().values
        g1_subsort = g1_sort[perm, :]#.sort(dim=-1)
        g1_subsort = g1_subsort[:, perm]
        g1feat_sub = g1feat[perm, :]
        g0_cat = torch.concatenate([g0_subsort, g0feat_sub], dim=-1)
        g1_cat = torch.concatenate([g1_subsort, g1feat_sub], dim=-1)
        dist += ot.sliced.sliced_wasserstein_distance(g0_cat, g1_cat, n_projections=n_projections)
    return dist/subsample_num

def scaled_quantile_values(q, grid_n=100, p=2):
    """
    Compute trapezoid-weighted, p-scaled quantile function evaluations for Wasserstein integration.

    Parameters:
        q       : Sorted samples (N,) or (S, N) as torch tensor
        grid_n  : Number of evaluation points in [0, 1]
        p       : Power for Wasserstein-p

    Returns:
        q_scaled : Scaled quantile evaluations (grid_n,) or (S, grid_n)
        t_grid   : Evaluation grid (grid_n,)
    """
    q = torch.as_tensor(q)
    batched = q.ndim == 2
    if not batched:
        q = q.unsqueeze(0)  # shape becomes (1, N)

    S, N = q.shape
    device = q.device
    dtype = q.dtype

    # Integration grid in [0, 1]
    t_grid = torch.linspace(0, 1, grid_n, device=device, dtype=dtype)

    # Trapezoidal weights (length grid_n)
    weights = torch.zeros_like(t_grid)
    weights[1:-1] = (t_grid[2:] - t_grid[:-2]) / 2
    weights[0] = (t_grid[1] - t_grid[0]) / 2
    weights[-1] = (t_grid[-1] - t_grid[-2]) / 2
    weights_p = weights.pow(1 / p)  # shape (grid_n,)

    # Empirical CDF grid
    cdf = torch.linspace(0, 1, N, device=device, dtype=dtype)  # shape (N,)

    # Interpolation using linear weights
    idx = torch.searchsorted(cdf, t_grid, right=True).clamp(1, N - 1)
    t0 = cdf[idx - 1]
    t1 = cdf[idx]
    w1 = (t_grid - t0) / (t1 - t0 + 1e-12)
    w0 = 1 - w1

    q_left = q[:, idx - 1]
    q_right = q[:, idx]
    q_interp = w0 * q_left + w1 * q_right  # shape (S, grid_n)

    q_scaled = q_interp * weights_p  # broadcasted over (S, grid_n)

    if not batched:
        return q_scaled[0], t_grid
    else:
        return q_scaled, t_grid



def wasserstein_p_trapezoid(q1, q2, p=2, grid_n=100, squared=True):
    """
    Compute Wasserstein-p distance between 1D distributions using trapezoidal rule.

    Parameters:
        q1, q2  : Sorted samples (N,) or (S, N) as torch tensors
        p       : Order of Wasserstein distance
        grid_n  : Number of grid points
        squared: Return squared distance if True

    Returns:
        Wasserstein distance(s): scalar or tensor of shape (S,)
    """
    q1_scaled, _ = scaled_quantile_values(q1, grid_n=grid_n, p=p)
    q2_scaled, _ = scaled_quantile_values(q2, grid_n=grid_n, p=p)

    diff = q1_scaled - q2_scaled
    dist_p = torch.sum(diff**2, dim=-1)  # shape (S,) or scalar

    if squared:
        return dist_p
    else:
        return dist_p**(1 / p)

def calc_SFTLB_via_numerical_integration(graph0, graph1, alpha=.5, grid_n=10, n_projections=100):
    gdim0, gdim1, mindim, g0feat, g0_sort, g1feat, g1_sort = tlb_process(graph0, graph1, alpha=alpha)
    dist = 0.
    g0_sort_interp, _ = scaled_quantile_values(g0_sort, grid_n=grid_n)
    g1_sort_interp, _ = scaled_quantile_values(g1_sort, grid_n=grid_n)
    g0_cat = torch.concatenate([g0_sort_interp, g0feat], dim=-1)
    g1_cat = torch.concatenate([g1_sort_interp, g1feat], dim=-1)
    dist += ot.sliced.sliced_wasserstein_distance(g0_cat, g1_cat, n_projections=n_projections)
    return dist

def calc_SFTLB(graph0, graph1, alpha=.5, subsample_num=10, n_projections=100, subsample_dim=False):
    gdim0, gdim1, mindim, g0feat, g0_sort, g1feat, g1_sort = tlb_process(graph0, graph1, alpha=alpha)
    if subsample_dim:
        mindim = subsample_dim
    dist = 0.
    for i in range(subsample_num):
        perm = torch.randperm(gdim0)[:mindim].sort().values
        g0_subsort = g0_sort[perm, :]#.sort(dim=-1)
        g0_subsort = g0_subsort[:, perm]
        g0feat_sub = g0feat[perm, :]
        perm = torch.randperm(gdim1)[:mindim].sort().values
        g1_subsort = g1_sort[perm, :]#.sort(dim=-1)
        g1_subsort = g1_subsort[:, perm]
        g1feat_sub = g1feat[perm, :]
        g0_cat = torch.concatenate([g0_subsort, g0feat_sub], dim=-1)
        g1_cat = torch.concatenate([g1_subsort, g1feat_sub], dim=-1)
        dist += ot.sliced.sliced_wasserstein_distance(g0_cat, g1_cat, n_projections=n_projections)
    return dist/subsample_num

def calc_GW(g0, g1, alpha=None):
    dist = ot.gromov.gromov_wasserstein2(
        C1=g0[0], C2=g1[0],
        p=g0[1], q=g1[1],
        loss_fun='square_loss',
        log=False, armijo=False,
        max_iter=int(1e8)
    )
    return dist
    
def calc_FGW(g0, g1, alpha=.5, G0=None, plan=False):
    """
    alpha parameter is reversed in POT library from us!
    """
    M = ot.dist(g0[2], g1[2], metric='sqeuclidean')
    M = to_float64(M)
    if plan:
        return ot.gromov.fused_gromov_wasserstein(
        M=M,
        C1=to_float64(g0[0]), C2=to_float64(g1[0]),
        p=to_float64(g0[1]), q=to_float64(g1[1]),
        loss_fun='square_loss',
        alpha=alpha,
        log=False, armijo=False,
        G0=G0,
        max_iter=int(1e8)
    )
    else:
        return ot.gromov.fused_gromov_wasserstein2(
            M=M,
            C1=to_float64(g0[0]), C2=to_float64(g1[0]),
            p=to_float64(g0[1]), q=to_float64(g1[1]),
            loss_fun='square_loss',
            alpha=alpha,
            log=False, armijo=False,
            G0=G0,
            max_iter=int(1e8)
        )
    


def energy_distance(X, Y):
    """
    Estimate the energy distance between samples X and Y.

    Args:
        X (ndarray): shape (n1, d)
        Y (ndarray): shape (n2, d)

    Returns:
        float: estimated squared energy distance
    """
    n1, n2 = len(X), len(Y)

    # Cross-distance between X and Y
    d_xy = cdist(X, Y, metric='euclidean')
    cross_term = 2 * d_xy.mean()

    # Within distances
    d_xx = pdist(X, metric='euclidean')
    d_yy = pdist(Y, metric='euclidean')
    self_term_x = d_xx.mean() if n1 > 1 else 0.0
    self_term_y = d_yy.mean() if n2 > 1 else 0.0

    #return cross_term - self_term_x - self_term_y
    dist = cross_term - self_term_x - self_term_y
    return dist.clip(0.)

def calc_EnergyLB(graph0, graph1, alpha=.5):
    """
    calcs Anchor Energy distance, see Sato 2020
    """
    gdim0, gdim1, mindim, g0feat, g0_sort, g1feat, g1_sort = tlb_process(graph0, graph1, alpha=alpha)
    g0_subsort = g0_sort
    g0feat_sub = g0feat
    g1_subsort = g1_sort
    g1feat_sub = g1feat
    g0_cat = torch.concatenate([g0_subsort, g0feat_sub], dim=-1)
    g1_cat = torch.concatenate([g1_subsort, g1feat_sub], dim=-1)
    loss_fn = SamplesLoss(loss="energy", p=1)  # p=1 gives the standard energy distance
    return loss_fn(g0_cat, g1_cat)
    #return energy_distance(g0_cat, g1_cat)