import torch
import ot
#try:
#    from graph_distances import *
#except:
#    from SFTLB_CodeCopy.graph_distances import *


def estimate_quantile_values(q, grid_n=100, p=2):
    """
    Compute  quantile function evaluations for Wasserstein integration.

    Parameters:
        q       : Sorted samples (N,) or (S, N) as torch tensor
        grid_n  : Number of evaluation points in [0, 1]
        p       : Power for Wasserstein-p

    Returns:
        q_interp : quantile evaluations (grid_n,) or (S, grid_n)
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


    if not batched:
        return q_interp[0], t_grid
    else:
        return q_interp, t_grid

def calc_functional_SWD_from_sorted(mu_0_proj, mu_1_proj, grid_n=10, n_projections=100, length_scale=1.):
    mu0_interp, _ = estimate_quantile_values(mu_0_proj, grid_n=grid_n)
    mu1_interp, _ = estimate_quantile_values(mu_1_proj, grid_n=grid_n)
    if ("r" in str(length_scale).lower()) and not ("b" in str(length_scale).lower()):
        K = riesz_kernel_covariance(_)
    elif "b" in str(length_scale).lower():
        K = brownian_kernel_covariance(_)
    else:
        K = gaussian_kernel_covariance(_, length_scale=length_scale)
    direction_samples = sample_from_gaussian_with_covariance(K, n_projections)
    sw2 = sliced_functional_wasserstein_distance(mu0_interp, mu1_interp, direction_samples, _)
    return sw2


def calc_functional_SWD_from_discrete_data(mu_0, mu_1, grid, n_projections=100, length_scale=1.):
    if ("r" in str(length_scale).lower()) and not ("b" in str(length_scale).lower()):
        K = riesz_kernel_covariance(grid)
    elif "b" in str(length_scale).lower():
        K = brownian_kernel_covariance(grid)
    else:
        K = gaussian_kernel_covariance(grid, length_scale=length_scale)
    direction_samples = sample_from_gaussian_with_covariance(K, n_projections)
    sw2 = sliced_functional_wasserstein_distance(mu_0, mu_1, direction_samples, grid)
    return sw2


def project_onto_directions(X: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """
    Projects X ∈ (N, D) onto directions ∈ (P, D), returns (P, N).
    """
    X = X.unsqueeze(0)  # (1, N, D)
    directions = directions.unsqueeze(1)  # (P, 1, D)
    return torch.sum(X * directions, dim=2)  # (P, N)



def project_onto_directions_trapz(X: torch.Tensor, directions: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Numerically project X ∈ (N, D) onto directions ∈ (P, D) using trapezoidal integration over the grid.
    Returns a tensor of shape (P, N) with scalar products approximated by trapezoidal rule.
    
    Args:
        X (Tensor): shape (N, D), N samples discretized over D points.
        directions (Tensor): shape (P, D), P direction functions discretized over same D points.
        grid (Tensor): shape (D,), increasing grid points over the domain.
    
    Returns:
        projections (Tensor): shape (P, N), approximate scalar products.
    """
    N, D = X.shape
    P = directions.shape[0]
    # assert grid.ndim == 1 and grid.shape[0] == D
    h = grid[1:] - grid[:-1]  
    X_exp = X.unsqueeze(0)           
    dir_exp = directions.unsqueeze(1)  
    prod = X_exp * dir_exp
    left = prod[:, :, :-1]    # (P, N, D-1)
    right = prod[:, :, 1:]    # (P, N, D-1)
    avg = (left + right) / 2  # (P, N, D-1)
    h = h.reshape(1, 1, -1)   # (1, 1, D-1)
    integral = (avg * h).sum(dim=2)  # (P, N)
    return integral

def estimate_mean_1d_wasserstein2(
    X_proj: torch.Tensor,
    Y_proj: torch.Tensor
) -> torch.Tensor:
    """
    Estimates the mean 1D Wasserstein-2 distance between corresponding
    projected distributions.

    This function assumes that X_proj and Y_proj are already the 1D
    projections of your original data X and Y onto a set of directions.

    Arguments:
        X_proj: A tensor of shape (P, N), where P is the number of projections
                and N is the number of samples in X for each projection.
        Y_proj: A tensor of shape (P, M), where P is the number of projections
                and M is the number of samples in Y for each projection.

    Returns:
        A scalar tensor representing the mean 1D Wasserstein-2 distance
        across all projections.
    """
    # Ensure the number of projections (P) is the same for both inputs
    assert X_proj.shape[0] == Y_proj.shape[0], \
        "Number of projections (P) must be the same for X_proj and Y_proj."
    
    w2_distances_per_projection = ot.wasserstein_1d(X_proj.permute(1, 0), Y_proj.permute(1, 0), p=2)


    return w2_distances_per_projection.mean()
    
def sliced_functional_wasserstein_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    directions: torch.Tensor,
    grid: torch.Tensor, # This argument is passed to project_onto_directions_trapz
) -> torch.Tensor:
    """
    Compute sliced Wasserstein-p distance between X ∈ (N, D) and Y ∈ (M, D)
    along projection directions ∈ (P, D).

    Arguments:
        X, Y: tensors of shape (N, D), (M, D) representing the datasets.
        directions: tensor of shape (P, D) representing the projection directions.
        grid: An external grid parameter used by project_onto_directions_trapz.
              (Assumed to be correctly handled by the external function).
        p: power of Wasserstein distance (1 or 2).
        use_pot: boolean, if True, uses the POT library for 1D Wasserstein calculation.

    Returns:
        Scalar sliced Wasserstein-p distance (SW_p).
    """

    X_proj = project_onto_directions_trapz(X, directions.to(X.device), grid)  # (P, N)
    Y_proj = project_onto_directions_trapz(Y.to(X.device), directions.to(X.device), grid)  # (P, M)
    return estimate_mean_1d_wasserstein2(X_proj, Y_proj)

def gaussian_kernel_covariance(
    x: torch.Tensor,
    length_scale: float = 0.1,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Compute the covariance matrix of a 1D Gaussian Process using the RBF kernel.

    Args:
        x: (N,) or (N, 1) tensor of input locations (interval samples)
        length_scale: float > 0, kernel length scale (ℓ)
        variance: float > 0, kernel variance (σ²)

    Returns:
        K: (N, N) covariance matrix
    """
    x = x.view(-1, 1)  # Ensure shape (N, 1)
    sq_dist = (x - x.T) ** 2  # (N, N)
    K = variance * torch.exp(-0.5 * sq_dist / length_scale**2)
    return K.float()

def riesz_kernel_covariance(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Compute the covariance matrix of a 1D Gaussian Process using the Riesz kernel.
    Args:
        x: (N,) or (N, 1) tensor of input locations (interval samples)
        length_scale: float > 0, kernel length scale (ℓ)
        variance: float > 0, kernel variance (σ²)

    Returns:
        K: (N, N) covariance matrix
    """
    x = x.view(-1, 1)  # Ensure shape (N, 1)
    sq_dist = (x - x.T) ** 2  # (N, N)
    K = 1 - torch.sqrt(sq_dist)
    return K.float()

def brownian_kernel_covariance(
    x: torch.Tensor,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Compute the covariance matrix of a 1D Brownian motion (Wiener process).
    Args:
        x: (N,) or (N, 1) tensor of time points (increasing, e.g. linspace)
        variance: float > 0, variance parameter σ²

    Returns:
        K: (N, N) covariance matrix with entries 1+σ² * min(x_i, x_j)
    """
    x = x.view(-1, 1)  # Shape (N, 1)
    K = variance * torch.minimum(x, x.T)  # (N, N)
    return K.float()


def sample_from_gaussian_with_covariance(
    K: torch.Tensor,   # (N, N) covariance matrix
    P: int,             # number of samples
    normed = False # legacy
) -> torch.Tensor:
    """
    Sample P vectors from N(0, K), where K is (N, N) covariance matrix.

    Returns:
        samples: (P, N) tensor; each row is a sample from N(0, K)
    """
    N = K.shape[0]
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(N, device=K.device))  # Add jitter for numerical stability

    z = torch.randn(P, N, device=K.device)  # Standard normal samples (P, N)
    samples = z @ L.T                      # (P, N) * (N, N)ᵗ = (P, N)

    return samples

def sample_on_trapezoidal_grid(func_ls, start: float, end: float, grid_n: int, device=None):
    """
    Sample multiple 1D functions on a trapezoidal integration grid.

    Args:
        func_ls (list of callables): each function maps a Tensor of shape (N,) to (N,) or (N, 1).
        start (float): left endpoint of the interval.
        end (float): right endpoint.
        grid_n (int): number of grid points (including endpoints).
        device (torch.device or str, optional): device for tensors.

    Returns:
        grid (Tensor): shape (grid_n,), the shared grid points.
        values (Tensor): shape (grid_n, FUNC_NUM), stacked function values.
    """
    grid = torch.linspace(start, end, steps=grid_n, device=device)
    values_list = []

    for f in func_ls:
        vals = f(grid)
        # Ensure output is at least (N, 1)
        if vals.ndim == 0:
            vals = vals.expand(grid.shape[0], 1)
        elif vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        elif vals.ndim == 2 and vals.shape[0] != grid_n:
            raise ValueError(f"Function output shape mismatch: expected first dimension {grid_n}, got {vals.shape[0]}")

        values_list.append(vals)
    values = torch.cat(values_list, dim=1)  # shape: (grid_n, FUNC_NUM)
    return values.permute(1, 0), grid

def calc_functional_SW(func0_ls, func1_ls, grid_n=8, n_projections=10, length_scale=10., start=0., end=1., normed=False):
    f0_discrete, grid = sample_on_trapezoidal_grid(func0_ls, start=start, end=end, grid_n=int(grid_n))
    f1_discrete, grid = sample_on_trapezoidal_grid(func1_ls, start=start, end=end, grid_n=int(grid_n))
    if ("r" in str(length_scale).lower()) and not ("b" in str(length_scale).lower()):
        K = riesz_kernel_covariance(grid)
    elif "b" in str(length_scale).lower():
        K = brownian_kernel_covariance(grid)
    else:
        K = gaussian_kernel_covariance(grid, length_scale=length_scale)
    direction_samples = sample_from_gaussian_with_covariance(K, n_projections, normed=normed)
    sw2 = sliced_functional_wasserstein_distance(f0_discrete, f1_discrete, direction_samples, grid)
    return sw2