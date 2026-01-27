# Load packages
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from ct_setup import ct_astra, astra
import csv
from projector_setup import fp_astra, bp_astra
from solvers import BA_GMRES
import sigpy as sp

#lets create the ct-experiment 
def create_ct(num_pixels, num_angles, num_dets, det_width=1.0,
            proj_model="linear", proj_geom="parallel",
            source_origin=1000, origin_det=0, gpu=True):
    angles = np.linspace(0, 360, num_angles, endpoint=False) * np.pi / 180.0
    ct = ct_astra(num_pixels, num_angles, num_dets, angles,
                  proj_model, proj_geom, source_origin, origin_det, det_width, gpu)
    return ct

#For now we just create a random ground truth
def make_ground_truth_random(num_pixels):
    return np.random.randn(num_pixels, num_pixels).astype(np.float32)


def create_sheppLogan_phantom(num_pixels):
    return sp.shepp_logan((num_pixels, num_pixels), dtype='complex')



#We create the rhs, which is a sinogram. We also make this function return the reshaped (vectorized) sinogram
###The reshaped sinogram is the noise-free rhs
def make_sinogram(ct, X_true):
    _, sino = astra.create_sino(X_true.astype(np.float32), ct.proj_id)
    return sino.astype(np.float32), sino.reshape(-1).astype(np.float32)  # shape of sino (num_angles, num_dets)


#We add noise to the sinogram to be able to "create" semiconvergence
def add_relative_noise(b_exact, rnl):
    e = np.random.randn(b_exact.size).astype(np.float32)
    e /= np.linalg.norm(e)
    b = b_exact + rnl * np.linalg.norm(b_exact) * e
    return b

#We define the forward and backward projector for the specific problem and run BA-GMRES to be able to extract the iterations, residuals 
#and Hessenberg matrices
##X_BA is a matrix where the columns are the iterations. Its dimensions are ((numpixels^2) x (iter + 1))
###H is simply the final (rectangular) Hessenberg matrix whose leading minors are the Hessenberg matrices in the previous steps.
def run_ba_gmres(ct, b, iters):
    A = fp_astra(ct)
    B = bp_astra(ct)
    X_BA, R_BA, H, W = BA_GMRES(A, B, b, iters, ct.m, ct.n, ct.num_angles)
    return X_BA, R_BA, H, A, B, W


#We extract the square and rectangular Hessenberg matrices from the lists to be able to conduct an analysis
#similar to one for the small test problems
def extract_hessenberg_blocks(H):
    """Return lists [H_1,...,H_K] and [H_{2,1},...,H_{K+1,K}] from final H."""
    K = H.shape[1]
    H_square = [H[:k, :k].copy() for k in range(1, K+1)]
    H_rect   = [H[:k+1, :k].copy() for k in range(1, K+1)]
    return H_square, H_rect

def extract_V_bases(V):
    # V is (n, K+1) in BA-GMRES; usable basis sizes are 1..K
    K = V.shape[1] - 1
    return [V[:, :k].copy() for k in range(1, K+1)]


##This function computes the relative errors in each iteration and also the the smallest error and the index for this
def compute_rel_errors(X_BA, X_true):
    x_true = X_true.reshape(-1)
    relative_errors = np.array([np.linalg.norm(x_true - X_BA[:,k]) 
                     for k in range(X_BA.shape[1])])
    Val_BA = np.min(relative_errors)
    idx_BA = np.argmin(relative_errors)

    return relative_errors, Val_BA, idx_BA




def grains(N: int, num_cells: int | None = None, seed: int | None = None) -> np.ndarray:
    """
    Generate a 'grains' phantom: Voronoi cells with random intensities in [0, 1].

    Parameters
    ----------
    N : int
        Image size (N x N).
    num_cells : int | None
        Number of Voronoi seeds/cells. Default = round(3*sqrt(N)).
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    im : (N, N) ndarray
        Float image with values in [0, 1].
    """
    if N <= 0:
        raise ValueError("N must be positive")

    if num_cells is None:
        num_cells = int(np.round(3 * np.sqrt(N)))
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")

    rng = np.random.default_rng(seed)

    # Random seed locations in continuous pixel coordinates [0, N)
    seeds = rng.uniform(0, N, size=(num_cells, 2))  # columns: (row, col)

    # Pixel grid coordinates (row, col) for all pixels
    rr, cc = np.indices((N, N))
    pixels = np.stack([rr, cc], axis=-1).reshape(-1, 2)  # shape: (N*N, 2)

    # Compute squared distances from each pixel to each seed:
    # d^2(p, s) = (pr - sr)^2 + (pc - sc)^2
    # Result shape: (N*N, num_cells)
    diff = pixels[:, None, :] - seeds[None, :, :]
    d2 = np.sum(diff * diff, axis=2)

    # Nearest seed index for each pixel
    labels = np.argmin(d2, axis=1)  # shape: (N*N,)

    # Random intensity per cell
    intensities = rng.uniform(0.0, 1.0, size=num_cells)

    # Assign intensities to pixels via their Voronoi label
    im = intensities[labels].reshape(N, N).astype(np.float64)
    return im


##############-------------------Lets also choose the test problems in this script---------------------##########
def make_ba_test_problem(
    name,
    *,
    rnl: float,
    seed: int = 0,
    gpu: bool = True,
    ground_truth: str = "shepp_logan",
):
    """
    If ground_truth == "shepp_logan" or "random":
        name ∈ {"lille","medium","stor"} decides the size.

    If ground_truth == "grains":
        name is interpreted as num_pixels (int), and we set num_dets=num_pixels
        and choose a simple default num_angles.
    """
    np.random.seed(seed)

    # ---------- Choose geometry based on ground truth ----------
    if ground_truth == "shepp_logan" or ground_truth == "random":
        if name == "lille":
            num_pixels, num_dets, num_angles = 32, 32, 180
        elif name == "medium":
            num_pixels, num_dets, num_angles = 128, 128, 180
        elif name == "stor":
            num_pixels, num_dets, num_angles = 420, 420, 600
        else:
            raise ValueError(f"Unknown BA test problem '{name}'")

    elif ground_truth == "grains":
        num_pixels = int(name)          # name is now e.g. 200, 300, ...
        num_dets   = num_pixels
        num_angles = 180                # simple default (change if you want)

    else:
        raise ValueError("Not a valid ground truth")

    # ---------- CT setup ----------
    CT_setup = create_ct(
        num_pixels, num_angles, num_dets,
        det_width=1.0,
        proj_model="linear",
        proj_geom="parallel",
        source_origin=1000,
        origin_det=0,
        gpu=gpu
    )

    # ---------- Ground truth ----------
    if ground_truth == "shepp_logan":
        x_true = create_sheppLogan_phantom(num_pixels)
    elif ground_truth == "grains":
        x_true = grains(num_pixels)
    elif ground_truth == "random":
        x_true = make_ground_truth_random(num_pixels)

    # ---------- Data ----------
    sinogram, b_exact = make_sinogram(CT_setup, x_true)
    b_noisy = add_relative_noise(b_exact, rnl)

    info = {
        "name": name,
        "ground_truth": ground_truth,
        "num_pixels": num_pixels,
        "num_dets": num_dets,
        "num_angles": num_angles,
        "rnl": rnl,
    }

    return CT_setup, b_exact, b_noisy, x_true, info



from scipy.sparse.linalg import LinearOperator

def make_BA_operator(A, B, dtype=np.float64):
    """
    This function takes the forward and backward projector
    and returns a linear operator which performs the 
    matrix-vector multiplication B(A(x))
    
    :param A: forward projector: linear operator 
    :param B: backward projector: linear operator
    
    Returns:
    A linear operator which performs the matrix-vector
    multiplication  B(A(x))
    """
    n = A.num_pixels * A.num_pixels

    def matvec(x):
        x = np.asarray(x, dtype=dtype).reshape(-1)
        return np.asarray(B @ (A @ x), dtype=dtype).reshape(-1)

    return LinearOperator((n, n), matvec=matvec, dtype=dtype)


############-----------------------------------------------------------------------------------------------############
############_-------------------------This is for the new Ritz basis---------------------------------------############

def ritz_decomp_from_arnoldi_lists(V_list, H_rect_list, H_square_list, rhs_vec, k, one_based=True, return_Yk=False):
    """
    Compute harmonic Ritz coordinates of rhs_vec using Arnoldi outputs stored in lists.

    Your arnoldi_mgs stores, for each step:
        V_list[j]         = V_{j+1}   with shape (n, j+1)
        H_rect_list[j]    = \bar H_{j+1} with shape (j+2, j+1)
        H_square_list[j]  = H_{j+1}   with shape (j+1, j+1)

    Here, k denotes the Krylov subspace dimension (number of basis vectors). If one_based=True,
    then k=1 corresponds to j=0 (first stored entry). If one_based=False, k is the list index.

    It forms the harmonic Ritz generalized eigenproblem (as in your code):
        (H_rect^H H_rect) y = θ (H_square^H) y

    then computes:
        g = V_k^H rhs_vec
        c = Y_k^{-1} g

    Parameters
    ----------
    V_list : list of ndarray
        V_list[j] has shape (n, j+1).
    H_rect_list : list of ndarray
        H_rect_list[j] has shape (j+2, j+1).
    H_square_list : list of ndarray
        H_square_list[j] has shape (j+1, j+1).
    rhs_vec : array_like, shape (n,)
        Vector to expand (in the same space as V_k columns).
    k : int
        Krylov dimension to use.
        - if one_based=True: k=1,2,... corresponds to V_list[k-1]
        - if one_based=False: k is the list index (0,1,2,...)
    one_based : bool, default True
        Interpret k as 1-based Krylov dimension.
    return_Yk : bool, default False
        If True, also return harmonic Ritz values theta and matrix Y_k.

    Returns
    -------
    c : ndarray, shape (k,)
        Harmonic Ritz coordinates of rhs_vec on span(V_k).
    (optional) theta : ndarray, shape (k,)
        Harmonic Ritz values.
    (optional) Y_k : ndarray, shape (k, k)
        Harmonic Ritz eigenvectors in Krylov coordinates.
    """
    j = k - 1 if one_based else k

    V_k = np.asarray(V_list[j])            # (n, k)
    H_rect = np.asarray(H_rect_list[j])    # (k+1, k)?? actually (k+1+1, k) = (k+1? see below)
    H_sq = np.asarray(H_square_list[j])    # (k, k)
    r = np.asarray(rhs_vec).reshape(-1)

    n, kk = V_k.shape
    if r.shape[0] != n:
        raise ValueError(f"rhs_vec has length {r.shape[0]} but V_k has n={n}.")
    if kk != k:
        raise ValueError(f"Requested k={k} but V_list[{j}] has {kk} columns.")

    # Your stored shapes imply:
    # H_rect_list[j] should be (k+1, k)?? but from your code it's H_current = H[:k+2, :k+1]
    # with k replaced by j => (j+2, j+1) = (k+1, k)
    # Actually if kk = k then H_rect should be (k+1, k). Let's enforce that:
    if H_rect.shape != (k + 1, k):
        raise ValueError(f"H_rect_list[{j}] must be shape {(k+1, k)}, got {H_rect.shape}.")
    if H_sq.shape != (k, k):
        raise ValueError(f"H_square_list[{j}] must be shape {(k, k)}, got {H_sq.shape}.")

    from scipy.linalg import eig
    # Harmonic Ritz generalized eigenproblem
    lhs = H_rect.conj().T @ H_rect   # (k, k)
    rhs = H_sq.conj().T              # (k, k)
    theta, Y_k = eig(lhs, rhs)

    # Coordinates
    g = V_k.conj().T @ r             # (k,)
    c = np.linalg.solve(Y_k, g)      # (k,)

    if return_Yk:
        return c, theta, Y_k
    return c


###The function that plots the homemade Harmonic Ritz basis
def ritz_piccard(ritz_xi_exact, ritz_xi_noisy, harmonic_ritz_values, k, outpath=None):
    """
    Picard-ish plot for the harmonic Ritz setting (SAVES as PDF; does not show).

    Plots (semilogy) for iteration k:
      - |theta_i|       : harmonic Ritz values at iteration k
      - |c_i| (exact)   : harmonic Ritz coordinates of the exact rhs/residual
      - |c_i| (noisy)   : harmonic Ritz coordinates of the noisy rhs/residual

    Parameters
    ----------
    ritz_xi_exact : array_like
        Harmonic Ritz coordinates c for the exact rhs/residual (length k).
    ritz_xi_noisy : array_like
        Harmonic Ritz coordinates c for the noisy rhs/residual (length k).
    harmonic_ritz_values : list
        harmonic_ritz_values[k-1] contains the harmonic Ritz values theta at iteration k.
    k : int
        Iteration number (1-based).
    outpath : str or None
        Where to save the figure. If None, saves to "ritz_piccard_k{k}.pdf" in the current folder.
    """
    theta = np.asarray(harmonic_ritz_values[k - 1]).reshape(-1)
    c_exact = np.asarray(ritz_xi_exact).reshape(-1)
    c_noisy = np.asarray(ritz_xi_noisy).reshape(-1)

    # Make lengths consistent (use the smallest)
    m = min(theta.size, c_exact.size, c_noisy.size)
    theta = theta[:m]
    c_exact = c_exact[:m]
    c_noisy = c_noisy[:m]

    fig = plt.figure(figsize=(8, 4))
    plt.semilogy(np.abs(theta), 'o-', label=r'$|\theta_i|$')
    plt.semilogy(np.abs(c_exact), 's-', label=r'$|c_i|$ (exact)')
    plt.semilogy(np.abs(c_noisy), 'x--', label=r'$|c_i|$ (noisy)')
    plt.xlabel('Index $i$')
    plt.ylabel('Magnitude (log scale)')
    plt.title(f'Harmonic Ritz Picard-ish Condition (iteration k={k})')
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    if outpath is None:
        outpath = f"ritz_piccard_k{k}.pdf"

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_bagmres_ritz_weights_vs_iteration_error(
    ritz_xi,
    ritz_values_list,
    iteration_errors,
    k,
    outpath=None,
    use_ratio=True,
    title_prefix="BA-GMRES"
):
    """
    Save a plot comparing harmonic-Ritz “spectral” weights with the BA-GMRES iteration error curve.

    This is the BA-GMRES analogue of plotting |xi_i|/|lambda_i| vs ||x_k - x_true||:
      - Replace eigenbasis coefficients xi_i with harmonic Ritz coefficients c_i (= ritz_xi)
      - Replace eigenvalues lambda_i with harmonic Ritz values theta_i (at iteration k)

    Parameters
    ----------
    ritz_xi : array_like, shape (k,) or (<=k,)
        Harmonic Ritz coordinates c at iteration k (output of ritz_decomp_from_arnoldi_lists(...)).
    ritz_values_list : list
        List where ritz_values_list[k-1] contains harmonic Ritz values theta at iteration k.
        (In your script, this is typically `ritz_noisefree`.)
    iteration_errors : array_like
        Error curve across BA-GMRES iterations, e.g. ||x_j - x_true|| for j=1..iters.
    k : int
        Iteration number (1-based) selecting ritz_values_list[k-1].
    outpath : str or None
        Where to save the figure as PDF. If None, saves to "ritz_weights_vs_itererr_k{k}.pdf".
    use_ratio : bool, default True
        If True, plot |c_i|/|theta_i| (closest analogue to |xi_i|/|lambda_i|).
        If False, plot |c_i| alone.
    title_prefix : str, default "BA-GMRES"
        Prefix used in the plot title.

    Notes
    -----
    - The two curves have different x-axes meanings (component index vs iteration index),
      but overlaying them is still a useful qualitative diagnostic (as in your eigenbasis plots).
    """
    theta = np.asarray(ritz_values_list[k - 1]).reshape(-1)
    c = np.asarray(ritz_xi).reshape(-1)
    it_err = np.asarray(iteration_errors).reshape(-1)

    # Align Ritz quantities to the smallest available length
    m = min(theta.size, c.size)
    theta = theta[:m]
    c = c[:m]

    if use_ratio:
        eps = 1e-300  # avoid divide-by-zero
        weights = np.abs(c) / np.maximum(np.abs(theta), eps)
        w_label = r"$|c_i|/|\theta_i|$"
    else:
        weights = np.abs(c)
        w_label = r"$|c_i|$"

    fig = plt.figure(figsize=(10, 6))

    plt.plot(
        weights,
        label=w_label,
        linewidth=2,
        marker='o',
        markersize=4
    )

    plt.plot(
        it_err,
        label=r"$\|x_j - x_{\mathrm{true}}\|$",
        linewidth=2,
        marker='s',
        markersize=4
    )

    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.xlabel("Index (component i for weights, iteration j for errors)")
    plt.ylabel("Magnitude (log scale)")
    plt.title(f"{title_prefix}: Ritz weights vs iteration error (k={k})")
    plt.legend()
    plt.tight_layout()

    if outpath is None:
        outpath = f"ritz_weights_vs_itererr_k{k}.pdf"

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


#The below function creates the full BA-matrix so we can conduct the analysis similar to the on e we conducted for the small test probs
def construct_full_BA(BA, n, dtype=float):
    """
    Construct the full matrix BA by applying the linear operator BA to
    all canonical basis vectors e_j.

    Parameters
    ----------
    BA : callable
        Function that maps x (shape (N,)) -> BAx (shape (N,))
    n : int
        Image side length; total dimension N = n^2
    dtype : numpy dtype

    Returns
    -------
    M : ndarray, shape (N, N)
        Explicit matrix representation of BA
    """
    N = n * n
    M = np.empty((N, N), dtype=dtype)

    e = np.zeros(N, dtype=dtype)
    for j in range(N):
        e.fill(0)
        e[j] = 1
        M[:, j] = BA(e)

    return M