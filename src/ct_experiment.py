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
    #e /= np.linalg.norm(e)
    b = b_exact + rnl * e
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
        if name == "mini":
            num_pixels, num_dets, num_angles = 16, 16, 60
        elif name == "lille":
            num_pixels, num_dets, num_angles = 32, 32, 180
        elif name == "BIGlille":
            num_pixels, num_dets, num_angles = 64, 64, 180
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

def ritz_decomp_from_HW(W, H, rhs_vec, k, return_Yk=False, use_lstsq=True):
    """
    Convenience wrapper for BA_GMRES outputs.

    Given BA_GMRES outputs:
      W : (n, p+1) Krylov basis (columns orthonormal; W[:,0]=r0/beta)
      H : (p+1, p) final rectangular Hessenberg (last cycle)

    We form:
      V_k = W[:, :k]
      H_k = H[:k, :k]
    and compute ordinary Ritz coordinates c of rhs_vec:
      theta, Y_k = eig(H_k)
      g = V_k^H rhs_vec
      c solves Y_k c = g
    """
    import numpy as np
    import scipy.linalg

    W = np.asarray(W)
    H = np.asarray(H)
    rhs_vec = np.asarray(rhs_vec).reshape(-1)

    n, wp1 = W.shape
    p = wp1 - 1

    if H.shape != (p + 1, p):
        raise ValueError(f"Expected H shape {(p+1, p)} for W shape {W.shape}, got {H.shape}.")
    if not (1 <= k <= p):
        raise ValueError(f"k must satisfy 1 <= k <= p={p}, got k={k}.")
    if rhs_vec.shape[0] != n:
        raise ValueError(f"rhs_vec has length {rhs_vec.shape[0]} but W has n={n}.")

    V_k = W[:, :k]      # (n, k)
    H_k = H[:k, :k]     # (k, k)

    theta, Y_k = scipy.linalg.eig(H_k)

    g = V_k.conj().T @ rhs_vec

    if use_lstsq:
        c = np.linalg.lstsq(Y_k, g, rcond=None)[0]
    else:
        c = np.linalg.solve(Y_k, g)

    if return_Yk:
        return c, theta, Y_k
    return c




def ritz_piccard(c, theta, k, outpath=None, title_prefix="Ritz", sort_by_theta=True):
    theta = np.asarray(theta).reshape(-1)
    c = np.asarray(c).reshape(-1)

    m = min(theta.size, c.size)
    theta = theta[:m]
    c = c[:m]

    if sort_by_theta:
        perm = np.argsort(-np.abs(theta))
        theta = theta[perm]
        c = c[perm]

    fig = plt.figure(figsize=(8, 4))
    plt.semilogy(np.abs(theta), 'o-', label=r'$|\theta_i|$')
    plt.semilogy(np.abs(c), 'x--', label=r'$|c_i|$')
    plt.xlabel('Index $i$ (ordered by $|\theta|$)' if sort_by_theta else 'Index $i$')
    plt.ylabel('Magnitude (log scale)')
    plt.title(f'{title_prefix} Picard plot (iteration k={k})')
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    if outpath is None:
        outpath = f"ritz_piccard_k{k}.pdf"

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_bagmres_ritz_weights_vs_total_error(
    c, theta, total_errors, k,
    outpath=None, title_prefix="BA-GMRES", sort_by_theta=True
):
    theta = np.asarray(theta).reshape(-1)
    c = np.asarray(c).reshape(-1)
    tot_err = np.asarray(total_errors).reshape(-1)

    m = min(theta.size, c.size)
    theta = theta[:m]
    c = c[:m]

    if sort_by_theta:
        perm = np.argsort(-np.abs(theta))
        theta = theta[perm]
        c = c[perm]

    eps = 1e-300
    weights = np.abs(c) / np.maximum(np.abs(theta), eps)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(weights, label=r"$|c_i|/|\theta_i|$", linewidth=2, marker='o', markersize=4)
    plt.plot(tot_err, label=r"$\|x_k - \bar{x}\|_2$", linewidth=2, marker='s', markersize=4)

    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Index (weights) / iteration (error)")
    plt.ylabel("Magnitude (log scale)")
    plt.title(f"{title_prefix}: Ritz weights vs total error (k={k})")
    plt.legend()
    plt.tight_layout()

    if outpath is None:
        outpath = f"ritz_weights_vs_totalerr_k{k}.pdf"

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


def expected_propagated_noise_norm(
    W,
    filter_matrix,
    eigenvalues,
    eta,
    k_expect=10,            # <-- NEW name: how many k's to compute (first k_expect)
    use_solve=True,
    make_plot=True,
    figsize=(5, 3),
    outpath=None,
):
    """
    Compute E(||R_k^Phi xi^e||_2^2) for k = 0..k_expect-1 and optionally plot sqrt(.) vs k.

    If make_plot is True:
      - If outpath is not None: saves the figure to outpath and closes.
      - Else: closes without showing (headless-safe).

    Returns (exppropsquared, exppropnorm).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    W = np.asarray(W)
    filter_matrix = np.asarray(filter_matrix)
    eigenvalues = np.asarray(eigenvalues)

    # Cap to available iterations in filter_matrix
    k_expect = int(k_expect)
    if k_expect < 1:
        raise ValueError("k_expect must be >= 1.")
    k_expect = min(k_expect, filter_matrix.shape[0])

    if filter_matrix.shape[1] != eigenvalues.size:
        raise ValueError(
            f"filter_matrix has {filter_matrix.shape[1]} columns, "
            f"but eigenvalues has size {eigenvalues.size}."
        )
    if np.any(eigenvalues == 0):
        raise ValueError("eigenvalues contains zeros; cannot form Lambda^{-1}.")

    G = W.conj().T @ W

    exppropsquared = []
    for k in range(k_expect):
        Dk = np.diag(filter_matrix[k, :] / eigenvalues)
        M = Dk.conj().T @ G @ Dk

        if use_solve:
            val = (eta**2) * np.trace(np.linalg.solve(G, M))
        else:
            val = (eta**2) * np.trace(M @ np.linalg.inv(G))

        exppropsquared.append(np.real_if_close(val))

    exppropnorm = [np.sqrt(v) for v in exppropsquared]

    if make_plot:
        plt.figure(figsize=figsize)
        plt.plot(range(1, k_expect + 1), exppropnorm, marker="o")
        plt.yscale("log")
        plt.xlabel("Iteration $k$")
        plt.title(r"Expected norm of propagated noise $\sqrt{\mathbb{E}\|\bar{ \mathbf{R}}^{\Phi}_k \xi^e\|_2^2}$")
        plt.tight_layout()

        if outpath is not None:
            plt.savefig(outpath, bbox_inches="tight")
            plt.close()
            print("Saved:", outpath)
        else:
            plt.close()

    return exppropsquared, exppropnorm


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_sinograms(
    CT_setup,
    b_exact: np.ndarray,
    b_noisy: np.ndarray,
    *,
    noise: float,
    savepath: str | None = None,
):
    """
    Visualize sinograms (exact vs noisy) from flattened RHS vectors.
    Title becomes: "Sinogram noise = <noise>"
    """

    num_angles = int(CT_setup.num_angles)
    num_dets   = int(CT_setup.num_dets)
    m_expected = num_angles * num_dets

    b_exact = np.asarray(b_exact).reshape(-1)
    b_noisy = np.asarray(b_noisy).reshape(-1)

    if b_exact.size != m_expected or b_noisy.size != m_expected:
        raise ValueError(
            f"Expected b_exact and b_noisy to have length {m_expected} "
            f"(num_angles*num_dets = {num_angles}*{num_dets}), "
            f"got {b_exact.size} and {b_noisy.size}."
        )

    sino_exact = b_exact.reshape(num_angles, num_dets)
    sino_noisy = b_noisy.reshape(num_angles, num_dets)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)

    vmin = float(min(sino_exact.min(), sino_noisy.min()))
    vmax = float(max(sino_exact.max(), sino_noisy.max()))

    im0 = axes[0].imshow(sino_exact, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Exact sinogram")
    axes[0].set_xlabel("Detector index")
    axes[0].set_ylabel("Angle index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(sino_noisy, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Noisy sinogram")
    axes[1].set_xlabel("Detector index")
    axes[1].set_ylabel("Angle index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle(f"Sinogram noise = {noise}", fontsize=14)

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()