import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_reconstructions_around_optimal(
    iterations: np.ndarray,
    idx_BA: int,
    num_pixels: int,
    x_true: np.ndarray | None = None,
    before_factor: float = 0.5,
    after_factor: float = 2.0,
    after_offset: int = 0,
    outdir: str = ".",
    prefix: str = "reconstruction",
    scale: str = "truth",  # "truth" | "shared_selected" | "each"
    interpolation: str = "nearest",
):
    """
    Same as before, but with optional shared colormap scaling.

    scale:
      - "truth": use vmin/vmax from x_true (recommended if x_true provided)
      - "shared_selected": vmin/vmax from {x_true (if any), before, optimal, after}
      - "each": default matplotlib behavior (auto-scale each image separately)
    """
    os.makedirs(outdir, exist_ok=True)
    K = iterations.shape[1]

    def _clip(k):
        return max(0, min(int(k), K - 1))

    k_before = _clip(before_factor * idx_BA)
    k_opt    = _clip(idx_BA)
    k_after  = _clip(after_factor * idx_BA + after_offset)

    # Prepare images we will show
    imgs = {}
    imgs["before"] = np.real(iterations[:, k_before].reshape(num_pixels, num_pixels))
    imgs["optimal"] = np.real(iterations[:, k_opt].reshape(num_pixels, num_pixels))
    imgs["after"] = np.real(iterations[:, k_after].reshape(num_pixels, num_pixels))
    if x_true is not None:
        imgs["truth"] = np.real(np.asarray(x_true).reshape(num_pixels, num_pixels))

    # Decide color scaling
    vmin = vmax = None
    if scale == "truth":
        if x_true is None:
            raise ValueError("scale='truth' requires x_true.")
        vmin, vmax = float(imgs["truth"].min()), float(imgs["truth"].max())
    elif scale == "shared_selected":
        stack = np.stack(list(imgs.values()), axis=0)
        vmin, vmax = float(stack.min()), float(stack.max())
    elif scale == "each":
        vmin = vmax = None
    else:
        raise ValueError("scale must be one of: 'truth', 'shared_selected', 'each'.")

    # -------- Ground truth plot --------
    if x_true is not None:
        plt.figure(figsize=(4, 4))
        plt.imshow(imgs["truth"], cmap="gray", vmin=vmin, vmax=vmax, interpolation=interpolation)
        plt.axis("off")
        plt.title("Ground truth")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_ground_truth.pdf"), bbox_inches="tight")
        plt.close()

    # -------- Reconstructions --------
    cases = [
        ("before", k_before, f"Reconstruction before optimal — iteration {k_before}"),
        ("optimal", k_opt,   f"Optimal reconstruction — iteration {k_opt}"),
        ("after",  k_after,  f"Reconstruction after optimal — iteration {k_after}"),
    ]

    for tag, k, title in cases:
        plt.figure(figsize=(4, 4))
        plt.imshow(imgs[tag], cmap="gray", vmin=vmin, vmax=vmax, interpolation=interpolation)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_{tag}.pdf"), bbox_inches="tight")
        plt.close()

    return {"k_before": k_before, "k_opt": k_opt, "k_after": k_after, "vmin": vmin, "vmax": vmax}


#################################################################################################
################---------------THE NEW FUNCTION THAT DOES EVERYTING-------------------###########
#################################################################################################

def semiconvergence_from_iterates(
    X_noisy: np.ndarray,
    X_noisefree: np.ndarray,
    X_true: np.ndarray,
    outpath: str,
    title: str = "Convergence History",
    compute_noise_error: bool = True,
    noise_error: np.ndarray | None = None,
):
    """
    Combines:
      - compute_rel_errors() for noisy and noise-free iterates
      - plot_semiconvergence()

    Parameters
    ----------
    X_noisy : (n, K) array
        BA-GMRES iterates from noisy RHS (columns are iterates x_k).
    X_noisefree : (n, K) array
        BA-GMRES iterates from noise-free RHS (columns are iterates \bar{x}_k).
    X_true : array
        Ground truth image (2D) or vector (1D). Will be flattened.
    outpath : str
        Output PDF path.
    title : str
        Figure title.
    compute_noise_error : bool
        If True, compute noise error ||x_k - \bar{x}_k|| from iterates.
    noise_error : (K,) array or None
        If provided, uses this instead of computing (even if compute_noise_error=True).

    Returns
    -------
    out : dict
        Contains curves + minima indices/values:
        - relative_errors, idx_BA, val_BA
        - noisefree_rel_errors, noisefree_idx_BA, noisefree_val_BA
        - noise_error
    """

    import os
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    # --- Basic shape checks
    if X_noisy.ndim != 2 or X_noisefree.ndim != 2:
        raise ValueError("X_noisy and X_noisefree must be 2D arrays of shape (n, K).")
    if X_noisy.shape != X_noisefree.shape:
        raise ValueError(f"Shape mismatch: X_noisy {X_noisy.shape} vs X_noisefree {X_noisefree.shape}.")

    n, K = X_noisy.shape
    x_true = np.asarray(X_true).reshape(-1)
    if x_true.shape[0] != n:
        raise ValueError(f"X_true has length {x_true.shape[0]} after flattening, but iterates have length {n}.")

    # --- Compute relative errors (noisy)
    relative_errors = np.array([np.linalg.norm(x_true - X_noisy[:, k]) for k in range(K)])
    val_BA = float(np.min(relative_errors))
    idx_BA = int(np.argmin(relative_errors))

    # --- Compute relative errors (noise-free)
    noisefree_rel_errors = np.array([np.linalg.norm(x_true - X_noisefree[:, k]) for k in range(K)])
    noisefree_val_BA = float(np.min(noisefree_rel_errors))
    noisefree_idx_BA = int(np.argmin(noisefree_rel_errors))

    # --- Noise error
    if noise_error is None:
        if compute_noise_error:
            noise_error = np.array([np.linalg.norm(X_noisy[:, k] - X_noisefree[:, k]) for k in range(K)])
        else:
            noise_error = np.zeros(K)
    else:
        noise_error = np.asarray(noise_error)
        if noise_error.shape[0] != K:
            raise ValueError(f"noise_error has length {noise_error.shape[0]}, expected {K}.")

    # --- Plot
    ks = np.arange(K)

    plt.figure()
    plt.plot(ks, relative_errors, 'k-', linewidth=2)
    plt.plot(idx_BA, val_BA, 'k*', markersize=10)

    plt.plot(ks, noisefree_rel_errors, 'b--', linewidth=2)
    plt.plot(noisefree_idx_BA, noisefree_val_BA, 'b*', markersize=10)

    plt.plot(ks, noise_error, 'r-.', linewidth=2)

    plt.title(title, fontname='cmr10', fontsize=16)
    plt.xlabel('Iteration $k$', fontname='cmr10', fontsize=16)
    plt.ylabel('Norm of error', fontname='cmr10', fontsize=16)

    plt.legend([
        'BA-GMRES (total error)',
        f'noisy min: k={idx_BA}, err={val_BA:.2e}',
        'BA-GMRES (iteration error)',
        f'noise-free min: k={noisefree_idx_BA}, err={noisefree_val_BA:.2e}',
        r'Noise error $\|x_k - \bar{x}_k\|$'
    ])

    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()

    return {
        "relative_errors": relative_errors,
        "idx_BA": idx_BA,
        "val_BA": val_BA,
        "noisefree_rel_errors": noisefree_rel_errors,
        "noisefree_idx_BA": noisefree_idx_BA,
        "noisefree_val_BA": noisefree_val_BA,
        "noise_error": noise_error,
    }



def plot_eigs_complex_plane(
    eigenvalues,
    outpath="figs/eigs_BA.pdf",
    title="Eigenvalues of BA",
    plot_abs=True,
    abs_outpath=None,
    abs_title=None
):
    """
    Plot eigenvalues in the complex plane and (optionally) also plot their absolute values.

    Parameters
    ----------
    eigenvalues : array_like
        Eigenvalues to plot.
    outpath : str
        PDF path for the complex-plane scatter plot.
    title : str
        Title for the complex-plane plot.
    plot_abs : bool, default True
        If True, also create a second plot of |eigenvalues|.
    abs_outpath : str or None
        PDF path for the |eigenvalues| plot. If None, it is derived from outpath by appending "_abs".
    abs_title : str or None
        Title for the |eigenvalues| plot. If None, derived from `title`.
    """
    ev = np.asarray(eigenvalues).reshape(-1)

    # ---- 1) Complex plane plot ----
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(np.real(ev), np.imag(ev), s=25)
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Real part")
    plt.ylabel("Imag part")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- 2) Absolute values plot (optional) ----
    if plot_abs:
        if abs_outpath is None:
            base, ext = os.path.splitext(outpath)
            abs_outpath = f"{base}_abs{ext}"
        if abs_title is None:
            abs_title = title + r" (magnitudes $|\lambda|$)"

        mags = np.abs(ev)
        idx = np.argsort(mags)[::-1]  # sort descending, usually nicer
        mags_sorted = mags[idx]

        fig2 = plt.figure(figsize=(5.2, 3.6))
        plt.semilogy(np.arange(1, len(mags_sorted) + 1), mags_sorted, "o-", markersize=3.5, linewidth=1.4)
        plt.xlabel("Index (sorted by magnitude)")
        plt.ylabel(r"$|\lambda|$ (log scale)")
        plt.title(abs_title)
        plt.grid(True, which="both", linestyle=":")
        plt.tight_layout()

        os.makedirs(os.path.dirname(abs_outpath) or ".", exist_ok=True)
        fig2.savefig(abs_outpath, format="pdf", bbox_inches="tight")
        plt.close(fig2)
