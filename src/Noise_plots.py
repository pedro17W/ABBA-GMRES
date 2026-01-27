import numpy as np

def many_noise_total_and_noise_errors(
    CT_setup,
    b_exact,
    X_true,
    run_ba_gmres,
    add_relative_noise,
    iters: int,
    rnl: float,
    n_realizations: int = 10,
    seed0: int = 0,
):
    """
    Computes total error and noise error curves for many noise realizations.

    Returns
    -------
    total_errors : (R, K) array
        total_errors[r, k] = ||x_true - x_k(noisy realization r)||
    noise_errors : (R, K) array
        noise_errors[r, k] = ||x_k(noisy realization r) - x_k(noise-free)||
    X_noisefree : (n, K) array
        noise-free iterates (baseline)
    """

    # ---- Noise-free baseline run (only once)
    X_noisefree, _, _, A, B, W = run_ba_gmres(CT_setup, b_exact, iters)

    if X_noisefree.ndim != 2:
        raise ValueError("Noise-free iterates must be a 2D array of shape (n, K).")

    n, K = X_noisefree.shape

    x_true = np.asarray(X_true).reshape(-1)
    if x_true.shape[0] != n:
        raise ValueError(
            f"X_true has length {x_true.shape[0]} after flattening, but iterates have length {n}."
        )

    total_errors = np.zeros((n_realizations, K), dtype=float)
    noise_errors = np.zeros((n_realizations, K), dtype=float)

    # ---- Loop over realizations
    for r in range(n_realizations):
        b_noisy = add_relative_noise(b_exact, rnl)
        X_noisy, _, _ , A, B, _= run_ba_gmres(CT_setup, b_noisy, iters)

        if X_noisy.ndim != 2:
            raise ValueError("X_noisy must be a 2D array of shape (n, K).")
        if X_noisy.shape != X_noisefree.shape:
            raise ValueError(f"Shape mismatch: X_noisy {X_noisy.shape} vs X_noisefree {X_noisefree.shape}.")

        # total error: ||x_true - x_k||
        total_errors[r, :] = np.array([np.linalg.norm(x_true - X_noisy[:, k]) for k in range(K)])

        # noise error: ||x_k - x̄_k||
        noise_errors[r, :] = np.array([np.linalg.norm(X_noisy[:, k] - X_noisefree[:, k]) for k in range(K)])

    return total_errors, noise_errors, X_noisefree



import matplotlib.pyplot as plt
from pathlib import Path

#We also need the functions to plot the semivoncergence and increaing noise errors

def _maybe_savefig(outpath):
    if outpath is None:
        return
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, format="pdf", bbox_inches="tight")


def plot_iter_total_and_noise(iteration_error, total_errors, noise_errors, title=None, outpath=None, show=False):
    """
    Plot:
      - iteration_error (single curve)
      - all total error curves
      - all noise error curves

    If outpath is provided, saves the plot as a PDF.
    """
    it = np.asarray(iteration_error)
    Tot = np.asarray(total_errors)   # (R,K)
    Noi = np.asarray(noise_errors)   # (R,K)

    K = it.shape[0]
    k = np.arange(1, K + 1)

    plt.figure(figsize=(10, 6))
    #plt.yscale("log")

    # iteration error
    plt.plot(
        k, it,
        color="darkblue", linestyle="-", marker="o", markersize=4,
        label=r"Iteration error $\|x_k^{\mathrm{exact}} - x_{\mathrm{true}}\|$"
    )

    # all noise errors
    for j in range(Noi.shape[0]):
        plt.plot(k, Noi[j, :], color="lightgreen", linestyle="--", alpha=0.35)

    # all total errors
    for j in range(Tot.shape[0]):
        plt.plot(k, Tot[j, :], color="red", linestyle="-.", alpha=0.25)

    # legend entries (avoid duplicates)
    plt.plot([], [], color="lightgreen", linestyle="--",
             label=r"Noise error $\|x_k^{\mathrm{noisy}} - x_k^{\mathrm{exact}}\|$ (all)")
    plt.plot([], [], color="red", linestyle="-.",
             label=r"Total error $\|x_k^{\mathrm{noisy}} - x_{\mathrm{true}}\|$ (all)")

    plt.xlabel("Iteration")
    plt.ylabel("Error norm")
    plt.title(title or "Semiconvergence: iteration, noise, and total errors (all realizations)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    _maybe_savefig(outpath)
    if show:
        plt.show()
    plt.close()


def plot_iter_all_total_and_mean_total(iteration_error, total_errors, title=None, outpath=None, show=False):
    """
    Plot:
      - iteration_error (single curve)
      - all total error curves
      - mean total error curve

    If outpath is provided, saves the plot as a PDF.
    """
    it = np.asarray(iteration_error)
    Tot = np.asarray(total_errors)   # (R,K)

    K = it.shape[0]
    k = np.arange(1, K + 1)

    mean_tot = np.nanmean(Tot, axis=0)

    plt.figure(figsize=(10, 6))
    #plt.yscale("log")

    plt.plot(
        k, it,
        color="darkblue", linestyle="-", marker="o", markersize=4,
        label=r"Iteration error $\|x_k^{\mathrm{exact}} - x_{\mathrm{true}}\|$"
    )

    for j in range(Tot.shape[0]):
        plt.plot(k, Tot[j, :], color="red", linestyle="-.", alpha=0.25)

    plt.plot(
        k, mean_tot,
        color="red", linestyle="-.", linewidth=3,
        label=r"Mean total error $\mathbb{E}\|x_k^{\mathrm{noisy}} - x_{\mathrm{true}}\|$"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Error norm")
    plt.title(title or "Semiconvergence: iteration error + total errors (all + mean)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    _maybe_savefig(outpath)
    if show:
        plt.show()
    plt.close()


def plot_iter_all_noise_and_mean_noise(iteration_error, noise_errors, title=None, outpath=None, show=False):
    """
    Plot:
      - iteration_error (single curve)
      - all noise error curves
      - mean noise error curve

    If outpath is provided, saves the plot as a PDF.
    """
    it = np.asarray(iteration_error)
    Noi = np.asarray(noise_errors)   # (R,K)

    K = it.shape[0]
    k = np.arange(1, K + 1)

    mean_noi = np.nanmean(Noi, axis=0)

    plt.figure(figsize=(10, 6))
    #plt.yscale("log")

    plt.plot(
        k, it,
        color="darkblue", linestyle="-", marker="o", markersize=4,
        label=r"Iteration error $\|x_k^{\mathrm{exact}} - x_{\mathrm{true}}\|$"
    )

    for j in range(Noi.shape[0]):
        plt.plot(k, Noi[j, :], color="lightgreen", linestyle="--", alpha=0.35)

    plt.plot(
        k, mean_noi,
        color="lightgreen", linestyle="--", linewidth=3,
        label=r"Mean noise error $\mathbb{E}\|x_k^{\mathrm{noisy}} - x_k^{\mathrm{exact}}\|$"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Error norm")
    plt.title(title or "Semiconvergence: iteration error + noise errors (all + mean)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    _maybe_savefig(outpath)
    if show:
        plt.show()
    plt.close()
