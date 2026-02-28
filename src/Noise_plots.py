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

import os
import numpy as np
import matplotlib.pyplot as plt


def _maybe_savefig(outpath: str | None):
    """Save figure if outpath is provided (creates parent dirs)."""
    if outpath is None:
        return
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")


import os
import numpy as np
import matplotlib.pyplot as plt


def _maybe_savefig(outpath: str | None):
    """Save figure if outpath is provided (creates parent dirs)."""
    if outpath is None:
        return
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")


def _slice_first_n(it, Tot=None, Noi=None, n=200):
    """Slice arrays to first n iterations (safe if K < n). If n is None, keep all."""
    it = np.asarray(it)
    K = it.shape[0]

    if n is None:
        Kp = K
    else:
        Kp = min(K, int(n))

    it = it[:Kp]
    if Tot is not None:
        Tot = np.asarray(Tot)[:, :Kp]
    if Noi is not None:
        Noi = np.asarray(Noi)[:, :Kp]
    k = np.arange(1, Kp + 1)
    return k, it, Tot, Noi


def plot_iter_total_and_noise(iteration_error, total_errors, noise_errors,
                              title=None, outpath=None, show=False, n_iter=None):
    """
    Plot (linear scale):
      - iteration_error (single curve)
      - all total error curves
      - all noise error curves

    If n_iter is None: plot all iterations.
    """
    k, it, Tot, Noi = _slice_first_n(iteration_error, Tot=total_errors, Noi=noise_errors, n=n_iter)

    plt.figure(figsize=(10, 6))

    plt.plot(
        k, it,
        color="darkblue", linestyle="-", marker="o", markersize=4,
        linewidth=3.0, alpha=0.95, zorder=6,
        label=r"Iteration error $\|\bar{x}_k - \bar{x}\|$"
    )

    for j in range(Noi.shape[0]):
        plt.plot(
            k, Noi[j, :],
            color="lightgreen", linestyle="--",
            linewidth=0.6, alpha=0.7, zorder=2
        )

    for j in range(Tot.shape[0]):
        plt.plot(
            k, Tot[j, :],
            color="red", linestyle="-.",
            linewidth=0.6, alpha=0.4, zorder=1
        )

    plt.plot([], [], color="lightgreen", linestyle="--", linewidth=1.6,
             label=r"Noise error $\|x_k - \bar{x}_k\|$ (all)")
    plt.plot([], [], color="red", linestyle="-.", linewidth=1.6,
             label=r"Total error $\|x_k - \bar{x}\|$ (all)")

    plt.xlabel("Iteration")
    plt.ylabel("Error norm")
    plt.title(title or f"Semiconvergence (first {len(k) - 1} iterations): iteration, noise, and total errors")
    plt.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.8)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    plt.yscale('log')
    plt.minorticks_on()
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    _maybe_savefig(outpath)
    if show:
        plt.show()
    plt.close()


def plot_iter_all_total_and_mean_total(iteration_error, total_errors,
                                       title=None, outpath=None, show=False, n_iter=None):
    """
    Plot (linear scale):
      - iteration_error (single curve)
      - all total error curves
      - mean total error curve

    If n_iter is None: plot all iterations.
    """
    k, it, Tot, _ = _slice_first_n(iteration_error, Tot=total_errors, Noi=None, n=n_iter)
    mean_tot = np.nanmean(Tot, axis=0)

    plt.figure(figsize=(10, 6))

    plt.plot(
        k, it,
        color="darkblue", linestyle="-", marker="o", markersize=4,
        linewidth=2.6, alpha=0.90, zorder=4,
        label=r"Iteration error $\|\bar{x}_k - \bar{x}\|$"
    )

    for j in range(Tot.shape[0]):
        plt.plot(
            k, Tot[j, :],
            color="red", linestyle="-.",
            linewidth=0.6, alpha=0.8, zorder=1
        )

    plt.plot(
        k, mean_tot,
        color="maroon", linestyle="-.", linewidth=3, alpha=1.0, zorder=6,
        label=r"Mean total error $\mathbb{E}\|x_k - \bar{x}\|$"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Error norm")
    plt.title(title or f"Semiconvergence (first {len(k) - 1} iterations): iteration error + total errors (all + mean)")
    plt.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.8)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    plt.yscale('log')
    plt.minorticks_on()
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    _maybe_savefig(outpath)
    if show:
        plt.show()
    plt.close()


def plot_iter_all_noise_and_mean_noise(iteration_error, noise_errors,
                                       title=None, outpath=None, show=False, n_iter=None):
    """
    Plot (linear scale):
      - iteration_error (single curve)
      - all noise error curves
      - mean noise error curve

    If n_iter is None: plot all iterations.
    """
    k, it, _, Noi = _slice_first_n(iteration_error, Tot=None, Noi=noise_errors, n=n_iter)
    mean_noi = np.nanmean(Noi, axis=0)

    plt.figure(figsize=(10, 6))

    plt.plot(
        k, it,
        color="darkblue", linestyle="-", marker="o", markersize=4,
        linewidth=2.6, alpha=0.90, zorder=4,
        label=r"Iteration error $\|\bar{x}_k - \bar{x}\|$"
    )

    for j in range(Noi.shape[0]):
        plt.plot(
            k, Noi[j, :],
            color="lightgreen", linestyle="--",
            linewidth=0.8, alpha=0.70, zorder=1
        )

    plt.plot(
        k, mean_noi,
        color="green", linestyle="--", linewidth=3, alpha=1.0, zorder=6,
        label=r"Mean noise error $\mathbb{E}\|x_k - \bar{x}_k\|$"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Error norm")
    plt.title(title or f"Semiconvergence (first {len(k) - 1} iterations): iteration error + noise errors (all + mean)")
    plt.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.8)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    plt.yscale('log')
    plt.minorticks_on()
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    _maybe_savefig(outpath)
    if show:
        plt.show()
    plt.close()