import os
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_residual_poly_raw_save_pdf(
    residual_polynomials,
    eigenvalues,
    k=1,
    outdir="results/analysis_fullBA/residualpoly",
    prefix="BA_full",
    num_eigs_show=None,
    pad=0.01,
):
    """
    Raw plot of residual polynomial p_k over a real x-grid, with eigenvalues (real parts)
    marked on the real axis. Saves as PDF.
    """
    os.makedirs(outdir, exist_ok=True)

    if k < 1 or k > len(residual_polynomials):
        raise ValueError(f"k={k} out of range. Have {len(residual_polynomials)} polynomials.")

    p = residual_polynomials[k - 1]
    eigenvalues = np.asarray(eigenvalues)

    eig_real = np.real(eigenvalues)
    eig_real_plot = eig_real[: int(num_eigs_show)] if num_eigs_show is not None else eig_real

    lam_min = np.min(eig_real) - pad
    lam_max = np.max(eig_real) + pad

    x = np.linspace(lam_min, lam_max, 1200)
    y = p(x)
    y_plot = np.real(y)  # keep visualization consistent with "discard imag parts"

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, y_plot, linewidth=1.6, label=rf"$\Re(p_{{{k}}}(\lambda))$")

    # Eigenvalues as small, semi-transparent red x's
    plt.scatter(
        eig_real_plot,
        np.zeros_like(eig_real_plot),
        marker="x",
        s=12,
        linewidths=1.0,
        alpha=0.35,
        color="red",
        label="Eigenvalues (Re)",
        zorder=2,
    )

    # Textbox: are all roots (of p_k) real?
    imag_tol = 1e-12
    roots = np.asarray(p.roots())
    roots_are_almost_real = bool(np.all(np.abs(np.imag(roots)) <= imag_tol))

    ax = plt.gca()
    ax.text(
        0.7, 0.2,
        f"roots ~ real: {roots_are_almost_real}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        zorder=5,
    )

    plt.axhline(0, linewidth=0.9)
    plt.title(rf"Residual polynomial $p_{{{k}}}(\lambda)$ with eigenvalues")
    plt.xlabel(r"$\Re(\lambda)$")
    plt.ylabel(rf"$\Re(p_{{{k}}}(\lambda))$")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{prefix}_respoly_raw_k{k}.pdf")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    return outpath


def plot_residual_poly_zoom_save_pdf(
    residual_polynomials,
    eigenvalues,
    k=1,
    outdir="results/analysis_fullBA/residualpoly",
    prefix="BA_full",
    num_eigs_show=None,
    ylims=(-1.0, 1.0),
    pad=0.01,
):
    """
    Zoomed version (default y in [-1,1]) of residual polynomial p_k over a real x-grid,
    with eigenvalues (real parts) marked on the real axis. Saves as PDF.
    """
    os.makedirs(outdir, exist_ok=True)

    if k < 1 or k > len(residual_polynomials):
        raise ValueError(f"k={k} out of range. Have {len(residual_polynomials)} polynomials.")

    p = residual_polynomials[k - 1]
    eigenvalues = np.asarray(eigenvalues)

    eig_real = np.real(eigenvalues)
    eig_real_plot = eig_real[: int(num_eigs_show)] if num_eigs_show is not None else eig_real

    lam_min = np.min(eig_real) - pad
    lam_max = np.max(eig_real) + pad

    x = np.linspace(lam_min, lam_max, 1200)
    y = p(x)
    y_plot = np.real(y)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, y_plot, linewidth=1.6, label=rf"$\Re(p_{{{k}}}(\lambda))$")
    plt.ylim(ylims)

    # Eigenvalues as small, semi-transparent red x's
    plt.scatter(
        eig_real_plot,
        np.zeros_like(eig_real_plot),
        marker="x",
        s=12,
        linewidths=1.0,
        alpha=0.35,
        color="red",
        label="Eigenvalues (Re)",
        zorder=2,
    )

    # Textbox: roots real? and p_k at lambda_max
    imag_tol = 1e-12
    roots = np.asarray(p.roots())
    roots_are_almost_real = bool(np.all(np.abs(np.imag(roots)) <= imag_tol))

    lam_max_idx = int(np.argmax(np.abs(eigenvalues)))
    lam_max = eigenvalues[lam_max_idx]
    p_at_lam_max = p(lam_max)

    ax = plt.gca()
    ax.text(
        0.07, 0.2,
        f"roots ~ real: {roots_are_almost_real}\n"
        rf"$p_{{{k}}}(\lambda_{{max}})$ = {p_at_lam_max:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        zorder=5,
    )

    plt.axhline(0, linewidth=0.9)
    plt.title(rf"Residual polynomial $p_{{{k}}}(\lambda)$ with eigenvalues (zoomed)")
    plt.xlabel(r"$\Re(\lambda)$")
    plt.ylabel(rf"$\Re(p_{{{k}}}(\lambda))$")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{prefix}_respoly_zoom_k{k}.pdf")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    return outpath


######################--------------------------------------------------------------------###################
######################-----------------------------NOW THE FILTER FACTORS-----------------###################
######################--------------------------------------------------------------------###################
def compute_filter_factors_fullBA(eigenvalues, residual_polynomials, i_max=50):
    """
    Compute filter factors for ALL available residual polynomials:
        phi_i^(k) = 1 - p_k(lambda_i)

    Parameters
    ----------
    eigenvalues : array_like (n,)
        Eigenvalues of BA_full (complex allowed). Assumed already sorted if you want that convention.
    residual_polynomials : list of callables / numpy.polynomial.polynomial.Polynomial
        residual_polynomials[k] corresponds to p_{k+1} in 1-based notation.
    i_max : int
        Number of eigenvalues to include (first i_max).

    Returns
    -------
    phi : ndarray, shape (K, I), complex
        phi[k, i] = 1 - p_{k+1}(lambda_i)  (1-based: row k corresponds to iteration k+1)
        where K = len(residual_polynomials), I = min(i_max, len(eigenvalues)).
    """
    eigenvalues = np.asarray(eigenvalues)
    K = len(residual_polynomials)
    I = int(min(i_max, len(eigenvalues)))

    phi = np.zeros((K, I), dtype=complex)
    lam = eigenvalues[:I]

    for k in range(K):
        p_k = residual_polynomials[k]
        phi[k, :] = 1 - p_k(lam)

    return phi


def save_filter_factor_plots_fullBA(
    phi_noisefree,
    phi_noisy=None,
    outdir="results/analysis_fullBA/filter_factors",
    prefix="BA_full",
    use_abs=True,
    overlay_noisy=False,
    labels=("noise-free", "noisy"),
):
    """
    Saves one PDF per iteration k showing filter factors vs eigenvalue index i.
    Optionally overlays noisy and noise-free in the same plot.

    Parameters
    ----------
    phi_noisefree : ndarray, shape (K, I), complex
        Noise-free filter factors.
    phi_noisy : ndarray or None, shape (K, I), complex
        Noisy filter factors. Required if overlay_noisy=True.
    outdir : str
        Folder to save PDFs.
    prefix : str
        Filename prefix for outputs.
    use_abs : bool
        If True plot |phi|. If False plot Re(phi).
    overlay_noisy : bool
        If True, overlay phi_noisy on the same plot as phi_noisefree.
    labels : tuple(str, str)
        Legend labels for (noise-free, noisy).
    """
    os.makedirs(outdir, exist_ok=True)

    phi_nf = np.asarray(phi_noisefree)
    K, I = phi_nf.shape
    x = np.arange(1, I + 1)

    if overlay_noisy:
        if phi_noisy is None:
            raise ValueError("overlay_noisy=True but phi_noisy is None.")
        phi_ny = np.asarray(phi_noisy)
        if phi_ny.shape != phi_nf.shape:
            raise ValueError(f"phi_noisy shape {phi_ny.shape} must match phi_noisefree shape {phi_nf.shape}.")

    for k in range(1, K + 1):
        row_nf = phi_nf[k - 1, :]
        y_nf = np.abs(row_nf) if use_abs else np.real(row_nf)

        plt.figure(figsize=(6.2, 4.0))
        plt.plot(x, y_nf, marker="o", markersize=3, linewidth=1.4, label=labels[0])

        if overlay_noisy:
            row_ny = phi_ny[k - 1, :]
            y_ny = np.abs(row_ny) if use_abs else np.real(row_ny)
            plt.plot(x, y_ny, marker="x", markersize=3, linewidth=1.2, linestyle="--", label=labels[1])

        plt.title(f"Filter factors at iteration k = {k}")
        plt.xlabel("Eigenvalue index i")
        plt.ylabel(r"$|\phi_i^{(k)}| = |1 - p_k(\lambda_i)|$" if use_abs else r"$\Re(\phi_i^{(k)})$")
        plt.grid(True, which="both", linestyle=":", alpha=0.5)
        if overlay_noisy:
            plt.legend(loc="best")
        plt.tight_layout()

        outpath = os.path.join(outdir, f"{prefix}_filter_factors_k{k}.pdf")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()



########-----------------------------------------------------------------------------------------##########
########------------------------And now the spectral/iteration error plot------------------------##########
########-----------------------------------------------------------------------------------------##########

def plot_spectral_weights_vs_iteration_error_save_pdf(
    eigenvalues,
    xi,
    iteration_errors,
    outdir="results/analysis_fullBA/figs",
    prefix="BA_full",
    use_abs=True,
    i_max=None,
):
    """
    Compare spectral weights |xi_i| / |lambda_i| with GMRES iteration error curve.
    Saves as a PDF.

    Parameters
    ----------
    eigenvalues : array_like (n,)
        Eigenvalues of BA_full (complex ok). Should be sorted consistently with xi.
    xi : array_like (n,)
        Eigenbasis coefficients (e.g., xi = W^{-1} Bb), aligned with eigenvalues.
    iteration_errors : array_like (m,)
        Norms ||x_k - x_true|| for GMRES iterates (noise-free or noisy).
    outdir : str
        Output directory for the PDF.
    prefix : str
        Filename prefix.
    use_abs : bool
        If True uses |xi|/|lambda|. If False uses Re(xi)/Re(lambda) with safe handling.
    i_max : int or None
        Plot only the first i_max spectral components/eigenvalues (default: all).
    """
    os.makedirs(outdir, exist_ok=True)

    eigenvalues = np.asarray(eigenvalues)
    xi = np.asarray(xi).reshape(-1)
    iteration_errors = np.asarray(iteration_errors).reshape(-1)

    n = min(len(eigenvalues), len(xi))
    eigenvalues = eigenvalues[:n]
    xi = xi[:n]

    if i_max is not None:
        n_plot = int(min(i_max, n))
        eigenvalues = eigenvalues[:n_plot]
        xi = xi[:n_plot]

    # Spectral weights
    eps = 1e-15
    if use_abs:
        denom = np.maximum(np.abs(eigenvalues), eps)
        component_spectral = np.abs(xi) / denom
        ylab = r"$|\xi_i| / |\lambda_i|$"
    else:
        denom = np.real(eigenvalues)
        denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)
        component_spectral = np.real(xi) / denom
        ylab = r"$\Re(\xi_i) / \Re(\lambda_i)$"

    # Plot
    plt.figure(figsize=(7.2, 4.6))

    plt.plot(
        component_spectral[:len(iteration_errors) + 1],
        label=ylab,
        linewidth=2,
        marker="o",
        markersize=3,
    )

    plt.plot(
        iteration_errors,
        label=r"$\|x_k - x_{\mathrm{true}}\|$",
        linewidth=2,
        marker="s",
        markersize=3,
    )

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.xlabel("Index (spectral component / iteration)")
    plt.ylabel("Magnitude (log scale)")
    plt.title("Eigenbasis weights vs GMRES iteration error")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{prefix}_spectralweights_vs_itererr.pdf")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    return outpath
