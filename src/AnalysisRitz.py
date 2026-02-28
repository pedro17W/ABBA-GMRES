import numpy as np
from scipy.linalg import eig
from numpy.polynomial import Polynomial
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#####################---------------These are the functions used in the small test problems----------------##############
####################---------------------------------------------------------------------------------------##############
def compute_harmonic_ritz(H_rect, H_square):
    lhs = H_rect.conj().T @ H_rect
    rhs = H_square.conj().T

    thetas_all, Ys_all = eig(lhs, rhs) 
    return thetas_all

def residual_polynomial(Harmonic: list):
    """Return normalized GMRES residual polynomial with p(0)=1."""

    
    p = Polynomial.fromroots(Harmonic)
    p /= p(0)

    #num_points = int((max(Harmonic) - min(Harmonic)) * 100)
    #x = np.linspace(min(Harmonic) - 0.1, max(Harmonic) + 0.1, num_points)
    #y = p(x)

    return p

###############################------------------------------------#########################################################
############################################################################################################################


def ba_harmonic_ritz_values(H_rect_list, H_square_list, k_max=None):
    """
    Args: 
    H_rect_list, H_square_list: lists of the rectangular and square Hessenberg matrices
    k_max: int: the maximum number of Hessenberg matrices for which want harmonic Ritz values

    Returns a list: ritz[k] = harmonic Ritz values at iteration k (0-based list index).
    """
    if k_max is None:
        k_max = min(len(H_rect_list), len(H_square_list))
    k_max = min(k_max, len(H_rect_list), len(H_square_list))

    harmonic_ritz = []
    for k in range(k_max):
        harmonic_ritz.append(compute_harmonic_ritz(H_rect_list[k], H_square_list[k]))
    return harmonic_ritz

def ba_residual_polynomials(ritz_values, num_polys=10):
    """
    Returns list of residual polynomials built from the first num_polys Ritz sets.
    """
    m = min(num_polys, len(ritz_values))
    return [residual_polynomial(ritz_values[i]) for i in range(m)]



#####We also need the functions that can plot these harmonic RItz values and the corresponding reidual polynomials####
import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_harmonic_ritz_complex(harmonic_ritz_values, iteration=1, outpath=None, eigenvalues=None):
    k = iteration
    row = np.asarray(harmonic_ritz_values[k - 1])

    # Optional eigenvalues
    eig = None
    if eigenvalues is not None:
        eig = np.asarray(eigenvalues).reshape(-1)
        if eig.size == 0:
            eig = None

    # --- Create figure/axes explicitly (so we can place text in the figure margin) ---
    fig, ax = plt.subplots(figsize=(5.4, 4))

    # Scatter plot: harmonic Ritz values
    ax.scatter(
        np.real(row), np.imag(row),
        label="Harmonic Ritz values",
        color="blue", marker="o", s=40
    )

    # Scatter plot: eigenvalues (optional)
    if eig is not None:
        ax.scatter(
            np.real(eig), np.imag(eig),
            label="Eigenvalues (eigs)",
            color="red", marker="x", s=55
        )

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title(
        f"Harmonic Ritz values (iteration {k})\n"
        f"Generalized eigenproblem $H({k+1},{k})$"
    )
    ax.grid(True)

    # y-limits include both, if eigenvalues exist
    imag_all = np.imag(row)
    if eig is not None:
        imag_all = np.concatenate([imag_all, np.imag(eig)])
    pad = 2e-2
    if imag_all.size > 0:
        ax.set_ylim(np.min(imag_all) - pad, np.max(imag_all) + pad)

    ax.legend(loc="best", fontsize=9, frameon=True)

    # ---- Build diagnostics text (goes BELOW the axes) ----
    imag_tol = 1e-12
    imag_parts = np.abs(np.imag(row))
    if np.any(imag_parts > imag_tol):
        diag_lines = [
            "Complex harmonic Ritz values detected",
            rf"max |Im(θ_i)| = {np.max(imag_parts):.2e}",
        ]
    else:
        diag_lines = ["All harmonic Ritz values real"]

    # Pairwise top-m diffs by |.| (optional)
    if eig is not None:
        m = 5
        idx_e = np.argsort(np.abs(eig))[::-1]
        idx_r = np.argsort(np.abs(row))[::-1]
        top_e = eig[idx_e[:m]]
        top_r = row[idx_r[:m]]

        # FIX: compare only as many as exist in BOTH
        m_eff = min(m, top_e.size, top_r.size)

        diag_lines += ["", f"Top-{m_eff} by |.| pairing (sorted by magnitude)"]
        if m_eff == 0:
            diag_lines.append("Not enough values to compare at this iteration.")
        else:
            diffs = np.abs(top_e[:m_eff] - top_r[:m_eff])
            for i in range(m_eff):
                diag_lines.append(
                    f"{i+1}: |λ-θ|={diffs[i]:.2e}   |λ|={np.abs(top_e[i]):.2e}   |θ|={np.abs(top_r[i]):.2e}"
                )

    # Put diagnostics below plot
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.02, 0.02, "\n".join(diag_lines), fontsize=8, va="bottom", family="monospace")

    # Save/show
    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight")
    if show := False:  # keep your signature if you want; otherwise remove this line
        plt.show()
    plt.close(fig)




import numpy as np
import matplotlib.pyplot as plt


def plot_harmonicandnormal_ritz_complex(harmonic_ritz_values, iteration, outpath=None,
                                        eigenvalues=None, normal_ritz_values=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if iteration < 1:
        raise ValueError("iteration must be >= 1")

    idx = iteration - 1  # iteration k -> Python index k-1

    # Harmonic Ritz (shifted)
    hvals = np.asarray(harmonic_ritz_values[idx], dtype=complex).ravel()

    # Normal Ritz (also shifted)
    if normal_ritz_values is not None:
        nvals = np.asarray(normal_ritz_values[idx], dtype=complex).ravel()
    else:
        nvals = np.array([], dtype=complex)

    # Eigenvalues (global)
    if eigenvalues is not None:
        evals = np.asarray(eigenvalues, dtype=complex).ravel()
    else:
        evals = np.array([], dtype=complex)

    plt.figure(figsize=(9, 3))

    # Eigenvalues: red x, more transparent
    if evals.size > 0:
        plt.scatter(
            evals.real, evals.imag,
            marker='x', c='red', alpha=0.30, s=35, linewidths=1.0,
            label='Eigenvalues'
        )

    # Normal Ritz: green open circles, a bit bigger
    if nvals.size > 0:
        plt.scatter(
            nvals.real, nvals.imag,
            marker='o', facecolors='none', edgecolors='lightgreen',
            s=90, linewidths=2.0,
            label='Normal Ritz'
        )

    # Harmonic Ritz: blue open circles, a bit bigger
    if hvals.size > 0:
        plt.scatter(
            hvals.real, hvals.imag,
            marker='o', facecolors='none', edgecolors='blue',
            s=70, linewidths=1.8,
            label='Harmonic Ritz'
        )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.title(f"Ritz values at iteration k={iteration}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compute_normal_ritz_values(hessenberg_list):
    normal_ritz = []
    for H in hessenberg_list:
        if H is None:
            normal_ritz.append(np.array([], dtype=complex))
        else:
            H = np.asarray(H)
            if H.size == 0:
                normal_ritz.append(np.array([], dtype=complex))
            else:
                normal_ritz.append(np.linalg.eigvals(H))
    return normal_ritz


import numpy as np
import matplotlib.pyplot as plt

def plot_harmonic_ritz_complex_noisys(
    ritz_noisefree,
    iteration=1,
    outpath=None,
    ritz_noisy=None,
    x_zoom=None,          # e.g. (0, 6000) or (-1, 1); None means no zoom
    y_zoom=None,          # optional; same idea
    imag_tol=1e-12,
):
    """
    Plot harmonic Ritz values in the complex plane at a given iteration.

    Parameters
    ----------
    ritz_noisefree : list/array-like
        List of arrays; ritz_noisefree[k-1] contains the harmonic Ritz values at iteration k.
    iteration : int
        1-based iteration index k.
    outpath : str or None
        Where to save PDF/PNG. If None, does not save.
    ritz_noisy : list/array-like or None
        Same structure as ritz_noisefree, for the noisy run. If None, only noisefree is plotted.
    x_zoom : tuple(float, float) or None
        If given, set x-limits to (xmin, xmax).
    y_zoom : tuple(float, float) or None
        If given, set y-limits to (ymin, ymax).
    imag_tol : float
        Tolerance for declaring values "real".
    """
    k = int(iteration)

    row_nf = np.asarray(ritz_noisefree[k - 1]).reshape(-1)

    row_no = None
    if ritz_noisy is not None:
        row_no = np.asarray(ritz_noisy[k - 1]).reshape(-1)
        if row_no.size == 0:
            row_no = None

    fig, ax = plt.subplots(figsize=(5.4, 4))

    # Noise-free Ritz values
    ax.scatter(
        np.real(row_nf), np.imag(row_nf),
        label="Harmonic Ritz (noisefree)",
        color="blue", marker="o", s=40
    )

    # Noisy Ritz values (optional)
    if row_no is not None:
        ax.scatter(
            np.real(row_no), np.imag(row_no),
            label="Harmonic Ritz (noisy)",
            color="red", marker="x", s=55
        )

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title(
        f"Harmonic Ritz values (iteration {k})\n"
        f"Generalized eigenproblem $H({k+1},{k})$"
    )
    ax.grid(True)
    ax.legend(loc="best", fontsize=9, frameon=True)

    # Auto y-lims unless user zooms y
    if y_zoom is not None:
        ax.set_ylim(y_zoom)
    else:
        imag_all = np.imag(row_nf)
        if row_no is not None:
            imag_all = np.concatenate([imag_all, np.imag(row_no)])
        pad = 2e-2
        if imag_all.size > 0:
            ax.set_ylim(np.min(imag_all) - pad, np.max(imag_all) + pad)

    # Optional x zoom
    if x_zoom is not None:
        ax.set_xlim(x_zoom)

    # ---- Diagnostics text ----
    diag_lines = []

    # noisefree: real/complex
    imag_nf = np.abs(np.imag(row_nf))
    if np.any(imag_nf > imag_tol):
        diag_lines += [
            "Noisefree: complex harmonic Ritz detected",
            rf"max |Im(θ_i)| = {np.max(imag_nf):.2e}",
        ]
    else:
        diag_lines += ["Noisefree: all harmonic Ritz values real"]

    # noisy: real/complex
    if row_no is not None:
        imag_no = np.abs(np.imag(row_no))
        if np.any(imag_no > imag_tol):
            diag_lines += [
                "",
                "Noisy: complex harmonic Ritz detected",
                rf"max |Im(\tilde{{θ}}_i)| = {np.max(imag_no):.2e}",
            ]
        else:
            diag_lines += ["", "Noisy: all harmonic Ritz values real"]

        # Pairing diagnostics between noisefree and noisy (top-m by magnitude)
        m = 5
        idx_nf = np.argsort(np.abs(row_nf))[::-1]
        idx_no = np.argsort(np.abs(row_no))[::-1]
        top_nf = row_nf[idx_nf[:m]]
        top_no = row_no[idx_no[:m]]
        m_eff = min(m, top_nf.size, top_no.size)

        diag_lines += ["", f"Top-{m_eff} pairing noisefree vs noisy (sorted by |.|)"]
        if m_eff == 0:
            diag_lines.append("Not enough values to compare at this iteration.")
        else:
            diffs = np.abs(top_nf[:m_eff] - top_no[:m_eff])
            for i in range(m_eff):
                diag_lines.append(
                    f"{i+1}: |θ-θ~|={diffs[i]:.2e}   |θ|={np.abs(top_nf[i]):.2e}   |θ~|={np.abs(top_no[i]):.2e}"
                )

    # Place diagnostics below plot
    fig.subplots_adjust(bottom=0.30)
    fig.text(0.02, 0.02, "\n".join(diag_lines), fontsize=8, va="bottom", family="monospace")

    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)



import os
import numpy as np
import matplotlib.pyplot as plt

def plot_residual_poly_raw(residual_polynomials, eigenvalues, k=1, outpath=None, pad=0.01, num_eigs_show=None):
    """
    Raw polynomial plot \bar{p}_k(λ) on real axis, with eigenvalues (real parts)
    marked on x-axis. Saves exactly to outpath (unchanged behavior).
    """
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

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, y, linewidth=1.6, label=rf"$\bar{{p}}_{{{k}}}(\lambda)$")

    # Eigenvalues as small, semi-transparent red x's
    plt.scatter(
        eig_real_plot,
        np.zeros_like(eig_real_plot),
        marker="x",
        s=25,
        linewidths=1.0,
        alpha=0.6,
        color="red",
        label="Eigenvalues",
        zorder=2,
    )

    # Diagnostic: are roots (of p_k) almost real?
    imag_tol = 1e-12
    roots = np.asarray(p.roots())
    roots_are_almost_real = bool(np.all(np.abs(np.imag(roots)) <= imag_tol))

    #ax = plt.gca()
    #ax.text(
        #0.7, 0.2,
        #f"roots are real (up to 1e-12): {roots_are_almost_real}",
        #transform=ax.transAxes,
        #fontsize=9,
        #va="top",
        #ha="left",
        ##bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        #zorder=5,
    #)

    plt.axhline(0, linewidth=0.9)
    plt.title(rf"Residual polynomial $\bar{{p}}_{{{k}}}(\lambda)$ with eigenvalues")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(rf"$\bar{{p}}_{{{k}}}(\lambda)$")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    if outpath is None:
        outpath = f"residual_poly_raw_k{k}.pdf"
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()


def plot_residual_poly_zoom_near_zeros(
    residual_polynomials,
    eigenvalues,
    k=1,
    outpath=None,
    ylims=(-1.0, 1.0),
    pad=0.01,
    num_eigs_show=None,
):
    """
    Zoomed polynomial plot \bar{p}_k(λ) on real axis (default y in [-1,1]),
    with eigenvalues (real parts) marked on x-axis. Saves exactly to outpath (unchanged behavior).
    """
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

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, y, linewidth=1.6, label=rf"$\bar{{p}}_{{{k}}}(\lambda)$")
    plt.ylim(ylims)

    # Eigenvalues as small, semi-transparent red x's
    plt.scatter(
        eig_real_plot,
        np.zeros_like(eig_real_plot),
        marker="x",
        s=25,
        linewidths=1.0,
        alpha=0.6,
        color="red",
        label="Eigenvalues",
        zorder=2,
    )

    # Diagnostic: are roots (of p_k) almost real?
    imag_tol = 1e-12
    roots = np.asarray(p.roots())
    roots_are_almost_real = bool(np.all(np.abs(np.imag(roots)) <= imag_tol))

    #ax = plt.gca()
    #ax.text(
        #0.7, 0.2,
        #f"roots are real (up to 1e-12): {roots_are_almost_real}",
        #transform=ax.transAxes,
        #fontsize=9,
        #va="top",
       #ha="left",
        #bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        #zorder=5,
    #)

    plt.axhline(0, linewidth=0.9)
    plt.title(rf"Residual polynomial $\bar{{p}}_{{{k}}}(\lambda)$ with eigenvalues (zoomed)")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(rf"$\bar{{p}}_{{{k}}}(\lambda)$")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    if outpath is None:
        outpath = f"residual_poly_zoom_k{k}.pdf"
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()


def plot_residual_poly_zoom_near_zeros_both(
    residual_polynomials,
    eigenvalues,
    k=1,
    outpath=None,
    ylims=(-0.5, 0.5),
    xlims=(5000.0, 50000.0),
    pad=0.01,
    num_eigs_show=None,
):
    if k < 1 or k > len(residual_polynomials):
        raise ValueError(f"k={k} out of range. Have {len(residual_polynomials)} polynomials.")

    p = residual_polynomials[k - 1]
    eigenvalues = np.asarray(eigenvalues)

    eig_real = np.real(eigenvalues)
    eig_real_plot = eig_real[: int(num_eigs_show)] if num_eigs_show is not None else eig_real

    # x-grid matches what you actually display
    if xlims is not None:
        x = np.linspace(xlims[0], xlims[1], 1200)
    else:
        lam_min = np.min(eig_real) - pad
        lam_max = np.max(eig_real) + pad
        x = np.linspace(lam_min, lam_max, 1200)

    y = p(x)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, y, linewidth=1.6, label=rf"$\bar{{p}}_{{{k}}}(\lambda)$")
    plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)

    plt.scatter(
        eig_real_plot,
        np.zeros_like(eig_real_plot),
        marker="x",
        s=25,
        linewidths=1.0,
        alpha=0.6,
        color="red",
        label="Eigenvalues",
        zorder=2,
    )

    plt.axhline(0, linewidth=0.9)
    plt.title(rf"Residual polynomial $\bar{{p}}_{{{k}}}(\lambda)$ with eigenvalues (zoomed both axes)")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(rf"$\bar{{p}}_{{{k}}}(\lambda)$")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    if outpath is None:
        outpath = f"residual_poly_zoomboth_k{k}.pdf"
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()


#This is to visualize how the harmonic Ritz values approximate the eigenvalues computed from the full BA matrix
def plot_eigs_and_harmonic_ritz_complex(
    eigenvalues,
    harmonic_ritz_values,
    iterations=(1,),
    outpath=None,
    figsize=(6, 4),
):
    """
    Plot eigenvalues (tiny red 'x') and harmonic Ritz values (blue 'o') in the complex plane.

    Parameters
    ----------
    eigenvalues : (n,) array-like (complex)
    harmonic_ritz_values : list/array
        Either:
          - list where harmonic_ritz_values[k-1] contains the Ritz values at iteration k, or
          - 2D array with shape (K, m_k) (ragged lists are fine).
    iterations : iterable of ints
        Iterations k to plot (1-based).
    outpath : str or None
        If provided, saves to PDF and closes. If None, just returns after plotting (no show).
    """
    eig = np.asarray(eigenvalues).ravel()

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Eigenvalues
    ax.scatter(eig.real, eig.imag, marker="x", s=14, color="red", linewidths=0.8, label="Eigenvalues")

    # Harmonic Ritz values for selected iterations
    first = True
    for k in iterations:
        row = np.asarray(harmonic_ritz_values[k - 1]).ravel()
        ax.scatter(row.real, row.imag, marker="o", s=28, color="blue",
                   label="Harmonic Ritz values" if first else None)
        first = False

    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")
    ax.set_title("Eigenvalues and harmonic Ritz values (complex plane)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print("Saved:", outpath)
    else:
        plt.close()
