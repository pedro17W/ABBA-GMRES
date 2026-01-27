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
        color='blue', marker='o', s=40
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

    # Legend can stay inside without affecting canvas size much
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

    # Pairwise top-5 diffs by |.| (optional)
    if eig is not None:
        m = 5
        idx_e = np.argsort(np.abs(eig))[::-1]
        idx_r = np.argsort(np.abs(row))[::-1]
        top_e = eig[idx_e[:m]]
        top_r = row[idx_r[:m]]
        diffs = np.abs(top_e - top_r)

        diag_lines += ["", "Top-5 by |.| pairing (sorted by magnitude)"]
        for i in range(min(m, top_e.size, top_r.size)):
            diag_lines.append(
                f"{i+1}: |λ-θ|={diffs[i]:.2e}   |λ|={np.abs(top_e[i]):.2e}   |θ|={np.abs(top_r[i]):.2e}"
            )

    diag_text = "\n".join(diag_lines)

    # Reserve space at bottom for the text block
    fig.subplots_adjust(bottom=0.32)  # increase if your text block is longer

    # Place text in the figure margin (not in axes) -> doesn't change axes size
    fig.text(
        0.02, 0.02, diag_text,
        ha="left", va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95)
    )

    # ---- SAVE instead of show ----
    if outpath is None:
        outpath = f"harmonic_ritz_iter_{k}.pdf"

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    # IMPORTANT: don't use bbox_inches="tight" or it will resize to include margins
    fig.savefig(outpath, format="pdf")
    plt.close(fig)



########We also need to plot the residual polynomials
def plot_residual_poly_raw(residual_polynomials, harmonic_ritz_values, k=1, outpath=None):
    """
    Raw polynomial p_k(λ) on real axis + marks harmonic Ritz roots on x-axis.
    (Eigenvalues removed.)
    """
    p = residual_polynomials[k-1]
    roots = np.asarray(harmonic_ritz_values[k-1])

    # Range based only on roots (real parts)
    lam_min = np.min(np.real(roots)) - 0.01
    lam_max = np.max(np.real(roots)) + 0.01
    x = np.linspace(lam_min, lam_max, 1000)
    y = p(x)

    plt.figure(figsize=(5, 4))
    plt.plot(x, y, label=rf"$p_{{{k}}}(\lambda)$")
    plt.axhline(0, color='k', lw=0.8)

    # mark harmonic Ritz roots on x-axis
    plt.scatter(np.real(roots), [0]*len(roots), color='orange', marker='o', s=30, label='Harmonic Ritz values')

    plt.title(rf"Residual Polynomial $p_{{{k}}}(\lambda)$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(rf"$p_{{{k}}}(\lambda)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if outpath is None:
        outpath = f"residual_poly_raw_k{k}.pdf"
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()


def plot_residual_poly_zoom_near_zeros(residual_polynomials, harmonic_ritz_values, k=1, outpath=None):
    """
    Zoom y-axis to [-1,1] and show roots on real axis + diagnostics box.
    (Eigenvalues removed.)
    """
    p = residual_polynomials[k-1]
    roots = np.asarray(harmonic_ritz_values[k-1])

    lam_min = np.min(np.real(roots)) - 0.01
    lam_max = np.max(np.real(roots)) + 0.01
    x = np.linspace(lam_min, lam_max, 1000)
    y = p(x)

    plt.figure(figsize=(8, 6))
    plt.ylim(-1, 1)
    plt.plot(x, y, label=rf"$p_{{{k}}}(\lambda)$")
    plt.axhline(0, color='k', lw=0.8)

    # roots on real axis
    plt.scatter(np.real(roots), [0]*len(roots), color='blue', marker='x', s=20, label='Harmonic Ritz values')

    # ---- diagnostics box (kept; eigenvalue-dependent part removed) ----
    imag_tol = 1e-12
    roots_real = np.all(np.abs(np.imag(roots)) <= imag_tol)

    textstr = f"Roots real: {roots_real}"

    ax = plt.gca()
    ax.text(
        0.52, 0.25,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        va='top',
        ha='left',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )
    # ---------------------------------------------------------------

    plt.title(rf"Residual Polynomial $p_{{{k}}}(\lambda)$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(rf"$p_{{{k}}}(\lambda)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if outpath is None:
        outpath = f"residual_poly_zoom_k{k}.pdf"
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()
