import os
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum_fullBA(
    A,
    outdir="results/analysis_fullBA/figs",
    prefix="BA_full",
    save_svd=True,
    save_eig_abs=True,
    save_eig_complex=True,
):
    """
    Save singular value decay + eigenvalue plots for matrix A as PDFs.

    Creates (by default) these files under `outdir`:
      - <prefix>_singular_values.pdf
      - <prefix>_eigs_abs_sorted.pdf
      - <prefix>_eigs_complex_plane.pdf
    """
    os.makedirs(outdir, exist_ok=True)

    # ---- Singular values ----
    if save_svd:
        s = np.linalg.svd(A, compute_uv=False)
        plt.figure(figsize=(6, 4))
        plt.plot(s, marker="o")
        plt.xlabel("Index")
        plt.ylabel("Singular values")
        plt.yscale("log")
        plt.title("Singular values")
        plt.grid(True)
        out = os.path.join(outdir, f"{prefix}_singular_values.pdf")
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()

    # ---- Eigenvalues + eigenvectors (kept aligned) ----
    # If you later want eigenvectors too, you already have them here.
    eigenvalues, W = np.linalg.eig(A)

    # Sort by decreasing |lambda| (and keep eigenvectors aligned)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    W = W[:, idx]  # not used for plots, but kept correct/available

    # ---- |eigenvalues| (sorted) ----
    if save_eig_abs:
        plt.figure(figsize=(7, 4))
        plt.plot(np.abs(eigenvalues), marker="o")
        plt.xlabel("Index (sorted by decreasing |λ|)")
        plt.ylabel(r"$|\lambda|$")
        plt.yscale("log")
        plt.title("Eigenvalues (magnitude)")
        plt.grid(True)
        out = os.path.join(outdir, f"{prefix}_eigs_abs_sorted.pdf")
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()

    # ---- Eigenvalues in complex plane ----
    if save_eig_complex:
        plt.figure(figsize=(7, 4))
        plt.scatter(eigenvalues.real, eigenvalues.imag, marker="o")
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.title("Eigenvalues in complex plane")
        plt.grid(True, which="both", linestyle="--", alpha=0.7)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=0.8)
        out = os.path.join(outdir, f"{prefix}_eigs_complex_plane.pdf")
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()

    return {
        "singular_values": s if save_svd else None,
        "eigenvalues": eigenvalues,
        "eigenvectors": W,  # sorted to match eigenvalues
        "outdir": outdir,
    }




#This is to visualize how the harmonic Ritz values approximate the eigenvalues computed from the full BA matrix
def plot_eigs_and_harmonic_ritz_complex(
    eigenvalues,
    harmonic_ritz_values,
    iterations=(1,),
    outpath=None,
    figsize=(8, 4),
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
    ax.scatter(eig.real, eig.imag, marker="x", s=30, color="red", linewidths=0.8, label="Eigenvalues", alpha=0.9)

    # Harmonic Ritz values for selected iterations
    first = True
    for k in iterations:
        row = np.asarray(harmonic_ritz_values[k - 1]).ravel()
        ax.scatter(row.real, row.imag, marker="o", s=25, color="blue", alpha=0.7, linewidths=0.8,
                   label="Harmonic Ritz values" if first else None)
        first = False

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
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


import os
import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt


def eigen_picard_plot_save_fullBA(
    A,
    b_exact,
    b_noisy,
    W_inv,
    eigenvalues,
    outdir="results/analysis_fullBA/figs/figs_piccard_full",
    prefix="BA_full",
):
    """
    Computes eigen-decomposition of A, computes xi_hat and xi_noisy in the eigenbasis,
    and saves the Picard plot as a PDF.

    Saves:
      - <outdir>/<prefix>_picard.pdf

    Returns
    -------
    dict with keys:
        eigenvalues, W, condW, xi_hat, xi_noisy, picard_pdf
    """
    os.makedirs(outdir, exist_ok=True)

    A = np.asarray(A)
    b_exact = np.asarray(b_exact).reshape(-1)
    b_noisy = np.asarray(b_noisy).reshape(-1)
        
    
    # Coefficients in eigenbasis
    xi_hat = W_inv @ b_exact
    xi_noisy = W_inv @ b_noisy

    # ---- Picard plot ----
    plt.figure(figsize=(8, 4))
    plt.semilogy(np.abs(eigenvalues[:20]), "o-", label=r"$|\lambda_i|$")
    plt.semilogy(np.abs(xi_hat[:20]), "s-", label=r"$|\bar{\xi}_i|$", alpha=0.3, color="orange")
    plt.semilogy(np.abs(xi_noisy[:20]), "x--", label=r"$|\xi_i|$ ", alpha=0.7, color="green")
    plt.xlabel("Index $i$")
    plt.ylabel("Magnitude (log scale)")
    plt.title("Eigenvalue Picard Condition")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()

    picard_path = os.path.join(outdir, f"{prefix}_picard.pdf")
    plt.tight_layout()
    plt.savefig(picard_path, bbox_inches="tight")
    plt.close()

    return {
        "eigenvalues": eigenvalues,     
        "xi_hat": xi_hat,
        "xi_noisy": xi_noisy,
        "picard_pdf": picard_path,
    }


#And now the energy function
def energy_and_reconstruction_from_eigcoords(
    eigenvectors,   # (N,N) sorted eigenvector matrix
    xi,             # (N,) coefficients in that eigenbasis
    b,              # (N,) target vector (e.g. rhs_vec = Bb/||Bb||)
    outdir="results/analysis_fullBA/figs/figs_piccard_full",
    prefix="BA_full_rhsBb",
    fractions=(0.90, 0.99, 0.999),
):
    """
    Uses precomputed eigenvectors (sorted) and coefficients xi (in that basis)
    to compute and plot:
        E_k = ||b_k||^2 / ||b||^2
        r_k = ||b - b_k|| / ||b||
    where b_k = eigenvectors[:, :k] @ xi[:k].

    Saves:
      - <prefix>_energy_curve.pdf
      - <prefix>_relative_reconstruction_error.pdf

    Returns:
      dict with E, rel_err, and k thresholds for 90/99/99.9%.
    """
    os.makedirs(outdir, exist_ok=True)

    V = np.asarray(eigenvectors)
    xi = np.asarray(xi).reshape(-1)
    b = np.asarray(b).reshape(-1)

    N = b.shape[0]
    assert V.shape[0] == N and V.shape[1] == N, f"eigenvectors must be (N,N), got {V.shape}"
    assert xi.shape[0] == N, f"xi must have length N={N}, got {xi.shape}"

    b_norm = np.linalg.norm(b)
    b_norm2 = b_norm**2 if b_norm > 0 else 1.0

    E = np.zeros(N)
    rel_err = np.zeros(N)

    # Compute b_k for k=1..N (simple and consistent with your truncation definition)
    for k in range(1, N + 1):
        b_k = V[:, :k] @ xi[:k]
        E[k - 1] = (np.linalg.norm(b_k) ** 2) / b_norm2
        rel_err[k - 1] = np.linalg.norm(b - b_k) / (b_norm + 1e-30)

    # k needed for reconstruction fractions (based on relative error <= 1-f)
    k_for_fraction = {}
    for f in fractions:
        thresh = 1.0 - f
        idx = np.where(rel_err <= thresh)[0]
        k_for_fraction[f] = int(idx[0] + 1) if idx.size else N

    # ---- Plot energy curve ----
    plt.figure(figsize=(8, 4))
    plt.plot(E, "o-", label=r"$E_k=\|b_k\|^2/\|b\|^2$")
    for f in fractions:
        plt.axhline(f, linestyle=":", linewidth=1)
    plt.xlabel("k (number of leading eigencomponents)")
    plt.ylabel("Relative energy")
    plt.yscale('log')
    plt.title("Energy captured by leading eigencomponents")
    plt.grid(True, which="both", linestyle=":")
    plt.ylim(-0.05, 1.05)
    plt.legend()

    lines = []
    for f in fractions:
        if f == 0.999:
            lines.append(f"k@99.9% = {k_for_fraction[f]}")
        else:
            lines.append(f"k@{int(100*f)}% = {k_for_fraction[f]}")
    plt.text(
        0.52, 0.10,
        "\n".join(lines),
        transform=plt.gca().transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    pdf_energy = os.path.join(outdir, f"{prefix}_energy_curve.pdf")
    plt.tight_layout()
    plt.savefig(pdf_energy, bbox_inches="tight")
    plt.close()

    # ---- Plot relative reconstruction error ----
    plt.figure(figsize=(8, 4))
    plt.semilogy(rel_err, "o-", label=r"$\|b-b_k\|/\|b\|$")
    for f in fractions:
        plt.axhline(1.0 - f, linestyle=":", linewidth=1)
    plt.xlabel("k (number of leading eigencomponents)")
    plt.ylabel("Relative reconstruction error (log scale)")
    plt.title("Relative reconstruction error vs k")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()

    lines = []
    for f in fractions:
        if f == 0.999:
            lines.append(f"k for 99.9% = {k_for_fraction[f]}")
        else:
            lines.append(f"k for {int(100*f)}% = {k_for_fraction[f]}")
    plt.text(
        0.52, 0.10,
        "\n".join(lines),
        transform=plt.gca().transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    pdf_relerr = os.path.join(outdir, f"{prefix}_relative_reconstruction_error.pdf")
    plt.tight_layout()
    plt.savefig(pdf_relerr, bbox_inches="tight")
    plt.close()

    return {
        "energy": E,
        "rel_err": rel_err,
        "k_for_fraction": k_for_fraction,
        "pdf_energy": pdf_energy,
        "pdf_relerr": pdf_relerr,
    }

##just the sall function that plots and savesthe subdiag of te final Hessenberg 
# and computes the condition numbers of all the intermediate ones
import os
import numpy as np
import matplotlib.pyplot as plt


def subdiag_cond_save_pdf(
    H_list,
    outdir="results/analysis_fullBA/figs/hessenberg",
    prefix="hessenberg",
):
    """
    Saves:
      - <outdir>/<prefix>_cond_numbers.pdf
      - <outdir>/<prefix>_final_subdiag.pdf
    """
    os.makedirs(outdir, exist_ok=True)

    subdiag = np.abs(np.diag(H_list[-1], -1))
    condition_numbers = np.array([np.linalg.cond(H) for H in H_list])

    k_cond = np.arange(1, len(condition_numbers) + 1)
    k_sub = np.arange(1, len(subdiag) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(k_cond, condition_numbers, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel(r"Arnoldi iteration $k$")
    plt.ylabel(r"$\kappa(H_k)$ (condition number)")
    plt.title("Condition numbers of intermediate Hessenberg matrices")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_cond = os.path.join(outdir, f"{prefix}_cond_numbers.pdf")
    plt.savefig(out_cond, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(k_sub, subdiag, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel(r"Subdiagonal index $k$  (entry $h_{k+1,k}$)")
    plt.ylabel(r"$|h_{k+1,k}|$")
    plt.title("Subdiagonal elements of final Hessenberg matrix")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_sub = os.path.join(outdir, f"{prefix}_final_subdiag.pdf")
    plt.savefig(out_sub, bbox_inches="tight")
    plt.close()

    return {"pdf_cond": out_cond, "pdf_subdiag": out_sub}
