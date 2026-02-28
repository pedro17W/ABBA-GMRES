# BAanalyse_ritz_eigs_polys.py
# Loads saved BA-GMRES run (.npz) and produces:
#  - (optional) eigenvalue plot of BA via eigs on LinearOperator
#  - harmonic Ritz value plots (noise-free; optionally noisy)
#  - residual polynomial plots (raw + zoom)
# All outputs saved under results/<something>/analysis_harmonic/

import os
import numpy as np


import time
total_start_fullscript = time.perf_counter()

# Saved-run loader
RESULTS_FILE = "results/lille_iters100_noise0.95.npz"  # <-- change this
OUTDIR = "results/analysis_harmonic"                   # <-- change if you want

# Optional toggles
DO_EIGS = True                 # eigs overlay requires reconstructing ct -> A,B (but NOT rerunning GMRES)
EIGS_K = 100
EIGS_WHICH = "LM"

DO_NOISY_HRITZ = True         # compute harmonic Ritz values from noisy_H too (if you want later)

K_MAX_RITZ = 50                # harmonic Ritz values computed for k=1..K_MAX_RITZ
NUM_POLYS = 21                 # how many residual polynomials your function should create

PLOT_HRITZ_ITERS = [1,2, 3, 4, 5, 6, 7, 8,9, 10, 20, 25, 30, 36, 43, 50]  # which iterations to plot harmonic Ritz values for (if you plot them iteratively)
PLOT_POLY_ITERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

# -----------------------------
# Imports from your project
# -----------------------------
from ct_experiment import make_ba_test_problem, extract_hessenberg_blocks, fp_astra, bp_astra, extract_V_bases
from ct_experiment import make_BA_operator
from scipy.sparse.linalg import eigs

from plotting import plot_eigs_complex_plane

from AnalysisRitz import (
    ba_harmonic_ritz_values,
    ba_residual_polynomials,
    plot_harmonic_ritz_complex,
    plot_harmonic_ritz_complex_noisys,
    plot_residual_poly_raw,
    plot_residual_poly_zoom_near_zeros,
    plot_residual_poly_zoom_near_zeros_both,
    compute_normal_ritz_values,
    plot_harmonicandnormal_ritz_complex
)

# -----------------------------
# Setup output folders
# -----------------------------
os.makedirs(OUTDIR, exist_ok=True)
FIG_EIGS = os.path.join(OUTDIR, "figs_eigs")
FIG_HRITZ = os.path.join(OUTDIR, "figs_harmonic")
os.makedirs(FIG_EIGS, exist_ok=True)
os.makedirs(FIG_HRITZ, exist_ok=True)

# -----------------------------
# Load saved data
# -----------------------------
data = np.load(RESULTS_FILE, allow_pickle=True)

H = data["H"]
W = data["W"]
W_noisy = data["W_noisy"]
noisy_H = data["noisy_H"]
noisy_rhs = data["b_noisy"]
noisy_iterations = data["noisy_iterations"]
x_true = data["x_true"]

name = str(data["name"])
iters = int(data["iters"])

rnl = float(data["rnl"])

# If you want to reconstruct CT_setup exactly, keep these consistent with your run script:
SEED = 1
GPU = True
GROUND_TRUTH = "shepp_logan"

print(f"Loaded: {RESULTS_FILE}")
print(f"name={name}, iters={iters}, rnl={rnl}")
print("H shape:", H.shape, "noisy_H shape:", noisy_H.shape)

# -----------------------------
# 1) Eigenvalues of BA (optional)
# -----------------------------
if DO_EIGS:
    # Rebuild CT setup (cheap) to recreate operators A and B (no GMRES run here!)
    CT_setup, b_exact, b_noisy, x_true, info = make_ba_test_problem(
        name, rnl=rnl, seed=SEED, gpu=GPU, ground_truth=GROUND_TRUTH
    )
    A = fp_astra(CT_setup)
    B = bp_astra(CT_setup)

    BA_operator = make_BA_operator(A, B, dtype=np.float64)
    ##########################################################

    v = np.random.randn(CT_setup.n).astype(np.float64)

    t0 = time.perf_counter()
    _ = BA_operator @ v
    t1 = time.perf_counter()

    print("One BA matvec time:", t1 - t0, "seconds")
    ###########################################################

    
    import time
    eig_timer_start = time.perf_counter()
    eigenvalues, _ = eigs(BA_operator, k=EIGS_K, which=EIGS_WHICH)
    eig_timer_end = time.perf_counter()
    print(f"Computed {EIGS_K} eigenvalues of BA in {eig_timer_end - eig_timer_start:.2f} seconds.")


    eigs_out = os.path.join(FIG_EIGS, f"eigs_BA_{name}_k{EIGS_K}_{EIGS_WHICH}.pdf")
    plot_eigs_complex_plane(eigenvalues, outpath=eigs_out, title="Eigenvalues of BA")
    print("Saved eigenvalue plot to:", eigs_out)
else:
    eigenvalues = None



# -----------------------------
# 2) Extract Hessenberg blocks from saved H and noisy_H
# -----------------------------
H_square, H_rect = extract_hessenberg_blocks(H)
noisy_H_square, noisy_H_rect = extract_hessenberg_blocks(noisy_H)
V_krylov_bases = extract_V_bases(W)

# -----------------------------
# 3) Harmonic Ritz values
# -----------------------------
ritz_noisefree = ba_harmonic_ritz_values(H_rect, H_square, k_max=K_MAX_RITZ)
normalritz_noisefree = compute_normal_ritz_values(H_square)

if DO_NOISY_HRITZ:
    ritz_noisy = ba_harmonic_ritz_values(noisy_H_rect, noisy_H_square, k_max=K_MAX_RITZ)
    

    
else:
    ritz_noisy = None
    



# ---- plot normal + harmonic Ritz in their own folder ----
FIG_HRITZ_COMPARE = os.path.join(OUTDIR, "figs_harmonic_normal")
os.makedirs(FIG_HRITZ_COMPARE, exist_ok=True)

normal_harm_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in normal_harm_list:
    out = os.path.join(FIG_HRITZ_COMPARE, f"ritz_harmonic_normal_k{i}.pdf")
    plot_harmonicandnormal_ritz_complex(
        ritz_noisefree,
        iteration=i,
        outpath=out,
        eigenvalues=eigenvalues,
        normal_ritz_values=normalritz_noisefree
    )
    print("Saved:", out)

# -----------------------------
# 4) Residual polynomials
# -----------------------------
polys_noisefree = ba_residual_polynomials(ritz_noisefree, num_polys=NUM_POLYS)
polys_noisy = ba_residual_polynomials(ritz_noisy, num_polys=NUM_POLYS) if ritz_noisy is not None else None


# -----------------------------
# 4b) SAVE residual polynomial coefficients (artifact for BAfullAnalysis.py)
# -----------------------------
ARTDIR = os.path.join(OUTDIR, "artifacts")
os.makedirs(ARTDIR, exist_ok=True)

m = len(polys_noisefree)
poly_k = np.arange(1, m + 1)  # matches your construction: i=0..m-1 -> k=1..m

poly_coeffs_noisefree = np.array([p.coef for p in polys_noisefree], dtype=object)
poly_coeffs_noisy = np.array([p.coef for p in polys_noisy], dtype=object) if polys_noisy is not None else None

ritz_noisefree_obj = np.array(ritz_noisefree, dtype=object)
ritz_noisy_obj = np.array(ritz_noisy, dtype=object) if ritz_noisy is not None else None


artifact_path = os.path.join(ARTDIR, f"residual_polys_{name}_iters{iters}_rnl{rnl}.npz")
np.savez(
    artifact_path,
    poly_k=poly_k,
    poly_coeffs_noisefree=poly_coeffs_noisefree,
    poly_coeffs_noisy=poly_coeffs_noisy,
    ritz_noisefree = ritz_noisefree_obj,
    ritz_noisy = ritz_noisy_obj,
    name=name,
    iters=iters,
    rnl=rnl,
    num_polys=m,
    k_max_ritz=K_MAX_RITZ,
)
print("Saved residual poly artifact to:", artifact_path)

print(type(poly_coeffs_noisefree))
print(type(poly_coeffs_noisefree[0]))
print(poly_coeffs_noisefree[0])

# -----------------------------
# 5) Plot harmonic Ritz values in the complex plane
# -----------------------------
for k in PLOT_HRITZ_ITERS:
    out = os.path.join(FIG_HRITZ, f"ritz_noisefree_k{k}.pdf")
    # Your earlier CT script called plot_harmonic_ritz_complex(ritz_noisefree, iteration=k, outpath=...)
    plot_harmonic_ritz_complex(ritz_noisefree, iteration=k, outpath=out, eigenvalues=eigenvalues)

    print("Saved:", out)

# -----------------------------
# 6) Plot residual polynomials (raw + zoom)
# -----------------------------
for k in PLOT_POLY_ITERS:
    out_raw = os.path.join(FIG_HRITZ, f"poly_raw_k{k}.pdf")
    out_zoom = os.path.join(FIG_HRITZ, f"poly_zoom_k{k}.pdf")
    out_zoomboth = os.path.join(FIG_HRITZ, f"poly_zoomboth_k{k}.pdf")

    plot_residual_poly_raw(polys_noisefree, eigenvalues, k=k, outpath=out_raw)
    plot_residual_poly_zoom_near_zeros(polys_noisefree, eigenvalues, k=k, outpath=out_zoom, ylims=(-0.4, 0.4))
    plot_residual_poly_zoom_near_zeros_both(polys_noisefree, eigenvalues, k=k, outpath=out_zoomboth, ylims=(-0.1, 0.1))

    print("Saved:", out_raw)
    print("Saved:", out_zoom)
    print("Saved:", out_zoomboth)

print("Done. All figures saved under:", OUTDIR)




ritz_artifact = os.path.join(ARTDIR, f"harmonic_ritz_{name}_iters{iters}_rnl{rnl}.npz")
np.savez(
    ritz_artifact,
    ritz_noisefree=np.array(ritz_noisefree, dtype=object),
    ritz_noisy=np.array(ritz_noisy, dtype=object) if ritz_noisy is not None else None,
    eigenvalues=eigenvalues if eigenvalues is not None else None,
    name=name, iters=iters, rnl=rnl, k_max_ritz=K_MAX_RITZ,
)
print("Saved harmonic Ritz artifact to:", ritz_artifact)





#####################Now lets plot the ritz piccard condition etc#########################################
########################-------------------------------------------#######################################
from ct_experiment import ritz_decomp_from_HW, ritz_piccard, plot_bagmres_ritz_weights_vs_total_error

#First, we compute the weight c_i which are the alternatives to \xi_i
##In iteration k, we compute the k ritz vector and corresponding Ritz values. We collect the Ritz vectors: Y_k = [y_1,..., y_k]
## c comes from the following procedure: We express the rhs in the Krylov basis by g = V_k^T @ rhs
##We then solve Y_k @ c = g for c to express the rhs in the Krylov basis, g, in the basis consisting of the Ritz vectors

##We now plot the "Ritz-Piccard plot"
#And finally we plot the weight  |\frac{c_i}{theta_i}| in the same plot as the total error
#We start by computing the relevant total error

x_true_vec = np.asarray(x_true).reshape(-1)          # (n,)
X_noisy = np.asarray(noisy_iterations)               # (n, iters+1)

# total error per iteration (including x0 as first entry)
total_errors = np.linalg.norm(X_noisy - x_true_vec[:, None], axis=0)

# optional: drop initial guess x0
total_errors = total_errors[1:]


k = 25
c, theta, Yk = ritz_decomp_from_HW(W_noisy, noisy_H, rhs_vec=W_noisy[:,0], k=k, return_Yk=True)

ritz_piccard(c, theta, k=k, outpath=None, title_prefix="Ritz", sort_by_theta=True)

plot_bagmres_ritz_weights_vs_total_error(
    c, theta, total_errors[:k], k=k, outpath=None, title_prefix="BA-GMRES", sort_by_theta=True
)
########################-------------------------------------------#######################################
##########################################################################################################









total_end_fullscript = time.perf_counter()
print(f"[TIMER] TOTAL runtime of full script: {total_end_fullscript - total_start_fullscript:.3f} s")