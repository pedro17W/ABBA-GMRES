# BAanalyse_ritz_eigs_polys.py
# Loads saved BA-GMRES run (.npz) and produces:
#  - (optional) eigenvalue plot of BA via eigs on LinearOperator
#  - harmonic Ritz value plots (noise-free; optionally noisy)
#  - residual polynomial plots (raw + zoom)
# All outputs saved under results/<something>/analysis_harmonic/

import os
import numpy as np

# Saved-run loader
RESULTS_FILE = "results/lille_iters100_noise0.04.npz"  # <-- change this
OUTDIR = "results/analysis_harmonic"                   # <-- change if you want

# Optional toggles
DO_EIGS = True                 # eigs overlay requires reconstructing ct -> A,B (but NOT rerunning GMRES)
EIGS_K = 30
EIGS_WHICH = "LM"

DO_NOISY_HRITZ = True         # compute harmonic Ritz values from noisy_H too (if you want later)

K_MAX_RITZ = 30                # harmonic Ritz values computed for k=1..K_MAX_RITZ
NUM_POLYS = 10                 # how many residual polynomials your function should create

PLOT_HRITZ_ITERS = [1, 5, 10]
PLOT_POLY_ITERS = [1, 5, 10]

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
    plot_residual_poly_raw,
    plot_residual_poly_zoom_near_zeros,
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
    eigenvalues, _ = eigs(BA_operator, k=EIGS_K, which=EIGS_WHICH)

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

if DO_NOISY_HRITZ:
    ritz_noisy = ba_harmonic_ritz_values(noisy_H_rect, noisy_H_square, k_max=K_MAX_RITZ)
    
else:
    ritz_noisy = None
    

# -----------------------------
# 4) Residual polynomials
# -----------------------------
polys_noisefree = ba_residual_polynomials(ritz_noisefree, num_polys=NUM_POLYS)
polys_noisy = ba_residual_polynomials(ritz_noisy, num_polys=NUM_POLYS) if ritz_noisy is not None else None

print("The type of the second element in polys_noise_free is", type(polys_noisefree[1]))
print("and it looks like this:", polys_noisefree[1])

# -----------------------------
# 4b) SAVE residual polynomial coefficients (artifact for BAfullAnalysis.py)
# -----------------------------
ARTDIR = os.path.join(OUTDIR, "artifacts")
os.makedirs(ARTDIR, exist_ok=True)

m = len(polys_noisefree)
poly_k = np.arange(1, m + 1)  # matches your construction: i=0..m-1 -> k=1..m

poly_coeffs_noisefree = np.array([p.coef for p in polys_noisefree], dtype=object)
poly_coeffs_noisy = np.array([p.coef for p in polys_noisy], dtype=object) if polys_noisy is not None else None

artifact_path = os.path.join(ARTDIR, f"residual_polys_{name}_iters{iters}_rnl{rnl}.npz")
np.savez(
    artifact_path,
    poly_k=poly_k,
    poly_coeffs_noisefree=poly_coeffs_noisefree,
    poly_coeffs_noisy=poly_coeffs_noisy,
    ritz_noisefree = ritz_noisefree,
    ritz_noisy = ritz_noisy,
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

    plot_residual_poly_raw(polys_noisefree, ritz_noisefree, k=k, outpath=out_raw)
    plot_residual_poly_zoom_near_zeros(polys_noisefree, ritz_noisefree, k=k, outpath=out_zoom)

    print("Saved:", out_raw)
    print("Saved:", out_zoom)

print("Done. All figures saved under:", OUTDIR)


#################----------------------------------------------------------------------------------#############
#################-------------Now we can perfor the Harmonic-Ritz-basis-analysis-------------------#############
#################----------------------------------------------------------------------------------#############
from ct_experiment import ritz_decomp_from_arnoldi_lists, ritz_piccard, plot_bagmres_ritz_weights_vs_iteration_error

#normalized rhs Bb because our function below does not take care of nomalizing. 
#### WE REMEMBER THAT the residual is Bb and not b_exact in BA-GMRES
rhs_vec = W[:, 0]
rhs_vec_noisy = W_noisy[:, 0]

kritz = K_MAX_RITZ
ritz_xi_exact = ritz_decomp_from_arnoldi_lists(V_krylov_bases, H_rect, H_square, rhs_vec, k=kritz, one_based=True, return_Yk=False)

V_krylov_bases_noisy = extract_V_bases(W_noisy)
ritz_xi_noisy = ritz_decomp_from_arnoldi_lists( V_krylov_bases_noisy, noisy_H_rect, noisy_H_square, rhs_vec_noisy, k=kritz, one_based=True
)

out_piccard = os.path.join(FIG_HRITZ, f"ritz_piccard_k{K_MAX_RITZ}.pdf")
ritz_piccard(ritz_xi_exact, ritz_xi_noisy, ritz_noisefree, k=kritz, outpath=out_piccard)



#########################----------------------------------------#######################
#####Now the plot that shows iteration errors and the ritz-weights in the same plot

iterations = data["iterations"]          # shape (n, iters+1)
x_true = data["x_true"].reshape(-1)      # shape (n,)
iteration_errors = np.linalg.norm(iterations - x_true[:, None], axis=0)
# optional: drop the initial guess column
iteration_errors = iteration_errors[1:]

out_w = os.path.join(FIG_HRITZ, f"ritz_weights_vs_itererr_k{K_MAX_RITZ}.pdf")
plot_bagmres_ritz_weights_vs_iteration_error(
    ritz_xi=ritz_xi_exact,
    ritz_values_list=ritz_noisefree,
    iteration_errors=iteration_errors,  # or whatever you store it as
    k=K_MAX_RITZ,
    outpath=out_w,
    use_ratio=True,
    title_prefix=f"BA-GMRES ({name})"
)
