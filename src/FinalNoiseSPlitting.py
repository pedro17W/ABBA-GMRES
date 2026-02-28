# FinalNoiseSplitting.py
# In this script we perform the noise splitting to illustrate that
# both components of the noise error increase with the iteration number

import os
import numpy as np
import matplotlib.pyplot as plt

from ct_experiment import (
    make_ba_test_problem,
    fp_astra,
    bp_astra,
    make_BA_operator,
    extract_hessenberg_blocks,
    run_ba_gmres,
    expected_propagated_noise_norm
)

from AnalysisRitz import ba_harmonic_ritz_values, ba_residual_polynomials
from BAfull_plotting import compute_filter_factors_fullBA


# -----------------------------
# Config
# -----------------------------
RESULTS_FILE = "results/lille_iters100_noise0.95.npz"   # <-- load name/rnl/iters from here
SEED0 = 0
K_REALIZATIONS = 10

K_MAX = 100
NUM_POLYS = K_MAX
I_MAX = None  # for mini: None means "use all eigenvalues"; you can set e.g. 254 to truncate
K_PLOT = 8    # plotting window for the noise split plots

OUTDIR = "results/analysis_fullBA/noise_split"
os.makedirs(OUTDIR, exist_ok=True)


# -----------------------------
# Helper: consistent noise
# -----------------------------
def add_relative_noise(b_exact, rnl, seed=None):
    if seed is not None:
        np.random.seed(seed)
    e = np.random.randn(b_exact.size).astype(np.float32)
    #e /= np.linalg.norm(e)
    return b_exact + rnl * e


# -----------------------------
# 0) Load metadata from saved BA-GMRES run
# -----------------------------
data = np.load(RESULTS_FILE, allow_pickle=True)
name = str(data["name"])
iters = int(data["iters"])
rnl = float(data["rnl"])
rnl = 0.95

print(f"Loaded metadata from: {RESULTS_FILE}")
print(f"name={name}, iters={iters}, rnl={rnl}")


# -----------------------------
# 1) Rebuild CT setup (mini only) and operators
# -----------------------------
CT_setup, b_exact, _, x_true, info = make_ba_test_problem(
    name, rnl=rnl, seed=1, gpu=True, ground_truth="shepp_logan"
)

A = fp_astra(CT_setup)
B = bp_astra(CT_setup)

# mini size (safe to build BA_full explicitly)
BA_op = make_BA_operator(A, B, dtype=np.float64)
n = info["num_pixels"] ** 2

print(f"Building BA_full explicitly for '{name}' with N={n} ...")
BA_full = np.zeros((n, n), dtype=np.float64)
for j in range(n):
    ej = np.zeros(n, dtype=np.float64)
    ej[j] = 1.0
    BA_full[:, j] = BA_op @ ej

# Eigen-decomposition and sorting
eigenvalues, eigenvectors = np.linalg.eig(BA_full)
idx = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
W = eigenvectors
W_inv = np.linalg.inv(W)


# -----------------------------
# 2) Create noisy measurement b's (k realizations) + Bb versions
# -----------------------------
b_noisy_list = [add_relative_noise(b_exact, rnl, seed=SEED0 + r) for r in range(K_REALIZATIONS)]
Bb_exact = B @ b_exact
Bb_noisy_list = [B @ b for b in b_noisy_list]


# -----------------------------
# 3) Noise-free baseline: run BA-GMRES -> residual polys -> phi_bar
# -----------------------------
X_nf, _, H_nf, _, _, _ = run_ba_gmres(CT_setup, b_exact, iters)
H_nf_sq, H_nf_rect = extract_hessenberg_blocks(H_nf)

ritz_nf = ba_harmonic_ritz_values(H_nf_rect, H_nf_sq, k_max=K_MAX)
polys_nf = ba_residual_polynomials(ritz_nf, num_polys=NUM_POLYS)

i_max_use = (I_MAX or len(eigenvalues))

phi_bar = compute_filter_factors_fullBA(
    eigenvalues=eigenvalues,
    residual_polynomials=polys_nf,
    i_max=i_max_use,
)


# -----------------------------
# 4) Noisy realizations: run BA-GMRES -> polys -> phi_list
# -----------------------------
phi_list = []
X_noisy_list = []  # keep iterates so we don't rerun later

for r in range(K_REALIZATIONS):
    X_r, _, H_r, _, _, _ = run_ba_gmres(CT_setup, b_noisy_list[r], iters)
    X_noisy_list.append(X_r)

    H_r_sq, H_r_rect = extract_hessenberg_blocks(H_r)
    ritz_r = ba_harmonic_ritz_values(H_r_rect, H_r_sq, k_max=K_MAX)
    polys_r = ba_residual_polynomials(ritz_r, num_polys=NUM_POLYS)

    phi_r = compute_filter_factors_fullBA(
        eigenvalues=eigenvalues,
        residual_polynomials=polys_r,
        i_max=i_max_use,
    )
    phi_list.append(phi_r)


# -----------------------------
# 5) Build regularized inverses A_reg_bar and A_reg_list (explicit matrices)
# -----------------------------
def build_Areg_list_from_filter_matrix_fullBA(W, eigenvalues, phi_matrix, k_max):
    lam = np.asarray(eigenvalues).reshape(-1)
    k_max = int(min(k_max, phi_matrix.shape[0]))

    I = phi_matrix.shape[1]
    W_use = W[:, :I]
    lam_use = lam[:I]

    Lam_inv = np.diag(1.0 / (lam_use + 0j))  # complex safe
    A_reg = []
    for k in range(k_max):
        Phi_k = np.diag(phi_matrix[k, :])
        A_reg.append(W_use @ Phi_k @ Lam_inv)
    return A_reg

A_reg_bar = build_Areg_list_from_filter_matrix_fullBA(W, eigenvalues, phi_bar, k_max=K_MAX)
A_reg_list = [
    build_Areg_list_from_filter_matrix_fullBA(W, eigenvalues, phi_list[r], k_max=K_MAX)
    for r in range(K_REALIZATIONS)
]

print("Done building:")
print("  phi_bar:", phi_bar.shape)
print("  phi_list:", len(phi_list), "each", phi_list[0].shape)
print("  A_reg_bar:", len(A_reg_bar), "matrices of shape", A_reg_bar[0].shape)
print("  A_reg_list:", len(A_reg_list), "lists")
print("  Bb_exact shape:", Bb_exact.shape, "Bb_noisy_list len:", len(Bb_noisy_list))


# -----------------------------
# 6) Reconstruction check (noise-free)
# -----------------------------
DO_RECON_CHECK = True
N_RECON_TEST = 12

if DO_RECON_CHECK:
    I = A_reg_bar[0].shape[1]  # truncated spectral dimension used in A_reg
    xi_bar = (W_inv @ Bb_exact)[:I]  # IMPORTANT: truncate!

    n_test = min(N_RECON_TEST, len(A_reg_bar), X_nf.shape[1] - 1)

    print("\nRECONSTRUCTION CHECK (noise-free)")
    for k in range(1, n_test + 1):
        x_gmres_k = X_nf[:, k]
        x_reg_k = A_reg_bar[k - 1] @ xi_bar
        diff = x_gmres_k - x_reg_k 
        relative_diff = np.linalg.norm(diff) / np.linalg.norm(x_gmres_k)
        print(f"  k={k:2d}: ||x_gmres - A_reg*xi_bar|| = {np.linalg.norm(diff):.3e}")
        print(f"       relative diff = {relative_diff:.3e}")


# -----------------------------
# 6b) Quick polynomial diagnostics (optional)
# -----------------------------
print("\nPOLY CHECK (optional)")
for i in range(10):
    pv5 = polys_nf[5](eigenvalues[i])
    pv15 = polys_nf[15](eigenvalues[i])
    print(f"p_5(eig[{i}])  = {pv5:.6e}")
    print(f"p_15(eig[{i}]) = {pv15:.6e}")

for i in range(8, 20):
    pv = polys_nf[i](eigenvalues[0])
    print(f"p_{i}(lambda_max) = {pv:.6e}")


# -----------------------------
# 7) Compute propagated/deviation arrays
# -----------------------------
I = A_reg_bar[0].shape[1]
xi_bar = (W_inv @ Bb_exact)[:I]

n_real = len(Bb_noisy_list)
propagated = np.zeros((n_real, K_MAX), dtype=float)
deviation  = np.zeros((n_real, K_MAX), dtype=float)

for jj in range(n_real):
    e_j = Bb_noisy_list[jj].reshape(-1) - Bb_exact.reshape(-1)
    xi_e = (W_inv @ e_j)[:I]

    for k in range(K_MAX):
        v1 = A_reg_list[jj][k] @ xi_e
        v2 = (A_reg_list[jj][k] - A_reg_bar[k]) @ xi_bar
        propagated[jj, k] = np.linalg.norm(v1)
        deviation[jj,  k] = np.linalg.norm(v2)


# -----------------------------
# 8) Error curves for realization 0 (overlay on plots)
# -----------------------------
X_noisy0 = X_noisy_list[0]
x_true_vec = np.asarray(x_true).reshape(-1)

K_available = min(K_MAX, X_nf.shape[1] - 1, X_noisy0.shape[1] - 1)
iteration_error = np.array([np.linalg.norm(X_nf[:, k]    - x_true_vec) for k in range(1, K_available + 1)])
total_error     = np.array([np.linalg.norm(X_noisy0[:, k] - x_true_vec) for k in range(1, K_available + 1)])

k_plot = min(K_PLOT, K_available)


# -----------------------------
# 9) Make THREE plots (save to pdf)
# -----------------------------
eps = np.finfo(float).tiny
P = np.maximum(propagated[:, :k_plot], eps)
D = np.maximum(deviation[:, :k_plot], eps)
T = np.maximum(P + D, eps)

mean_P = np.mean(P, axis=0)
mean_D = np.mean(D, axis=0)

# ===== Plot A: ALL curves, NO means =====
plt.figure(figsize=(10, 6))
plt.yscale("log")

#plt.plot(range(1, k_plot + 1), iteration_error[:k_plot],
         #label=r"Iteration error $\|\bar{x}_k - x_{\mathrm{true}}\|$")
plt.plot(range(1, k_plot + 1), total_error[:k_plot], linestyle="-.", linewidth=3, color="red", alpha=0.35,
         label=r"Total error (noisy) $\|x_k - \bar{x}\|$")

for jj in range(n_real):
    plt.plot(range(1, k_plot + 1), P[jj, :], linestyle="--", alpha=0.7, linewidth=1, color='green',
             label="Propagated noise (all)" if jj == 0 else None)
for jj in range(n_real):
    plt.plot(range(1, k_plot + 1), D[jj, :], linestyle="--", alpha=0.7, linewidth=1, color='orange',
             label="Deviation noise (all)" if jj == 0 else None)
#for jj in range(n_real):
    #plt.plot(range(1, k_plot + 1), T[jj, :], linestyle="-", alpha=0.35, linewidth=1.8, color='black',
             #label="Total noise (all)" if jj == 0 else None)

plt.xlabel("Iteration $k$")
plt.ylabel("Magnitude (log scale)")
plt.title("Noise split: total error and the two noise components over 10 realizations")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

out_all = os.path.join(OUTDIR, f"{name}_rnl{rnl}_noise_split_ALLCURVES_k{k_plot}.pdf")
plt.savefig(out_all, bbox_inches="tight")
plt.close()
print("Saved:", out_all)

# ===== Plot B: Propagated only + mean =====
plt.figure(figsize=(10, 6))
plt.yscale("log")

for jj in range(n_real):
    plt.plot(range(1, k_plot + 1), P[jj, :], alpha=0.5, linewidth=0.8, color="green")
    
plt.plot(range(1, k_plot + 1), mean_P, linewidth=2, color="darkgreen", label="Mean propagated noise")

plt.xlabel("Iteration $k$")
plt.ylabel("Magnitude (log scale)")
plt.title("Propagated noise across 10 realizations of the noise - with empirical mean")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

out_prop = os.path.join(OUTDIR, f"{name}_rnl{rnl}_noise_split_PROPAGATED_k{k_plot}.pdf")
plt.savefig(out_prop, bbox_inches="tight")
plt.close()
print("Saved:", out_prop)

# ===== Plot C: Deviation only + mean =====
plt.figure(figsize=(10, 6))
plt.yscale("log")

for jj in range(n_real):
    plt.plot(range(1, k_plot + 1), D[jj, :], alpha=0.5, linewidth=0.8, color="orange")
plt.plot(range(1, k_plot + 1), mean_D, linewidth=2, color="darkorange", label="Mean operator deviation")

plt.xlabel("Iteration $k$")
plt.ylabel("Magnitude (log scale)")
plt.title("Operator deviation across realizations of the noise - with empirical mean")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

out_dev = os.path.join(OUTDIR, f"{name}_rnl{rnl}_noise_split_DEVIATION_k{k_plot}.pdf")
plt.savefig(out_dev, bbox_inches="tight")
plt.close()
print("Saved:", out_dev)

print("\nDone. All noise-splitting plots saved under:", OUTDIR)



out_exp = os.path.join(OUTDIR, f"{name}_rnl{rnl}_expected_propnoise_k{K_MAX}.pdf")

exppropsquared, exppropnorm = expected_propagated_noise_norm(
    W=W,
    filter_matrix=phi_bar,
    eigenvalues=eigenvalues,
    eta=rnl , ######We need to calculate the standard deviation!!!!!!! To enter as argument here
    k_expect=10,
    outpath=out_exp,
)

print("Saved:", out_exp)

