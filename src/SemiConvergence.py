# BAanalyse_from_saved.py
# Loads a saved BA-GMRES run (.npz) and produces:
#  1) semiconvergence plot + optimal index (for the saved noisy run)
#  2) reconstructions around the optimal iterate (for the saved noisy run)
#  3) (optional, toggle) multi-noise realization plots (reruns BA-GMRES internally)

import os
import numpy as np

from plotting import (
    semiconvergence_from_iterates,
    plot_reconstructions_around_optimal,
)

# --- Optional: multi-noise realization study (requires rerunning) ---
DO_NOISE_REALIZATION_STUDY = False

if DO_NOISE_REALIZATION_STUDY:
    from ct_experiment import run_ba_gmres, make_ba_test_problem, add_relative_noise
    from Noise_plots import (
        many_noise_total_and_noise_errors,
        plot_iter_all_noise_and_mean_noise,
        plot_iter_all_total_and_mean_total,
        plot_iter_total_and_noise,
    )

# -----------------------------
# Config
# -----------------------------
RESULTS_FILE = "results/lille_iters100_noise0.95.npz"  # <-- change this
OUTDIR = "results/analysis"                             # where figures go
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Helper: ensure iterate matrix is shape (n, K)
# -----------------------------
def as_n_by_k(X: np.ndarray, n: int) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D iterates array, got shape {X.shape}")
    if X.shape[0] == n:
        return X
    if X.shape[1] == n:
        return X.T
    raise ValueError(f"Cannot interpret iterates shape {X.shape} with n={n}")

# -----------------------------
# Load saved run
# -----------------------------
data = np.load(RESULTS_FILE, allow_pickle=True)

iterations = data["iterations"]            # noise-free iterates (from b_exact)
noisy_iterations = data["noisy_iterations"]# noisy iterates (from b_noisy)
x_true = np.asarray(data["x_true"]).reshape(-1)

num_pixels = int(data["num_pixels"])
name = str(data["name"])
iters = int(data["iters"])
rnl = float(data["rnl"])

n = x_true.size
iterations = as_n_by_k(iterations, n)
noisy_iterations = as_n_by_k(noisy_iterations, n)

print(f"Loaded: {RESULTS_FILE}")
print(f"name={name}, iters={iters}, rnl={rnl}, num_pixels={num_pixels}")
print("iterations:", iterations.shape, "noisy_iterations:", noisy_iterations.shape)

# -----------------------------
# 1) Semiconvergence plot (saved single noisy run)
# -----------------------------
semiconv_outpath = os.path.join(OUTDIR, f"{name}_semiconv_iters{iters}_rnl{rnl}.pdf")

out_gmres = semiconvergence_from_iterates(
    noisy_iterations,     # noisy iterates
    iterations,           # noise-free iterates
    x_true,
    semiconv_outpath,
    title="Convergence History",
    compute_noise_error=True,
    noise_error=None,
)

total_errors = out_gmres["relative_errors"]
idx_opt = int(out_gmres["idx_BA"])
print("Optimal iterate idx_BA:", idx_opt)

# Save it_err + total_errors for the saved run (always)
it_err = np.linalg.norm(iterations - x_true[:, None], axis=0)

err_outpath = os.path.join(OUTDIR, f"it_err_{name}_iters{iters}_rnl{rnl}.npz")
np.savez(
    err_outpath,
    it_err=it_err,
    name=name,
    iters=iters,
    rnl=rnl,
    total_errors=total_errors,  # total error curve for the saved noisy run
)
print("Saved it_err + total_errors to:", err_outpath)

# -----------------------------
# 2) Reconstructions around optimal (saved single noisy run)
# -----------------------------
recon_dir = os.path.join(OUTDIR, "reconstructions")
os.makedirs(recon_dir, exist_ok=True)

_ = plot_reconstructions_around_optimal(
    noisy_iterations,
    idx_opt,
    num_pixels,
    x_true,
    before_factor=0.2,
    after_factor=45,  # 128 total (including optimal), so 127 after optimal
    after_offset=0,
    outdir=recon_dir,
    prefix=f"reconstruction_{name}_iters{iters}_rnl{rnl}",
    cmap="viridis",
)

print("Saved semiconvergence + reconstruction figures to:", OUTDIR)

# -----------------------------
# 3) Multi-noise realization plots (optional)
# -----------------------------
if DO_NOISE_REALIZATION_STUDY:
    figs_semiconv_dir = os.path.join(OUTDIR, "figs_semiconv")
    os.makedirs(figs_semiconv_dir, exist_ok=True)

    # Recreate CT_setup & b_exact from the same parameters as the saved run.
    # Update these if your saved run used different settings.
    CT_setup, b_exact, b_noisy, x_true2, info2 = make_ba_test_problem(
        name,
        rnl=rnl,
        seed=1,
        gpu=True,
        ground_truth="shepp_logan",
    )

    # Choose how many noise realizations you want here:
    n_realizations = 10

    total_all, noise_all, X_noisefree = many_noise_total_and_noise_errors(
        CT_setup,
        b_exact,
        x_true2,
        run_ba_gmres,
        add_relative_noise,
        iters=iters,
        rnl=rnl,
        n_realizations=n_realizations,
        seed0=0,
    )

    x_true_vec2 = np.asarray(x_true2).reshape(-1)
    X_noisefree = as_n_by_k(X_noisefree, x_true_vec2.size)

    it_err2 = np.linalg.norm(X_noisefree - x_true_vec2[:, None], axis=0)

    plot_iter_total_and_noise(
        it_err2, total_all, noise_all,
        outpath=os.path.join(figs_semiconv_dir, "iter_total_noise.pdf"),
    )
    plot_iter_all_total_and_mean_total(
        it_err2, total_all,
        outpath=os.path.join(figs_semiconv_dir, "total_all_mean.pdf"),
    )
    plot_iter_all_noise_and_mean_noise(
        it_err2, noise_all,
        outpath=os.path.join(figs_semiconv_dir, "noise_all_mean.pdf"),
    )

    print("Saved multi-noise realization figures to:", figs_semiconv_dir)