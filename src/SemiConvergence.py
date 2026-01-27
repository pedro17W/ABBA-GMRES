# BAanalyse_from_saved.py
# Loads a saved BA-GMRES run (.npz) and produces:
#  1) semiconvergence plot + optimal index
#  2) reconstructions around the optimal iterate
#  3) (optional, toggle) multi-noise realization plots (reruns BA-GMRES internally)

import os
import numpy as np

from plotting import (
    semiconvergence_from_iterates,
    plot_reconstructions_around_optimal,
)

# --- Optional: multi-noise realization study (requires rerunning) ---
DO_NOISE_REALIZATION_STUDY = True

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
RESULTS_FILE = "results/lille_iters100_noise0.04.npz"  # <-- change this
OUTDIR = "results/analysis"                           # where figures go

os.makedirs(OUTDIR, exist_ok=True)


# -----------------------------
# Load saved run
# -----------------------------
data = np.load(RESULTS_FILE, allow_pickle=True)

iterations = data["iterations"]
noisy_iterations = data["noisy_iterations"]
x_true = data["x_true"].reshape(-1)

num_pixels = int(data["num_pixels"])
name = str(data["name"])
iters = int(data["iters"])
rnl = float(data["rnl"])

print(f"Loaded: {RESULTS_FILE}")
print(f"name={name}, iters={iters}, rnl={rnl}, num_pixels={num_pixels}")
print("iterations:", iterations.shape, "noisy_iterations:", noisy_iterations.shape)

# -----------------------------
# 1) Semiconvergence plot
# -----------------------------
semiconv_outpath = os.path.join(OUTDIR, f"{name}_semiconv_iters{iters}_rnl{rnl}.pdf")

out_gmres = semiconvergence_from_iterates(
    noisy_iterations,
    iterations,
    x_true,
    semiconv_outpath,
    title="Convergence History",
    compute_noise_error=True,
    noise_error=None,
)

print("Optimal iterate idx_BA:", out_gmres["idx_BA"])

# -----------------------------
# 2) Reconstructions around optimal
# -----------------------------
recon_dir = os.path.join(OUTDIR, "reconstructions")
os.makedirs(recon_dir, exist_ok=True)

_ = plot_reconstructions_around_optimal(
    noisy_iterations,
    out_gmres["idx_BA"],
    num_pixels,
    x_true,
    before_factor=0.4,
    after_factor=3.5,
    after_offset=0,
    outdir=recon_dir,
    prefix=f"reconstruction_{name}_iters{iters}_rnl{rnl}",
)

print("Saved semiconvergence + reconstruction figures to:", OUTDIR)


# -----------------------------
# 3) Multi-noise realization plots (optional)
# -----------------------------
if DO_NOISE_REALIZATION_STUDY:
    figs_semiconv_dir = os.path.join(OUTDIR, "figs_semiconv")
    os.makedirs(figs_semiconv_dir, exist_ok=True)

    # Recreate CT_setup & b_exact from the same parameters as the saved run.
    # NOTE: If your saved run used different values (e.g. gpu flag or ground_truth),
    # update them here to match.
    CT_setup, b_exact, b_noisy, x_true2, info = make_ba_test_problem(
        name,
        rnl=rnl,
        seed=1,
        gpu=True,
        ground_truth="shepp_logan",
    )

    total_all, noise_all, X_noisefree = many_noise_total_and_noise_errors(
        CT_setup,
        b_exact,
        x_true2,
        run_ba_gmres,
        add_relative_noise,
        iters=iters,
        rnl=rnl,
        n_realizations=3,
        seed0=0,
    )

    x_true_vec = np.asarray(x_true2).reshape(-1)
    it_err = np.array([np.linalg.norm(x_true_vec - X_noisefree[:, k]) for k in range(X_noisefree.shape[1])])

    

    plot_iter_total_and_noise(
        it_err, total_all, noise_all,
        outpath=os.path.join(figs_semiconv_dir, "iter_total_noise.pdf"),
    )
    plot_iter_all_total_and_mean_total(
        it_err, total_all,
        outpath=os.path.join(figs_semiconv_dir, "total_all_mean.pdf"),
    )
    plot_iter_all_noise_and_mean_noise(
        it_err, noise_all,
        outpath=os.path.join(figs_semiconv_dir, "noise_all_mean.pdf"),
    )

    print("Saved multi-noise realization figures to:", figs_semiconv_dir)

it_err = np.array([np.linalg.norm(x_true_vec - X_noisefree[:, k]) for k in range(X_noisefree.shape[1])])

err_outpath = os.path.join(OUTDIR, f"it_err_{name}_iters{iters}_rnl{rnl}.npz")
np.savez(
    err_outpath,
    it_err=it_err,
    name=name,
    iters=iters,
    rnl=rnl,
)
print("Saved it_err to:", err_outpath)