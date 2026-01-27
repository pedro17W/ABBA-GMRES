
from ct_experiment import (run_ba_gmres, make_ba_test_problem)
#from plotting import semiconvergence_from_iterates, plot_reconstructions_around_optimal, plot_eigs_complex_plane
import numpy as np
#from AnalysisRitz import (ba_harmonic_ritz_values, ba_residual_polynomials, plot_harmonic_ritz_complex, 
#plot_residual_poly_raw, plot_residual_poly_zoom_near_zeros)
#from Noise_plots import (many_noise_total_and_noise_errors, plot_iter_all_noise_and_mean_noise,
                          #plot_iter_all_total_and_mean_total, plot_iter_total_and_noise)
#import matplotlib.pyplot as plt
#To be able to compujte eigenvalues without having access to a stored matrix - but only the matrix-vector multiplication
#from scipy.sparse.linalg import eigs
import os

os.makedirs("results", exist_ok=True)

rnl = 0.04
CT_setup, b_exact, b_noisy, x_true, info = make_ba_test_problem(
    "lille", rnl=rnl, seed=1, gpu=True, ground_truth="shepp_logan"
)

print("Now the relative noise is:", np.linalg.norm(b_exact - b_noisy) / np.linalg.norm(b_exact))

iters = 100
iterations, residuals, H, A, B, W = run_ba_gmres(CT_setup, b_exact, iters)

Bb_exact = B @ b_exact
Bb_noisy = B @ b_noisy

noisy_iterations, noisy_residuals, noisy_H, A_noisy, B_noisy, W_noisy = run_ba_gmres(CT_setup, b_noisy, iters)

outpath = f"results/{info['name']}_iters{iters}_noise{rnl}.npz"
np.savez(
    outpath,
    iterations=iterations, residuals=residuals, H=H, W=W,
    noisy_iterations=noisy_iterations, noisy_residuals=noisy_residuals, noisy_H=noisy_H, W_noisy=W_noisy,
    b_exact=b_exact, b_noisy=b_noisy, Bb_exact = Bb_exact, Bb_noisy = Bb_noisy, x_true=np.asarray(x_true).reshape(-1),
    num_pixels=info["num_pixels"], name=info["name"],
    iters=iters, rnl=rnl, 
    m=CT_setup.m, n=CT_setup.n, num_angles=CT_setup.num_angles, num_dets=CT_setup.num_dets
)

print("Saved results to:", outpath)
print("b_noisy has shape", b_noisy.shape)
print("Bb_exact has shape", Bb_exact.shape)
print("Bb_noisy has shape", Bb_noisy.shape)