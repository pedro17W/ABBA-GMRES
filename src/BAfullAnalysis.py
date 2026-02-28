import time
import numpy as np
from ct_experiment import extract_hessenberg_blocks, extract_V_bases
from BAfull_functions import plot_spectrum_fullBA, plot_eigs_and_harmonic_ritz_complex
from numpy.polynomial.polynomial import Polynomial


t_total_start = time.perf_counter()

# Load full BA matrix
results_file = "results/BA_full_lille_N1024_rnl0.95_seed1.npz"
data = np.load(results_file)

BA_full = data["BA_full"]
N = int(data["N"])
n = int(data["n"])
name = str(data["name"])
rnl = float(data["rnl"])

print("Loaded:", results_file)
print("BA_full shape:", BA_full.shape, "N:", n, "n:", N, "name:", name, "rnl:", rnl)

# Load Arnoldi/Hessenberg objects from saved BA-GMRES run
result_file_2 = "results/lille_iters100_noise0.95.npz"  # <-- make sure name matches your actual file
data_2 = np.load(result_file_2, allow_pickle=True)


###below is just to load the total error from the first run
result_file_3 = "results/analysis/it_err_lille_iters100_rnl0.95.npz"  # <-- make sure name matches your actual file
data_3 = np.load(result_file_3, allow_pickle=True)
total_errors = data_3["total_errors"]


H = data_2["H"]
noisy_H = data_2["noisy_H"]
W = data_2["W"]
W_noisy = data_2["W_noisy"]
Bb_exact = data_2["Bb_exact"]
Bb_noisy = data_2["Bb_noisy"]
iters = data_2["iters"]
noisy_iter = data_2["noisy_iterations"]


H_square, H_rect = extract_hessenberg_blocks(H)
noisy_H_square, noisy_H_rect = extract_hessenberg_blocks(noisy_H)
V_krylov_bases = extract_V_bases(W)

# Dimension sanity checks
assert BA_full.shape == (N, N)
assert W.shape[0] == N, f"W has {W.shape[0]} rows but BA_full has N={N}"

#####--------Now we can start the same analysis as for the small test problems---------------######

# --- TIMER 1: eigendecomposition of BA_full ---
t_eig_start = time.perf_counter()
eigenvalues, eigenvectors = np.linalg.eig(BA_full)   # A = W @ diag(eigenvalues) @ inv(W)
t_eig_end = time.perf_counter()
print(f"[TIMER] eig(BA_full): {t_eig_end - t_eig_start:.3f} s")

idx = np.argsort(np.abs(eigenvalues))[::-1]  # descending |lambda|
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]  # keep eigenvectors aligned with eigenvalues

# --- TIMER 2: inverse of eigenvector matrix ---
t_inv_start = time.perf_counter()
W_inv = np.linalg.inv(eigenvectors)  # invert eigenvector matrix to express rhs in eigenbasis
t_inv_end = time.perf_counter()
print(f"[TIMER] inv(eigenvectors): {t_inv_end - t_inv_start:.3f} s")

print("The condition number and the shape of W is:", np.linalg.cond(eigenvectors), eigenvectors.shape)

# --- TIMER 3: rest of analysis ---
t_rest_start = time.perf_counter()

#This simply plots (saves as pdfs) the spectrum in different ways
plot_spectrum_fullBA(BA_full)

from BAfull_functions import eigen_picard_plot_save_fullBA
#We now simply plot our homemade Piccard condition and the other plot and thereby chech if BA is rich in rhs Bb
rhs_vec = W[:, 0]
rhs_vec_noisy = W_noisy[:, 0]

print("The dimensions of rhs_vec is:", rhs_vec.shape)
rel_Bb = np.linalg.norm(Bb_noisy - Bb_exact) / np.linalg.norm(Bb_exact)
print("relative noise in Bb:", rel_Bb)

out_piccard = eigen_picard_plot_save_fullBA(
    BA_full,
    b_exact=Bb_exact,           # or whatever RHS you want in image space
    b_noisy=Bb_noisy, W_inv=W_inv, eigenvalues=eigenvalues,
    outdir="results/analysis_fullBA/figs/figs_piccard_full",
    prefix=f"BA_full_{name}_N{N}_rnl{rnl}",
)

xi_hat, xi_noisy = out_piccard["xi_hat"], out_piccard["xi_noisy"]

from BAfull_functions import energy_and_reconstruction_from_eigcoords

out_energy = energy_and_reconstruction_from_eigcoords(
    eigenvectors=eigenvectors,   # sorted eigenvectors you already computed
    xi=xi_hat,                   # from eigen_picard_plot_save_fullBA
    b=Bb_exact,                  # Bb (normalized)
    outdir="results/analysis_fullBA/figs/figs_piccard_full",
    prefix=f"BA_full_{name}_N{N}_rnl{rnl}_rhsBb_exact",
)

from BAfull_functions import subdiag_cond_save_pdf
out = subdiag_cond_save_pdf(
    H_list=noisy_H_square,  # e.g. [H1, H2, ..., Hk]
    outdir="results/analysis_fullBA/figs/hessenberg",
    prefix=f"{name}"
)




#########################---------------------------------------------------------------##############################
#########################----Now residual polys, filter factors and specteal plot------###############################
#########################---------------------------------------------------------------##############################
print("Lets look at the polynomials")
#first we load the coefficients for the residual polynomials that we constructed in Harmonic_eigs_poly.py
poly_data = np.load(
    f"results/analysis_harmonic/artifacts/residual_polys_{name}_iters{int(iters)}_rnl{float(rnl)}.npz",
    allow_pickle=True
)

polys_noisefree = [Polynomial(c) for c in poly_data["poly_coeffs_noisefree"]]
polys_noisy = [Polynomial(c) for c in poly_data["poly_coeffs_noisy"]]

#To make sure it does not crash if the noisy Ritz and polys are not computed
polys_noisy = None
c_noisy = poly_data.get("poly_coeffs_noisy", None)
if c_noisy is not None and not (getattr(c_noisy, "shape", None) == () and c_noisy.item() is None):
    polys_noisy = [Polynomial(c) for c in c_noisy]

from BAfull_plotting import plot_residual_poly_raw_save_pdf, plot_residual_poly_zoom_save_pdf
k_poly_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for k in k_poly_list:
    plot_residual_poly_raw_save_pdf(
        polys_noisefree,
        eigenvalues=eigenvalues,
        k=k,
        outdir="results/analysis_fullBA/residualpoly",
        prefix="BA_full",
        num_eigs_show=None,
        pad=0.01
    )
    plot_residual_poly_zoom_save_pdf(
        residual_polynomials=polys_noisefree,
        eigenvalues=eigenvalues,
        k=k,
        outdir="results/analysis_fullBA/residualpoly",
        prefix="BA_full",
        num_eigs_show=None,
        ylims=(-0.3, 0.3),
        pad=0.01,
    )

####Lets compute and plot the filter factors
from BAfull_plotting import compute_filter_factors_fullBA, save_filter_factor_plots_fullBA
filter_factors = compute_filter_factors_fullBA(eigenvalues=eigenvalues, residual_polynomials=polys_noisefree, i_max=30)
noisy_filter_factors = compute_filter_factors_fullBA(eigenvalues=eigenvalues, residual_polynomials=polys_noisy, i_max=30) if polys_noisy is not None else None

#plot and save
save_filter_factor_plots_fullBA(
    filter_factors,
    phi_noisy=noisy_filter_factors,
    outdir="results/analysis_fullBA/filter_factors",
    prefix="BA_full",
    use_abs=True,
    overlay_noisy=True,
    labels=("noise-free", "noisy")
)

###And now the spectral/iteration error plot
from BAfull_plotting import plot_spectral_weights_vs_iteration_error_save_pdf
#We load the iteration error from the semi-convergence-script
iteration_error_data = np.load(f"results/analysis/it_err_{name}_iters{iters}_rnl{rnl}.npz")
iteration_error = iteration_error_data["it_err"]

out_spectral = plot_spectral_weights_vs_iteration_error_save_pdf(
    eigenvalues=eigenvalues,
    xi=xi_noisy,
    iteration_errors=total_errors,
    outdir="results/analysis_fullBA/figs",
    prefix="BA_full",
    use_abs=True,
    i_max=None
)

t_rest_end = time.perf_counter()
print(f"[TIMER] rest of analysis: {t_rest_end - t_rest_start:.3f} s")

t_total_end = time.perf_counter()
print(f"[TIMER] TOTAL runtime: {t_total_end - t_total_start:.3f} s")
print("Done all analysis for full BA matrix.")


print("the largest 10 eigenvalues are:", eigenvalues[:10])
print("the reciprocal of the smallest 10 eigenvalues are:", 1/eigenvalues[-10:])



#####################################-------################################################################
################################# Now we can also plot the harmonic ########################################
############################ Ritz values in the complex plane together with the eigenvalues#################
#######FIRST, WE NEED TO LOAD THEM FROM Harmonic_eigs_poly.py

ritz_data = np.load(
    f"results/analysis_harmonic/artifacts/residual_polys_{name}_iters{int(iters)}_rnl{float(rnl)}.npz",
    allow_pickle=True
)

ritz_noisefree = ritz_data["ritz_noisefree"].tolist()

ritz_noisy = None
rn = ritz_data.get("ritz_noisy", None)
if rn is not None and not (getattr(rn, "shape", None) == () and rn.item() is None):
    ritz_noisy = rn.tolist()




# Example: choose iterations to show
iters_to_show = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 20, 25, 30]

for i in iters_to_show:
    print(f"Iteration {i}: Ritz values (noisefree) = {ritz_noisefree[i-1]}")
    if ritz_noisy is not None:
        print(f"Iteration {i}: Ritz values (noisy) = {ritz_noisy[i-1]}")

    out_hr = f"results/analysis_fullBA/figs/ritz_vs_eigs_{name}_rnl{rnl}_k{i}.pdf"

    plot_eigs_and_harmonic_ritz_complex(
        eigenvalues=eigenvalues,
        harmonic_ritz_values=ritz_noisefree,   # or ritz_noisy
        iterations=[i],                        # <-- must be a list/tuple
        outpath=out_hr,
    )

###########################--------------------------------------------------############################
#############-----------------------------------------------------------------------###############
##################################################################################################
print("is the eigenvalue decomposition accurate? ||BA_full - W diag(eig) W_inv|| =", np.linalg.norm(BA_full - eigenvectors @ np.diag(eigenvalues) @ W_inv))