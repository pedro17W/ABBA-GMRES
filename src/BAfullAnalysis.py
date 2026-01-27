import numpy as np
from ct_experiment import extract_hessenberg_blocks, extract_V_bases
from BAfull_functions import plot_spectrum_fullBA
from numpy.polynomial.polynomial import Polynomial

# Load full BA matrix
results_file = "results/BA_full_lille_N1024_rnl0.04_seed1.npz"
data = np.load(results_file)

BA_full = data["BA_full"]
N = int(data["N"])
n = int(data["n"])
name = str(data["name"])
rnl = float(data["rnl"])

print("Loaded:", results_file)
print("BA_full shape:", BA_full.shape, "n:", n, "N:", N, "name:", name, "rnl:", rnl)

# Load Arnoldi/Hessenberg objects from saved BA-GMRES run
result_file_2 = "results/lille_iters100_noise0.04.npz"  # <-- make sure name matches your actual file
data_2 = np.load(result_file_2, allow_pickle=True)

H = data_2["H"]
noisy_H = data_2["noisy_H"]
W = data_2["W"]
W_noisy = data_2["W_noisy"]
Bb_exact = data_2["Bb_exact"]
Bb_noisy = data_2["Bb_noisy"]
iters = data_2["iters"]

H_square, H_rect = extract_hessenberg_blocks(H)
noisy_H_square, noisy_H_rect = extract_hessenberg_blocks(noisy_H)
V_krylov_bases = extract_V_bases(W)


# Dimension sanity checks
assert BA_full.shape == (N, N)
assert W.shape[0] == N, f"W has {W.shape[0]} rows but BA_full has N={N}"

#####--------Now we can start the same analysis as for the small test problems---------------######
eigenvalues, eigenvectors = np.linalg.eig(BA_full)   # A = W @ diag(eigenvalues) @ inv(W)

idx = np.argsort(np.abs(eigenvalues))[::-1]  # descending |lambda|
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]  # keep eigenvectors aligned with eigenvalues
W_inv = np.linalg.inv(eigenvectors) #invert the eigenvector matrix to be able to express rhs in eigenbasis

print("The condition number and the shape of W is:", np.linalg.cond(eigenvectors), eigenvectors.shape)

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
    b=Bb_exact,                   # Bb (normalized)
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

#To make sure it does not crash if the noisy Ritz and polys are not computed
polys_noisy = None
c_noisy = poly_data.get("poly_coeffs_noisy", None)
if c_noisy is not None and not (getattr(c_noisy, "shape", None) == () and c_noisy.item() is None):
    polys_noisy = [Polynomial(c) for c in c_noisy]

from BAfull_plotting import plot_residual_poly_raw_save_pdf, plot_residual_poly_zoom_save_pdf
k_poly_list = [1, 2, 3, 4, 5, 6, 10]
for k in k_poly_list:
    plot_residual_poly_raw_save_pdf(polys_noisefree, eigenvalues=eigenvalues, k=k, outdir="results/analysis_fullBA/residualpoly",
        prefix="BA_full",    num_eigs_show=None,    pad=0.01)
    plot_residual_poly_zoom_save_pdf(residual_polynomials = polys_noisefree,    eigenvalues=eigenvalues,
    k=k,    outdir="results/analysis_fullBA/residualpoly",    prefix="BA_full",    num_eigs_show=None,    ylims=(-0.3, 0.3),
    pad=0.01,)


####Lets compute and plot the filter factors
from BAfull_plotting import compute_filter_factors_fullBA, save_filter_factor_plots_fullBA
filter_factors = compute_filter_factors_fullBA(eigenvalues=eigenvalues, residual_polynomials=polys_noisefree, i_max=10)

#plot and save
save_filter_factor_plots_fullBA(    filter_factors,    phi_noisy=None,    outdir="results/analysis_fullBA/filter_factors",
    prefix="BA_full",    use_abs=True,    overlay_noisy=False,    labels=("noise-free", "noisy"))


###And now the spectral/iteration error plot
from BAfull_plotting import plot_spectral_weights_vs_iteration_error_save_pdf
#We load the iteration error from the semi-convergence-script
iteration_error_data = np.load(f"results/analysis/it_err_{name}_iters{iters}_rnl{rnl}.npz")
iteration_error = iteration_error_data["it_err"]

out_spectral = plot_spectral_weights_vs_iteration_error_save_pdf(eigenvalues=eigenvalues, xi=xi_hat, iteration_errors=iteration_error,
                                                                 outdir="results/analysis_fullBA/figs",
    prefix="BA_full",    use_abs=True,    i_max=None,)