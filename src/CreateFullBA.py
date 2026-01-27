# BAcreate_fullBA_matrix_minimal.py
import os
import numpy as np

from ct_experiment import make_ba_test_problem, fp_astra, bp_astra, make_BA_operator
from ct_experiment import construct_full_BA

RESULTS_FILE = "results/lille_iters100_noise0.04.npz" #<------- change this if we use other data
OUTDIR = "results"  # only this folder will be used/created if needed

SEED = 1
GPU = True
GROUND_TRUTH = "shepp_logan"

# Load run metadata (to rebuild the same CT problem)
data = np.load(RESULTS_FILE, allow_pickle=True)
name = str(data["name"])
rnl = float(data["rnl"])

# Rebuild operators
CT_setup, b_exact, b_noisy, x_true, info = make_ba_test_problem(
    name, rnl=rnl, seed=SEED, gpu=GPU, ground_truth=GROUND_TRUTH
)
A = fp_astra(CT_setup)
B = bp_astra(CT_setup)
BA_operator = make_BA_operator(A, B, dtype=np.float64)

# Build full BA matrix
N = BA_operator.shape[0]
n = int(np.sqrt(N))
assert n * n == N
BA_full = construct_full_BA(BA_operator, n, dtype=np.float64)

print("Lets perform a check:")
x_check = np.random.randn(BA_operator.shape[0])
print("The relative distance between the linear operator and fullBA applied to x is:")
print(np.linalg.norm(BA_operator @ x_check - BA_full @ x_check) / np.linalg.norm(BA_operator(x_check)))

# Auto-generated output filename (no overwrite)
os.makedirs(OUTDIR, exist_ok=True)
OUTFILE = os.path.join(OUTDIR, f"BA_full_{name}_N{N}_rnl{rnl}_seed{SEED}.npz")

# Save
np.savez_compressed(OUTFILE, BA_full=BA_full, name=name, rnl=rnl, n=n, N=N,
                    seed=SEED, gpu=GPU, ground_truth=GROUND_TRUTH)
print("Saved:", OUTFILE)

print(rnl)