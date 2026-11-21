import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
import sys
import torch
import pickle
import numpy as np
from pathlib import Path
from datacondabc.samplers import abcsmc_forward
from datacondabc.simulators.approxsim import finescale_em
from datacondabc.models import schlogl
from datacondabc.nnets import PENCNN
from datacondabc.simulators.exactsim import ou_exact
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd

def simulate_ou(particle):
    mu, sigma, tau = particle
    trajectory = ou_exact(
        h=0.01,
        T=10.0,
        mu=mu,
        sigma=sigma,
        tau=tau,
        x0=0.0,
        n_paths=1,
    )[0]
    return trajectory.flatten()

# Prior.
prior_bounds = np.array([[0.0, 0.2], [0.1, 0.3], [0.4, 0.6]])

# load saved obs data
project_root = Path(__file__).resolve().parents[5]
obs_1d = np.load(project_root / 'obs_data.npy')
obs = obs_1d.reshape(1, 1, -1)

results = abcsmc_forward(
    obs=obs,
    prior_bounds=prior_bounds,
    simulator=simulate_ou,
    npart=1000,
    niter=100,
    q=75, # use percentile instead of quartile
    max_sims=1e+7,
    amortized=True,
    ckpt_path=str(Path(__file__).resolve().parent / "models" / "best-model-pretrained-v1.ckpt")
)

# unpack results tuple
(
    abc_particles,
    abc_weights,
    round_times,
    training_times,
    acc_rates,
    obs_summaries,
    thresholds,
    cum_sims,
) = results

n_rounds, n_part, _ = abc_particles.shape
obs_summaries_np = (
    obs_summaries.detach().cpu().numpy()
    if hasattr(obs_summaries, "detach")
    else np.asarray(obs_summaries)
)

# Align all outputs to the exact number of completed rounds
n_valid = min(
    n_rounds,
    len(round_times),
    len(acc_rates),
    len(thresholds),
    len(cum_sims),
    len(obs_summaries_np),
)

abc_particles = abc_particles[:n_valid]
abc_weights = abc_weights[:n_valid]
obs_summaries_np = obs_summaries_np[:n_valid]

# Export per-round metrics
metrics_data = {
    "round": np.arange(1, n_valid + 1),
    "round_time": np.asarray(round_times)[:n_valid],
    "acceptance_rate": np.asarray(acc_rates)[:n_valid],
    "abc_threshold": np.asarray(thresholds)[:n_valid],
    "cum_total_sims": np.asarray(cum_sims)[:n_valid],
}
for j in range(obs_summaries_np.shape[1]):
    metrics_data[f"obs_summary_{j}"] = obs_summaries_np[:, j]

pd.DataFrame(metrics_data).to_csv("nn_round_metrics.csv", index=False)

# Export particles and weights
pd.DataFrame({
    "round": np.repeat(np.arange(1, n_valid + 1), n_part),
    "particle_idx": np.tile(np.arange(n_part), n_valid),
    "mu": abc_particles[:, :, 0].ravel(),
    "sigma": abc_particles[:, :, 1].ravel(),
    "tau": abc_particles[:, :, 2].ravel(),
    "weight": abc_weights.ravel(),
}).to_csv("nn_particles_weights.csv", index=False)

# true params and the labels latex
true_params = np.array([0.1, 0.2, 0.5])
param_names = [r'$\mu$', r'$\sigma$', r'$\tau$']

# extract particles and importance weight from last round
final_particles = results[0][-1]
final_weights = results[1][-1]
final_weights = final_weights / np.sum(final_weights)

fig, axes = plt.subplots(1, 3, figsize=(14,5))

for i, ax in enumerate(axes):
    low_bound, high_bound = prior_bounds[i, 0], prior_bounds[i, 1]
    x_grid = np.linspace(low_bound, high_bound, 500)

    ax.hist(
        final_particles[:, i],
        bins=25,
        weights=final_weights,
        density=True,
        color='#4C72B0',
        alpha=0.3,
        edgecolor='white',
    )
    kde = gaussian_kde(final_particles[:, i], weights=final_weights)
    posterior_pdf = kde(x_grid)
    ax.plot(x_grid, posterior_pdf, color='#4C72B0', linewidth=2, label='Posterior KDE')
    ax.fill_between(x_grid, posterior_pdf, color='#4C72B0', alpha=0.2)

    # overlay posterior mean and true parameter value
    post_mean = np.average(final_particles[:, i], weights=final_weights)
    ax.axvline(post_mean, color='#1f4e78', linestyle='-.', linewidth=1.5, label='Posterior Mean')
    ax.axvline(true_params[i], color='#D62728', linestyle='--', linewidth=2, label='True Value')

    # add labels and ticks
    ax.set_title(f'Posterior of {param_names[i]}', fontsize=12, fontweight='bold')
    ax.set_xlabel(param_names[i], fontsize=11)
    if i == 0:
        ax.set_ylabel('Probability Density')
    ax.set_xlim(low_bound - 0.05 * (high_bound - low_bound), high_bound + 0.05 * (high_bound - low_bound))
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(loc='upper left', frameon=True)
fig.suptitle('Marginal Posterior Parameter Distributions', fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('abcsmc_posterior_distributions_NN.png', dpi=300, bbox_inches='tight')
plt.show()