from simulator import abcsmc_forward, ou_exact
from simulator.sum_statistics import BaseSummary, CanonicalSummary, ModelSummary, NeuralSummary
from simulator.utilities import sample_prior
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from pathlib import Path

"""
This file is for experimenting the ABC-SMC loop with model-based summary statistics.
"""

# simulate an Ornstein-Uhlenbeck process to generate observed data
#obs = ou_exact(h=0.01, T=10.0, mu=0.1, sigma=0.2, tau=0.5, x0=0.0, n_paths=1)[0].flatten()
#np.save(OBS_DATA_PATH, obs)

# load saved obs data
OBS_DATA_PATH = Path(__file__).resolve().parent / 'obs_data.npy'
obs = np.load(OBS_DATA_PATH)

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

# sample 1000 particles from prior to fit summary statistic
prior_bounds = np.array([[0.0, 0.2], [0.1, 0.3], [0.4, 0.6]])
n_samples = 1000

# prior_bounds shape: (3, 2) where row 0 is mu, row 1 is sigma, row 2 is tau
low_bounds = prior_bounds[:, 0]
high_bounds = prior_bounds[:, 1]

prior_particles = np.random.uniform(
    low=low_bounds, 
    high=high_bounds, 
    size=(n_samples, len(prior_bounds))
)

# simulate trajectories for each sampled particle vector
calibration_paths = np.array([simulate_ou(particle) for particle in prior_particles])

# initialize model sum stats
summarise = ModelSummary(obs=obs, dt=0.01)
summarise.fit(calibration_paths)

# print diagonostic data
print('Fitted IAE balancing weight: ', summarise.weight)
print('Observed summary check: ', summarise(obs))


results = abcsmc_forward(
    obs=obs,
    prior_bounds=prior_bounds,
    simulator=simulate_ou,
    summarise=summarise,
    n_part=1000,
    max_iter=100,
    q=0.75,
    max_sims=1e+7
)

# Unpack 7-element results tuple
(
    abc_particles,
    abc_weights,
    round_times,
    acc_rates,
    obs_summaries,
    thresholds,
    cum_sims,
) = results

n_rounds, n_part, _ = abc_particles.shape
obs_summaries_np = np.asarray(obs_summaries)

# export results as csv
metrics_data = {
    "round": np.arange(1, n_rounds + 1),
    "round_time": round_times,
    "acceptance_rate": acc_rates,
    "abc_threshold": thresholds,
    "cum_total_sims": cum_sims,
}
for j in range(obs_summaries_np.shape[1]):
    metrics_data[f"obs_summary_{j}"] = obs_summaries_np[:, j]

round_metrics_df = pd.DataFrame(metrics_data)
round_metrics_df.to_csv(f"model_round_metrics.csv", index=False)

# save particles in long format
particles_df = pd.DataFrame({
    "round": np.repeat(np.arange(1, n_rounds + 1), n_part),
    "particle_idx": np.tile(np.arange(n_part), n_rounds),
    "mu": abc_particles[:, :, 0].ravel(),
    "sigma": abc_particles[:, :, 1].ravel(),
    "tau": abc_particles[:, :, 2].ravel(),
    "weight": abc_weights.ravel(),
})
particles_df.to_csv(f"model_particles_weights.csv", index=False)

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
plt.savefig('abcsmc_posterior_distributions_model.png', dpi=300, bbox_inches='tight')
plt.show()