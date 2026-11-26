import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import uniform

# Resolve repository paths cleanly regardless of working directory
REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_ROOT = REPO_ROOT / "DataConditionalABC-main"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from datacondabc.samplers import mcmc

results_dir = REPO_ROOT / "OU-results" / "21-11-26"
obs = np.load(results_dir / "obs_data.npy")

priors = [
    uniform(loc=0.0, scale=0.2),
    uniform(loc=0.1, scale=0.2),
    uniform(loc=0.4, scale=0.2),
]

chain = mcmc(
    theta=[0.1, 0.2, 0.5],
    prior=priors,
    h=0.01,
    xo=obs,
    iterations=100_000,
)

np.save(results_dir / "mcmc_posterior_samples.npy", chain)

rows = []

# print posterior CI
for name, col in zip(["mu", "sigma", "tau"], chain.T):
    q05, q50, q95 = np.percentile(col, [5, 50, 95])
    print(f"{name}: median = {q50}, 90% CI = [{q05}, {q95}]")
    rows.append({"param": name, "q05": q05, "median": q50, "q95": q95})

pd.DataFrame(rows).to_csv(results_dir / "mcmc_summary.csv", index=False)