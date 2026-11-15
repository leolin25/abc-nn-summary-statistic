import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import acf
from abc import abstractmethod
from scipy import integrate, signal, stats

class BaseSummary():
    """
    Base class for summary statistics
    """
    def __init__(self):
        self.num_sum_stats = None

    @abstractmethod
    def fit(self, data: np.ndarray):
        pass

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass

class CanonicalSummary(BaseSummary):
    """
    Canonical summary statistic: mean and variance
    """
    def __init__(self):
        super().__init__()
        self.num_sum_stats = 5
        self.medians = None
        self.mads = None
        self.fitted = False

    def _extract_raw(self, data: np.ndarray) -> np.ndarray:
        # take a single trajectory
        data = np.asarray(data).ravel()

        # extract the raw summary statistics
        mean_val = np.mean(data)
        var_val = np.var(data)
        std_val = np.sqrt(var_val) if var_val > 1e-8 else 1.0
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        if var_val > 1e-8 and len(data) > 2:
            acf_vals = acf(data, nlags=2, fft=True)
            acf_lag1 = acf_vals[1]
            acf_lag2 = acf_vals[2]
        else:
            acf_lag1 = 0.0
            acf_lag2 = 0.0

        return np.array([mean_val, var_val, skewness, acf_lag1, acf_lag2], dtype=np.float32)

    def fit(self, data:np.ndarray):
        # input n_part of trajectories
        raw_stats_batch = np.array([self._extract_raw(traj) for traj in data])

        # compute medians and MADs
        self.medians = np.median(raw_stats_batch, axis=0)
        raw_mads = np.median(np.abs(raw_stats_batch - self.medians), axis=0)

        # prevent division by zero in case of zero MAD
        self.mads = np.where(raw_mads < 1e-6, 1.0, raw_mads)
        self.fitted = True

        return self

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # single trajectory input
        raw_stats = self._extract_raw(data)

        if self.fitted:
            # normalize using medians and MADs
            normalized_stats = (raw_stats - self.medians) / self.mads
            return normalized_stats

        return raw_stats
    
class ModelSummary(BaseSummary):
    """
    Invariant density and spectral density of the SDE
    """
    def __init__(self, obs: np.ndarray, dt: float, n_grid: int=500, nperseg: int=256):
        super().__init__()
        self.num_sum_stats = 2
        self.weight = None
        self.fitted = False

        self.obs = np.asarray(obs).ravel()
        self.dt = dt
        self.fs = 1.0 / dt
        self.n_grid = n_grid
        self.nperseg = min(len(self.obs), nperseg)

        # Precompute observed invariant density (KDE) and spectral density (Welch PSD)
        pad = 4.0 * np.std(self.obs)
        self.x_grid = np.linspace(
            np.min(self.obs) - pad, np.max(self.obs) + pad, self.n_grid
        )
        self.obs_invariant_density = stats.gaussian_kde(self.obs)(self.x_grid)
        self.freqs, self.obs_spectral_density = signal.welch(
            self.obs, fs=self.fs, nperseg=self.nperseg, scaling="density"
        )

    def _extract_raw(self, data: np.ndarray, obs: np.ndarray = None, dt: float = None) -> np.ndarray:
        data_1d = np.asarray(data).ravel()

        if obs is not None:
            obs_1d = np.asarray(obs).ravel()
            fs = 1.0 / (dt if dt is not None else self.dt)
            nperseg = min(len(obs_1d), len(data_1d), self.nperseg)
            pad = 4.0 * np.std(obs_1d)
            x_grid = np.linspace(
                np.min(obs_1d) - pad, np.max(obs_1d) + pad, self.n_grid
            )
            obs_invariant_density = stats.gaussian_kde(obs_1d)(x_grid)
            freqs, obs_spectral_density = signal.welch(
                obs_1d, fs=fs, nperseg=nperseg, scaling="density"
            )
        else:
            fs = self.fs
            nperseg = self.nperseg
            x_grid = self.x_grid
            obs_invariant_density = self.obs_invariant_density
            freqs = self.freqs
            obs_spectral_density = self.obs_spectral_density

        if np.std(data_1d) < 1e-8 or len(data_1d) < nperseg:
            return np.array([2.0, 1e6], dtype=np.float32)

        try:
            sim_invariant_density = stats.gaussian_kde(data_1d)(x_grid)
            IAE_invariant_d = integrate.trapezoid(
                np.abs(sim_invariant_density - obs_invariant_density), x_grid
            )
        except np.linalg.LinAlgError:
            IAE_invariant_d = 2.0

        _, sim_spectral_density = signal.welch(
            data_1d, fs=fs, nperseg=nperseg, scaling="density"
        )
        IAE_spectral_d = integrate.trapezoid(
            np.abs(sim_spectral_density - obs_spectral_density), freqs
        )

        return np.array([IAE_invariant_d, IAE_spectral_d], dtype=np.float32)

    def fit(self, data: np.ndarray):
        raw_stats_batch = np.array([self._extract_raw(traj) for traj in data])

        median_IAE_invariant = np.median(raw_stats_batch[:, 0])
        median_IAE_spectral = np.median(raw_stats_batch[:, 1])

        self.weight = (
            median_IAE_invariant / median_IAE_spectral
            if median_IAE_spectral > 1e-12
            else 1.0
        )
        self.fitted = True
        return self

    def __call__(self, data: np.ndarray) -> np.ndarray:
        IAE_invariant_d, IAE_spectral_d = self._extract_raw(data)

        if self.fitted:
            return np.array([IAE_invariant_d, IAE_spectral_d * self.weight])

        return np.array([IAE_invariant_d, IAE_spectral_d])

class NeuralSummary(BaseSummary):
    """
    Neural network summary statistic
    """
    def __init__(self):
        super().__init__()
        self.num_sum_stats = 10