import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import acf
from abc import abstractmethod
from scipy import integrate, signal, stats, ndimage
from numpy.lib.stride_tricks import sliding_window_view

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
    Normalised canonical summary statistics
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
    IAE of invariant and spectral density with weighting on spectral density to normalise
    Uses binned KDE which is much faster than gaussian KDE
    """
    def __init__(
        self,
        obs: np.ndarray,
        dt: float = 0.01,
        n_grid: int = 200,
        nperseg: int = 256,
    ):
        super().__init__()
        self.num_sum_stats = 2
        self.weight = None
        self.fitted = False

        self.obs = np.asarray(obs, dtype=np.float64).ravel()
        self.dt = dt
        self.fs = 1.0 / dt
        self.n_grid = n_grid
        self.nperseg = min(len(self.obs), nperseg)

        # precompute uniform grid and bin edges for fast Binned KDE
        pad = 4.0 * np.std(self.obs)
        x_min, x_max = np.min(self.obs) - pad, np.max(self.obs) + pad
        self.bin_edges = np.linspace(x_min, x_max, self.n_grid + 1)
        self.dx = self.bin_edges[1] - self.bin_edges[0]
        self.x_grid = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # recompute Hann window, scaling factor, and df for fast Welch PSD
        self.step = self.nperseg // 2
        self.win = np.hanning(self.nperseg)
        self.psd_scale = 1.0 / (self.fs * np.sum(self.win ** 2))
        self.df = self.fs / self.nperseg
        self.freqs = np.fft.rfftfreq(self.nperseg, d=self.dt)

        # store observed densities
        self.obs_invariant_density = self._fast_kde(self.obs)
        self.obs_spectral_density = self._fast_welch(self.obs)

    def _fast_kde(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates 1D Gaussian KDE on self.x_grid via histogram + Gaussian convolution
        """
        n = len(x)
        bw = np.std(x, ddof=1) * (n ** -0.2)
        counts, _ = np.histogram(x, bins=self.bin_edges)
        raw_density = counts / (n * self.dx)
        return ndimage.gaussian_filter1d(raw_density, sigma=bw / self.dx, mode="constant")

    def _fast_welch(self, x: np.ndarray) -> np.ndarray:
        segments = sliding_window_view(x, self.nperseg)[:: self.step]
        segments = (segments - np.mean(segments, axis=1, keepdims=True)) * self.win
        fft_vals = np.fft.rfft(segments, axis=1)
        psd = np.mean(np.abs(fft_vals) ** 2, axis=0) * self.psd_scale

        if self.nperseg % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0
        return psd

    def _extract_raw(self, data: np.ndarray) -> np.ndarray:
        """
        Returns IAE of invariant and spectral densities against the observed densities
        """
        data_1d = np.asarray(data, dtype=np.float64).ravel()

        if np.std(data_1d) < 1e-8 or len(data_1d) < self.nperseg:
            return np.array([2.0, 1e6], dtype=np.float32)

        sim_invariant_density = self._fast_kde(data_1d)
        IAE_invariant_d = np.sum(np.abs(sim_invariant_density - self.obs_invariant_density)) * self.dx

        sim_spectral_density = self._fast_welch(data_1d)
        IAE_spectral_d = np.sum(np.abs(sim_spectral_density - self.obs_spectral_density)) * self.df

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
            return np.array(
                [IAE_invariant_d, IAE_spectral_d * self.weight], dtype=np.float32
            )

        return np.array([IAE_invariant_d, IAE_spectral_d], dtype=np.float32)

class NeuralSummary(BaseSummary):
    """
    Neural network summary statistic
    """
    def __init__(self):
        super().__init__()
        self.num_sum_stats = 10