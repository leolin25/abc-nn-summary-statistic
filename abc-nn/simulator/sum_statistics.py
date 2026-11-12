import numpy as np
from statsmodels.tsa.stattools import acf
from abc import abstractmethod

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
    def __init__(self):
        super().__init__()
        self.num_sum_stats = 2
        self.weight = None
        self.fitted = False

    def fit(self, data: np.ndarray):
        # calculate invariant density and spectral density ratios
        # input n_part of trajectories
        raw_stats_batch = np.array([self._extract_raw(traj) for traj in data])

        # compute median of IAE invariant density
        median_IAE_invariant = 
        median_IAE_spectral = 

        # calculate weighting
        self.weight = median_IAE_invariant / median_IAE_spectral
        self.fitted = True
        
        return self

    def _extract_raw(self, data: np.ndarray) -> np.ndarray:
        # returns integrated absolute error of invariant density and spectral density
        IAE_invariant_d = 
        IAE_spectral_d = 
        
        return np.array([IAE_invariant_d, IAE_spectral_d])

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # single trajectory input
        raw_stats = self._extract_raw(data)

        if self.fitted:
            return np.array([raw_stats[0], raw_stats[1] * self.weight])

        return raw_stats

class NeuralSummary(BaseSummary):
    """
    Neural network summary statistic
    """
    def __init__(self):
        super().__init__()
        self.num_sum_stats = 10