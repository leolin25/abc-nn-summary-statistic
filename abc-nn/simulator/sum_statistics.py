import numpy as np
from statsmodels.tsa.stattools import acf
from abc import abstractmethod

class BaseSummary():
    """
    Base class for summary statistics
    """
    def __init__(self):
        self.num_sum_stats = None

    def get_num_sum_stats(self):
        return self.num_sum_stats

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
        print('Raw summary statistics shape: ', raw_stats_batch.shape)

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

    def fit(self, data: np.ndarray):
        # calculate invariant density and spectral density ratios
        return self

class NeuralSummary(BaseSummary):
    """
    Neural network summary statistic
    """
    def __init__(self):
        super().__init__()
        self.num_sum_stats = 10
