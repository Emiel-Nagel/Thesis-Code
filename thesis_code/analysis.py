import numpy as np
from scipy import stats

def get_trial_mean(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=0)

def get_median_line(data: np.ndarray) -> np.ndarray:
    return np.median(data, axis=0)

def get_percentile_band(data: np.ndarray, low: float = 25, high: float = 75) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(data, [low, high], axis=0)

def get_ci_band(data: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    mean = get_trial_mean(data)
    sem = stats.sem(data, axis=0) 
    return stats.t.interval(confidence, df=data.shape[0] - 1, loc=mean, scale=sem)