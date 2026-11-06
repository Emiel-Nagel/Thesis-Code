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

def get_neuron_spike_count(spikes: np.ndarray) -> int:
    return int(np.sum(spikes))

def get_layer_spike_count(spike_matrix: np.ndarray) -> int:
    return int(np.sum(spike_matrix))

def get_neuron_mean_spike_rate(spikes: np.ndarray) -> float:
    return np.mean(spikes)

def get_neurons_mean_spike_rate(spike_matrix: np.ndarray) -> list[float]:
    return [np.mean(s) for s in spike_matrix]

def get_layer_mean_spike_rate(spike_matrix: np.ndarray) -> float:
    return np.mean(spike_matrix)