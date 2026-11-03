import numpy as np
from scipy import stats
from scipy.signal import butter, sosfiltfilt, hilbert, welch

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

def get_population_oscillations(spike_matrix: np.ndarray, dt: float, n_strongest_freqs: int = 2) -> dict[int, float]:
    spike_signal = spike_matrix.sum(axis=0)
    sr = 1/dt
    freqs, psd = welch(spike_signal, fs=sr, nperseg=int(sr / 2))
    strongest_freqs = np.argsort(psd)[::-1][:n_strongest_freqs]   # indices of strongest bins
    return {float(freqs[i]): float(psd[i]) for i in strongest_freqs}

def apply_lowpass_filter(signal: np.ndarray, dt: float, cutoff_frequency: int, order: int = 2) -> np.ndarray:
    filter = butter(order, cutoff_frequency, btype='lowpass', output='sos', fs=1/dt)
    return sosfiltfilt(filter, signal)

def apply_highpass_filter(signal: np.ndarray, dt: float, cutoff_frequency: int, order: int = 2) -> np.ndarray:
    filter = butter(order, cutoff_frequency, btype='highpass', output='sos', fs=1/dt)
    return sosfiltfilt(filter, signal)

def apply_bandpass_filter(signal: np.ndarray, dt: float, cutoff_freqs: tuple[int, int], order: int = 2) -> np.ndarray:
    filter = butter(order, cutoff_freqs, btype='bandpass', output='sos', fs=1/dt)
    return sosfiltfilt(filter, signal)

def get_phases(signal: np.ndarray) -> np.ndarray:
    return np.angle(hilbert(signal))
