import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backends.backend_pdf import PdfPages
import snntorch.spikeplot as splt
import numpy as np

from . import analysis as a

def plot_performance(loss_rec: np.ndarray, acc_rec: np.ndarray) -> None:
    fig, (ax_loss, ax_plot) = plt.subplots(nrows=1, ncols=2, facecolor='w', figsize=(18, 7))
    fig.suptitle(f"Training losses and accuracies per epoch")

    ax_loss.plot(a.get_median_line(loss_rec), label="Median loss curve")
    ax_loss.fill_between(np.arange(loss_rec.shape[-1]), *a.get_percentile_band(loss_rec), label="Percentile band between 25 and 75")
    ax_loss.set_title(f"Train Set Loss for {loss_rec.shape[0]} trial(s)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_plot.plot(a.get_median_line(acc_rec), label="Median accuracy curve")
    ax_loss.fill_between(np.arange(loss_rec.shape[-1]), *a.get_percentile_band(acc_rec), label="Percentile band between 25 and 75")
    ax_plot.set_title(f"Train Set Accuracy for {acc_rec.shape[0]} trial(s)")
    ax_plot.set_xlabel("Epoch")
    ax_plot.set_ylabel("Accuracy")
    ax_plot.legend()

def plot_gradients(max_grad_rec: np.ndarray, avg_grad_rec: np.ndarray) -> None:
    fig, (ax_max, ax_avg) = plt.subplots(nrows=1, ncols=2, facecolor='w', figsize=(18, 7))
    fig.suptitle("Training udpate gradients per iteration per trial")

    for trial_i in range(max_grad_rec.shape[0]):
        ax_max.plot(max_grad_rec[trial_i], label=f"seed_{trial_i}")
    ax_max.set_title("Max Gradient")
    ax_max.set_xlabel("Iteration (n)")
    ax_max.set_ylabel("Max Gradient (y)")
    ax_max.legend()

    for trial_i in range(avg_grad_rec.shape[0]):
        ax_avg.plot(avg_grad_rec[trial_i], label=f"seed_{trial_i}")
    ax_avg.set_title("Average Gradient")
    ax_avg.set_xlabel("Iteration (n)")
    ax_avg.set_ylabel("Average Gradient (y)")
    ax_avg.legend()

def plot_weight_matrices(weights: np.ndarray, name: str) -> None:
    matrices = [np.asarray(w) for w in weights[0]]
    n_epochs = weights.shape[1]
    vmax = max(np.abs(m).max() for m in matrices)

    fig, axes = plt.subplots(nrows=1, ncols=n_epochs, squeeze=False, figsize=(8 * n_epochs, 8), constrained_layout=True)
    axes = axes[0]
    for epoch, (ax, w_matrix) in enumerate(zip(axes, matrices)):
        im = ax.imshow(w_matrix, cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Epoch {epoch}")

    fig.suptitle(f"Plots of weight-updating of {name} per training epoch")
    fig.colorbar(im, ax=axes, shrink=0.8)  # single colorbar for all axes
    fig.savefig(fname=name, dpi='figure')

class InteractiveSpikePlot:
    def __init__(self, layer_spikes: dict[str, np.ndarray], target_labels: list[str], epoch_i_start: int = 0) -> None:
        self.layer_spikes = layer_spikes
        self.target_labels = target_labels
        self.num_epochs = layer_spikes["spikes_in"].shape[0]
        self.epoch_i = epoch_i_start % self.num_epochs

        self.fig, axes = plt.subplots(
            len(layer_spikes),
            1,
            figsize=(10, 3 * len(layer_spikes)),
            squeeze=False,
            facecolor="w"
        )
        self.axes = axes[:, 0]
        self.fig.subplots_adjust(bottom=0.15, hspace=0.4)

        self.btn_prev = Button(self.fig.add_axes([0.30, 0.02, 0.15, 0.05]), "◀ Previous")
        self.btn_next = Button(self.fig.add_axes([0.55, 0.02, 0.15, 0.05]), "Next ▶")
        self.btn_prev.on_clicked(self.turn_page_left)
        self.btn_next.on_clicked(self.turn_page_right)

    def draw(self) -> None:
        for ax, (layer_name, data) in zip(self.axes, self.layer_spikes.items()):
            ax.clear()
            if layer_name == "targets":
                target = data[self.epoch_i]
                counts = np.zeros(len(self.target_labels))
                counts[target] = 1

                ax.bar(self.target_labels, counts)
                ax.set_title(f"{layer_name}")
                ax.set_ylabel("Count (n)")
                ax.set_xlabel("Target label")
                ax.set_xticks(range(1, 21))
                continue

            if layer_name == "mems_out":
                mems_matrix = data[self.epoch_i, :, :]
                im = ax.imshow(mems_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
                self.fig.colorbar(im, ax=ax, label="Membrane potential")
                ax.set_title(f"{layer_name}")
                ax.set_ylabel("Neuron (n)")
                ax.set_xlabel("Time step (t)")
                continue

            spike_matrix = data[self.epoch_i, :, :]
            splt.raster(torch.from_numpy(spike_matrix), ax, s=0.1, c="black")
            ax.set_title(f"{layer_name} (mean rate = {round(a.get_layer_mean_spike_rate(spike_matrix), 3)}")
            ax.set_ylabel("Neuron (n)")
            ax.set_xlabel("Time step (t)")

        self.fig.suptitle(f"Epoch {self.epoch_i + 1} / {self.num_epochs}")
        self.fig.canvas.draw_idle()

    def turn_page_left(self, event=None) -> None:
        self.epoch_i = (self.epoch_i - 1) % self.num_epochs
        self.draw()

    def turn_page_right(self, event=None) -> None:
        self.epoch_i = (self.epoch_i + 1) % self.num_epochs
        self.draw()

    def save_as_pdf(self, path: str = "spikes_plot.pdf") -> None:
        original_epoch = self.epoch_i
        buttons = (self.btn_prev.ax, self.btn_next.ax)
        for b in buttons:
            b.set_visible(False)

        with PdfPages(path) as pdf:
            for i in range(self.num_epochs):
                self.epoch_i = i
                self.draw()
                pdf.savefig(self.fig)

        for b in buttons:
            b.set_visible(True)

        self.epoch_i = original_epoch
        self.draw()