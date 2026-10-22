import torch
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory, ScaledTranslation
from matplotlib.patches import Patch
from matplotlib.widgets import Button
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML
import snntorch.spikeplot as splt
import numpy as np
from pathlib import Path

from . import analysis as a

def plot_performance(loss_rec: np.ndarray, acc_rec: np.ndarray) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(nrows=1, ncols=2, facecolor='w', figsize=(18, 7))
    fig.suptitle(f"Training losses and accuracies per epoch")

    ax_loss.plot(a.get_median_line(loss_rec), label="Median loss curve")
    ax_loss.fill_between(np.arange(loss_rec.shape[-1]), *a.get_percentile_band(loss_rec), alpha=0.3, label="Percentile band between 25 and 75")
    ax_loss.set_title(f"Train Set Loss for {loss_rec.shape[0]} trial(s)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss.legend()

    ax_acc.plot(a.get_median_line(acc_rec), label="Median accuracy curve")
    ax_acc.fill_between(np.arange(loss_rec.shape[-1]), *a.get_percentile_band(acc_rec), alpha=0.3, label="Percentile band between 25 and 75")
    ax_acc.set_title(f"Train Set Accuracy for {acc_rec.shape[0]} trial(s)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_acc.legend()

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

def add_bracket(ax, start, stop, label, side="left", offset=6, cap=4, label_pad=4, color="k", lw=1.2, fontsize=20):
    """Square bracket spanning pixel indices start..stop-1 (end exclusive).
    offset/cap/label_pad are in points."""
    low, high = start - 0.5, stop - 0.5
    horizontal = side in ("top", "bottom")
    base = (blended_transform_factory(ax.transData, ax.transAxes) if horizontal
            else blended_transform_factory(ax.transAxes, ax.transData))
    edge = {"left": 0, "right": 1, "top": 1, "bottom": 0}[side]
    dx, dy = {"left": (-1, 0), "right": (1, 0), "top": (0, 1), "bottom": (0, -1)}[side]
    ha, va, rot = {"left": ("right", "center", 90), "right": ("left", "center", 270),
                   "top": ("center", "bottom", 0), "bottom": ("center", "top", 0)}[side]

    def T(d):
        return base + ScaledTranslation(dx * d / 72, dy * d / 72, ax.figure.dpi_scale_trans)

    def pt(v):
        return (v, edge) if horizontal else (edge, v)

    arrow = dict(arrowprops=dict(arrowstyle="-", color=color, lw=lw, shrinkA=0, shrinkB=0), annotation_clip=False)
    ax.annotate("", xy=pt(low), xytext=pt(high), xycoords=T(offset), textcoords=T(offset), **arrow)
    for v in (low, high):
        ax.annotate("", xy=pt(v), xytext=pt(v), xycoords=T(offset), textcoords=T(offset - cap), **arrow)
    ax.annotate(label, xy=pt((low + high) / 2), xycoords=T(offset + label_pad),
                ha=ha, va=va, rotation=rot, fontsize=fontsize, annotation_clip=False)

def plot_weight_matrices(weights: np.ndarray, neuron_ranges: dict[str, tuple[int, int]] | None, name: str) -> None:
    matrices = [np.asarray(w) for w in weights]
    n_epochs = len(matrices)
    vmax = max(np.abs(m).max() for m in matrices)

    fig, axes = plt.subplots(nrows=1, ncols=n_epochs, squeeze=False, figsize=(8 * n_epochs, 8), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.1)
    axes = axes[0]
    for epoch, (ax, w_matrix) in enumerate(zip(axes, matrices)):
        im = ax.imshow(w_matrix, cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Epoch {epoch}", pad=40, fontsize=20)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Post-synaptic Neuron (n)")
        ax.set_xlabel("Pre-synaptic Neuron (n)")

        if neuron_ranges is not None:
            for neuron_type, (start, stop) in neuron_ranges.items():
                add_bracket(ax, start, stop, neuron_type, side="left", offset=6, cap=4, label_pad=4, color="k", lw=1.2, fontsize=16)
                add_bracket(ax, start, stop, neuron_type, side="top", offset=6, cap=4, label_pad=4, color="k", lw=1.2, fontsize=16)

    fig.suptitle(f"Plots of weight-updating of {name} per training epoch")
    fig.colorbar(im, ax=axes, shrink=0.8)  # single colorbar for all axes
    fig.savefig(fname=name, dpi='figure')

class InteractiveSpikePlot:
    def __init__(self, layer_spikes: dict[str, np.ndarray], target_labels: list[str], neuron_ranges: dict[str, tuple[int, int]], epoch_i_start: int = 0) -> None:
        self.layer_spikes = layer_spikes
        self.target_labels = target_labels
        self.neuron_ranges = neuron_ranges
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
                mems_matrix = data[self.epoch_i, :, :].T
                im = ax.imshow(mems_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
                handles = [
                    Patch(facecolor=im.cmap(im.norm(mems_matrix.min())), label="Low"),
                    Patch(facecolor=im.cmap(im.norm(mems_matrix.max())), label="High"),
                ]
                ax.legend(handles=handles, loc="upper right", title="Membrane potential")
                ax.set_title(f"{layer_name}")
                ax.set_ylabel("Neuron (n)")
                ax.set_xlabel("Time step (t)")
                continue

            spike_matrix = data[self.epoch_i, :, :]
            splt.raster(torch.from_numpy(spike_matrix), ax, s=0.1, c="black")
            ax.set_title(f"{layer_name} (mean rate = {round(a.get_layer_mean_spike_rate(spike_matrix), 3)}")
            ax.set_ylabel("Neuron (n)")
            ax.set_xlabel("Time step (t)")
            ax.set_xlim(0, spike_matrix.shape[0])
            ax.set_ylim(0, spike_matrix.shape[1])

            if layer_name == "spikes_hidden_0":
                for neuron_type, (start, stop) in self.neuron_ranges.items():
                    add_bracket(ax, start, stop, neuron_type, side="right", offset=6, cap=4, label_pad=4, color="k", lw=1.2, fontsize=10)
                    ax.hlines(y=start - 0.5, color="r", lw=1, ls="--", xmin=0, xmax=spike_matrix.shape[0])

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

def plot_spikes(spike_matrix: np.ndarray, title: str) -> None:
    fig = plt.figure(facecolor="w", figsize=(25, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_matrix, ax, c="black")
    ax.set_xticks(np.arange(int(0), int(spike_matrix.shape[0]) + 1))
    ax.tick_params(axis="x", labelsize=8)

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Neuron Number")

def plot_membrane_potentials(mem_data: np.ndarray, spike_matrix: np.ndarray, spike_threshold: float, title: str) -> None:
    n_steps, n_neurons = mem_data.shape
    fig, axes = plt.subplots(3 * n_neurons, 1, sharex=True, figsize=(25, 4 * n_neurons), squeeze=False, gridspec_kw={"height_ratios": [1, 3, 1] * n_neurons})
    for i in range(n_neurons):
        hide_ax = axes[3 * i, 0]
        hide_ax.axis("off")

        mem_ax = axes[3 * i + 1, 0]
        mem_ax.plot(mem_data[:, i], lw=1, label=f"Membrane traces")
        mem_ax.axhline(spike_threshold, c="red", ls="--", label=f"Spiking threshold = {spike_threshold}")
        mem_ax.set_ylabel("Membrane potential (V)")
        mem_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        mem_ax.legend(loc="upper right")
        mem_ax.set_title(f"Neuron {i + 1}")

        spk_ax = axes[3 * i + 2, 0]
        spike_times = np.nonzero(spike_matrix[:, i])[0]
        spk_ax.eventplot(spike_times, lineoffsets=0.5, linelengths=0.3, colors="black")
        spk_ax.set_ylim(0, 1)
        spk_ax.set_yticks([])
        spk_ax.set_ylabel("Spikes", rotation=0, labelpad=25)
        spk_ax.tick_params(labelbottom=True)

    bottom = axes[-1, 0]
    bottom.set_xlim(-0.5, n_steps - 0.5)
    bottom.set_xticks(np.arange(int(0), int(n_steps) + 1))
    bottom.tick_params(axis="x", labelsize=8)
        
    fig.suptitle(title)
    fig.supxlabel("Time Steps")
    fig.tight_layout()

def animate_spike_counts(spike_matrix: np.ndarray, title: str, save_dir: Path | None = None) -> ArtistAnimation:
    n_steps, n_neurons = spike_matrix.shape
    labels = [f"Neuron {i}" for i in range(1, n_neurons + 1)]

    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    fig.suptitle(title)
    anim = splt.spike_count(torch.from_numpy(spike_matrix), fig, ax, labels, interpolate=5, num_steps=n_steps, time_step=1e-3, animate=True)
    HTML(anim.to_html5_video())
    if save_dir is not None:
        anim.save(save_dir / f"{title}.gif")