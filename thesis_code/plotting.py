import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import snntorch.spikeplot as splt
import numpy as np

from .recording import SpikeRecorder
from . import analysis as a

def plot_performance(loss_rec: np.ndarray, acc_rec: np.ndarray) -> None:
    fig, (ax_loss, ax_plot) = plt.subplots(nrows=1, ncols=2, facecolor='w', figsize=(18, 7))
    fig.suptitle("Training losses and accuracies per epoch")

    ax_loss.plot(a.get_median_line(loss_rec), label="Median loss curve")
    # lower_bound, upper_bound = a.get_percentile_band(loss_rec)
    ax_loss.fill_between(np.arange(loss_rec.shape[-1]), *a.get_percentile_band(loss_rec), label="Percentile band between 25 and 75")
    ax_loss.set_title(f"Train Set Loss for {loss_rec.shape[0]} trials")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_plot.plot(a.get_median_line(acc_rec), label="Median accuracy curve")
    ax_loss.fill_between(np.arange(loss_rec.shape[-1]), *a.get_percentile_band(acc_rec), label="Percentile band between 25 and 75")
    ax_plot.set_title(f"Train Set Accuracy for {acc_rec.shape[0]} trials")
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

def plot_weight_matrices(weights: np.ndarray) -> None:
    n_epochs = weights.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=n_epochs, squeeze=False, figsize=(8, 8 * n_epochs))
    for ax, w_matrix in zip(axes, [*weights[0]]):
        print(w_matrix.shape)
        ax = ax[0]
        im = ax.imshow(w_matrix, cmap="bwr")

    fig.suptitle("Plots of weight-updating per training epoch")
    fig.tight_layout()
    fig.colorbar(im)

class InteractiveSpikePlot:
    def __init__(self, recorder: SpikeRecorder, iteration_i_start: int, batch_item_i: int) -> None:
        assert recorder.num_iterations > 0, "the recorder is empty"

        self.recorder = recorder
        self.iteration_i = iteration_i_start % self.recorder.num_iterations
        self.batch_item_i = batch_item_i

        self.fig, axes = plt.subplots(
            self.recorder.num_layers,
            1,
            figsize=(10, 3 * self.recorder.num_layers),
            sharex=True,
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
        for layer_l, ax in enumerate(self.axes):
            ax.clear()
            spk_rec, _ = self.recorder.get_recordings(layer_l, self.iteration_i)
            splt.raster(spk_rec[:, self.batch_item_i, :], ax, s=1.5, c="black")
            ax.set_title(f"Layer {layer_l}")
            ax.set_ylabel("Neuron (n)")

        self.axes[-1].set_xlabel("Time step (t)")
        self.fig.suptitle(f"Iteration {self.iteration_i + 1} / {self.recorder.num_iterations} (batch_item {self.batch_item_i})")
        self.fig.canvas.draw_idle()

    def turn_page_left(self, event=None) -> None:
        self.iteration_i = (self.iteration_i - 1) % self.recorder.num_iterations
        self.draw()

    def turn_page_right(self, event=None) -> None:
        self.iteration_i = (self.iteration_i + 1) % self.recorder.num_iterations
        self.draw()

def generate_spikeplot_pdf(recorder: SpikeRecorder) -> None:
    pass
    # TODO finish