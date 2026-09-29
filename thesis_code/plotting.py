import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import snntorch.spikeplot as splt

from .network_components import Recorder

def plot_performance(loss_hist: list, acc_hist: list) -> None:
    fig, (ax_loss, ax_plot) = plt.subplots(nrows=1, ncols=2, facecolor='w', figsize=(18, 7))

    ax_loss.plot(loss_hist)
    ax_loss.set_title("Train Set Loss")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")

    ax_plot.plot(acc_hist)
    ax_plot.set_title("Train Set Accuracy")
    ax_plot.set_xlabel("Iteration")
    ax_plot.set_ylabel("Accuracy")

def plot_gradients(max_grad_rec: list, avg_grad_rec: list) -> None:
    fig, (ax_max, ax_avg) = plt.subplots(nrows=1, ncols=2, facecolor='w', figsize=(18, 7))

    ax_max.plot(max_grad_rec)
    ax_max.set_title("Max Gradient")
    ax_max.set_xlabel("Iteration (n)")
    ax_max.set_ylabel("Max Gradient (y)")

    ax_avg.plot(avg_grad_rec)
    ax_avg.set_title("Average Gradient")
    ax_avg.set_xlabel("Iteration (n)")
    ax_avg.set_ylabel("Average Gradient (y)")

class InteractiveSpikePlot:
    def __init__(self, recorder: Recorder, iteration_i_start: int, batch_item_i: int) -> None:
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

