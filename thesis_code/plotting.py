import matplotlib.pyplot as plt

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