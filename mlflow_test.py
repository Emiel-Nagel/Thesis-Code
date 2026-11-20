import mlflow
import torch
import numpy as np
from pathlib import Path
import time
from tqdm.std import EMA
from datetime import datetime
from zoneinfo import ZoneInfo
def get_timestamp() -> int:
    return int(datetime.now(tz=ZoneInfo("Europe/Amsterdam")).timestamp() * 1000)

from thesis_code import SRNN
from thesis_code import datasets
from thesis_code.build_net import build_srnns
from thesis_code.configs import RecurrentLayerConfig, SamplingConfig
from thesis_code.training import compute_loss, measure_accuracy, Regularizer, test_net
from thesis_code.recording import Recorder, RecordingConfig

def get_gradients(net: SRNN) -> tuple[float, float]:
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    if grads:
        n = sum(g.numel() for g in grads)
        max_grad = torch.stack(torch._foreach_norm(grads, float("inf"))).max()
        avg_grad = torch.stack(torch._foreach_norm(grads, 1)).sum() / n
    else:
        max_grad = avg_grad = torch.zeros(())
    return max_grad.item(), avg_grad.item()

class RateTracker:
    def __init__(self, smoothing=0.3):
        self.ema_dn = EMA(smoothing)
        self.ema_dt = EMA(smoothing)
        self.last_n = 0
        self.last_t = time.perf_counter()

    def update(self, n):
        now = time.perf_counter()
        self.ema_dn(n - self.last_n)
        self.ema_dt(now - self.last_t)
        self.last_n, self.last_t = n, now
        return self.ema_dn() / self.ema_dt()

# The set_experiment API creates a new experiment if it doesn't exist.
mlflow.set_tracking_uri("sqlite:///MLflow_server/mlflow.db")
mlflow.set_experiment("SRNN-training Experiment 1")

mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)

flushing = torch.set_flush_denormal(True)
print(f"Floating point flushing is set to {flushing}")

torch.manual_seed(0)
torch.set_num_threads(4)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

SHD_trainloader, SHD_testloader, input_dim, classes = datasets.get_SHD_dataset(
    batch_size=128,
    device=device,
    n_timepoints=200,
)

output_dir = Path("./mlflow_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

regularizer = Regularizer(
    lam_lower = 100.0,
    v_lower = 1e-2,
    lam_uppers = [0.5],
    v_uppers = [50],
    L = 2,
)

params = {
    "epochs": 5,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "optimizer": "ADAM",
    "model_type": "SRNN",
    "hidden_units": [128],
}

# process = psutil.Process(os.getpid())

run_name = "Run 12"
params["run_name"] = run_name
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params, run_id=run.info.run_id)

    for seed in [0, 1]:
        seed_dir = output_dir / str(seed)

        net = build_srnns(
            dt=0.001,
            input_dim=input_dim,
            output_dim=len(classes),
            recurrent_layer_cfgs=[RecurrentLayerConfig(
                n_neurons = 128,
                p_pv = 0.25,
                p_som = 0.25,
                p_inh = 0.5,
                use_daleslaw = False,
                use_connectivity = False,
                refractory_period = 0,
                rec_layer_type = "alif",
                beta = SamplingConfig(sampling_mode = "normal", is_decay=True, tau_mean = 10, tau_std = 1),
                learn_beta = True,
                beta_w = SamplingConfig(sampling_mode = "normal", is_decay=True, tau_mean = 30, tau_std = 5),
                a = SamplingConfig(sampling_mode = "normal", is_decay=False, tau_mean = 3, tau_std = 2),
                b = SamplingConfig(sampling_mode = "normal", is_decay=False, tau_mean = 1, tau_std = 1)
            )],
            tau_out=20,
            add_control_net=False,
            device=device,
            output_dir=output_dir,
        )[0]

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        recorder = Recorder(
            num_hidden_layers=1,
            rec_cfg=RecordingConfig(
                record_performance=True,
                record_gradients=True,
                record_weights=True,
                record_spikes=True,
            )
        )

        with mlflow.start_run(run_name=f"seed {seed}", nested=True, parent_run_id=run.info.run_id) as seed_run:
            print("Logging params")

            for epoch in range(params["epochs"]):
                mlflow.set_tag("progress", f"Epoch {epoch + 1}/{params["epochs"]}")

                net.train()

                losses, accs = [], []
                tracker = RateTracker()
                for iteration, (data, targets) in enumerate(SHD_trainloader):
                    mlflow.log_metric("it/s", tracker.update(iteration), step=iteration, run_id=seed_run.info.run_id)

                    net.reset()
                    forward_fn = net.forward_sequence   # if not separate_timesteps else net
                    spk_outs, mem_outs, hidden_spks, hidden_mems = forward_fn(data)

                    loss_val = compute_loss(mem_outs, targets)
                    
                    if regularizer is not None:
                        loss_val = loss_val + regularizer(hidden_spks)

                    optimizer.zero_grad()
                    loss_val.backward()

                    if recorder is not None:
                        recorder.record_iteration(net=net)

                    optimizer.step()
                    net.apply_daleslaw()
                    net.constrain_betas()

                    losses.append(loss_val.item())
                    acc = measure_accuracy(mem_outs.detach(), targets)
                    accs.append(acc)

                acc_test, loss_test = test_net(net, SHD_testloader)

                print("Logging metrics")
                max_grad, avg_grad = get_gradients(net)
                mlflow.log_metrics(
                    {"train_loss": loss_val.item(), "train_accuracy": acc, "test_loss": loss_test, "test_accuracy": acc_test, 
                    "cuda_memory_peak GiB": (torch.cuda.max_memory_allocated(device) / 2**30),
                    "max_grad": max_grad, "avg_grad": avg_grad},
                    step=epoch, synchronous=False, timestamp=get_timestamp(), run_id=seed_run.info.run_id
                )

                if recorder is not None:
                    recorder.record_epoch(
                        loss_train=float(np.mean(losses)),
                        acc_train=float(np.mean(accs)),
                        loss_test=loss_test,
                        acc_test=acc_test,
                        weights=net.get_weights(),
                        spk_ins=data,
                        hidden_spks=hidden_spks,
                        mem_outs=mem_outs,
                        targets=targets,
                    )

            recorder.save(seed_dir / "raw_output_data")
    mlflow.log_artifacts(str(seed_dir))

                # mlflow.pytorch.log_model(net, name=f"checkpoint_{epoch}")
        # model_info = mlflow.pytorch.log_model(net, name="final_model")

