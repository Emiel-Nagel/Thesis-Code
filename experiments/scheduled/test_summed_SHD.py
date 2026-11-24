import torch
from pathlib import Path

from thesis_code import tasks, runs
from thesis_code.build_net import build_srnns, RecurrentLayerConfig, SamplingConfig
from thesis_code.run_logging import Logger
from thesis_code.recording import Recorder, RecordingConfig
from thesis_code.training import train_net, Regularizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
run_name = Path(__file__).stem
task = tasks.ClassifySummedSHD(
    batch_size=128,
    n_timepoints=200,
    device=device,
)

if __name__ == "__main__":
    output_dir, seed = runs.get_args({"output_dir": Path, "seed": int})
    output_dir = runs.create_seed_dir(output_dir, seed)
    logger = Logger(run_name=run_name, seed=seed)

    flushing = torch.set_flush_denormal(True)
    logger.log(f"Floating point flushing is set to {flushing}")

    torch.manual_seed(seed)
    torch.set_num_threads(4)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.log(f"Using device: {device}")

    dt = 0.001

    recurrent_layer_cfgs=[
        RecurrentLayerConfig(
            n_neurons=256,
            p_pv=0.25,
            p_som=0.25,
            p_inh=0.5,
            use_daleslaw=True,
            use_connectivity=True,
            refractory_period=0,
            rec_layer_type="alif",
            beta=SamplingConfig(sampling_mode="normal", is_decay=True, tau_mean=10, tau_std=1),
            learn_beta=True,
            beta_w=SamplingConfig(sampling_mode="normal", is_decay=True, tau_mean=30, tau_std=5),
            a=SamplingConfig(sampling_mode="normal", is_decay=False, tau_mean=3, tau_std=2),
            b=SamplingConfig(sampling_mode="normal", is_decay=False, tau_mean=1, tau_std=1)
        )
    ]
    net, _ = build_srnns(
        dt=dt,
        input_dim=task.input_dim,
        output_dim=task.output_dim,
        recurrent_layer_cfgs=recurrent_layer_cfgs,
        tau_out=20, #ms
        add_control_net=False,
        device=device,
        output_dir=output_dir,
    )

    logger.log("\n\n---------------")
    logger.log("Now testing net")

    net, optimizer, recorder = train_net(
        net=net,
        task=task,
        logger=logger,
        device=device,
        lr=1e-3,
        n_epochs=50,
        regularizer=Regularizer(
            lam_lower = 100.0,
            v_lower = 1e-2,
            lam_uppers = [0.5],
            v_uppers = [50],
            L = 2,
        ),
        recorder=Recorder(
            num_hidden_layers=len(recurrent_layer_cfgs),
            rec_cfg=RecordingConfig(
                record_performance = True,
                record_gradients = True,
                record_weights = True,
                record_spikes = True,
            ),
        ),
    )
    runs.save_net(net, optimizer, epoch=50, save_path=output_dir / "net.checkpoint.pt")
    recorder.save(output_dir / "data_net")