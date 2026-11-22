import mlflow
import torch
from pathlib import Path

from thesis_code import datasets, runs
from thesis_code.build_net import build_srnns
from thesis_code.recording import Recorder
from thesis_code.training import Regularizer, train_net
from thesis_code.configs import Config

def run(output_dir: Path, run_id: str, seed: int, cfg: Config) -> None:
    output_dir, run_id, seed = runs.get_args({"output_dir": Path, "run_id": str, "seed": int})

    mlflow.set_tracking_uri("sqlite:///MLflow_server/mlflow.db")
    mlflow.set_experiment("SRNN-training Experiment 1")

    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)

    with mlflow.start_run(run_name=f"seed {seed}", nested=True, parent_run_id=run_id) as seed_run:
        output_dir = runs.create_output_dir(output_dir, seed)
        
        flushing = torch.set_flush_denormal(True)
        print(f"Floating point flushing is set to {flushing}")

        torch.manual_seed(seed)
        torch.set_num_threads(4)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        train_cfg = cfg.training
        dt = train_cfg.dt
        SHD_trainloader, SHD_testloader, input_dim, classes = datasets.get_SHD_dataset(
            batch_size=train_cfg.batch_size,
            device=device,
            n_timepoints=train_cfg.dataset.n_timepoints,
        )

        reg_cfg = train_cfg.regularization
        regularizer = Regularizer(
            lam_lower=reg_cfg.lam_lower,
            v_lower=reg_cfg.v_lower,
            lam_uppers=reg_cfg.lam_uppers,
            v_uppers=reg_cfg.v_uppers,
            L=reg_cfg.L,
        ) if reg_cfg.apply else None

        srnn_cfg = cfg.srnn
        net, net_control = build_srnns(
            dt=dt,
            input_dim=input_dim,
            output_dim=len(classes),
            recurrent_layer_cfgs=srnn_cfg.recurrent_layers,
            tau_out=srnn_cfg.tau_out,

            device=device,
            output_dir=output_dir,
            add_control_net=srnn_cfg.add_control_net,
        )

        recorder = Recorder(
            num_hidden_layers=len(srnn_cfg.recurrent_layers),
            rec_cfg=cfg.recording,
        )

        net, optimizer, recorder = train_net(net, seed_run.info.run_id, device, SHD_trainloader, SHD_testloader, train_cfg.lr, train_cfg.n_epochs, regularizer, recorder)
        runs.save_net(net, optimizer, epoch=train_cfg.n_epochs, save_path=output_dir / "net.checkpoint.pt")
        recorder.save(output_dir / "raw_output_data_net")
        mlflow.log_artifacts(str(seed_dir))

        if net_control is not None:
            recorder.clear()   

            net_control, optimizer_control, recorder = train_net(net_control, seed_run.info.run_id, device, SHD_trainloader, SHD_testloader, train_cfg.lr, train_cfg.n_epochs, regularizer, recorder)
            runs.save_net(net_control, optimizer_control, epoch=train_cfg.n_epochs, save_path=output_dir / "net_control.checkpoint.pt")
            recorder.save(output_dir / "raw_output_data_net_control")