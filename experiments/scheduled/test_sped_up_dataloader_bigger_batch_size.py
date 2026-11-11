from pathlib import Path

from thesis_code.configs import *

cfg = Config(
    run_name = Path(__file__).stem,

    training = TrainingConfig(
        dt = 0.001,  # timesteps are now in milliseconds
        batch_size = 512,
        max_iters = 500,
        n_epochs = 50,
        lr = 1e-3,
        separate_timesteps = False,

        regularization = RegularizationConfig(
            apply = True,
            lam_lower = 100.0,
            v_lower = 1e-2,
            lam_uppers = [0.5],
            v_uppers = [50],
            L = 2,
        ),

        dataset = DatasetConfig(
            re_download = False,
            n_timepoints = 200,
        )
    ),

    srnn = SRNNConfig(
        hidden_layers=[
            RecurrentLayerConfig(
                n_neurons = 256,
                percent_pv = 25.0,
                percent_som = 25.0,
                rec_layer_type = "alif",
                beta = SamplingConfig(sampling_mode = "normal", is_decay=True, tau_mean = 10, tau_std = 1),
                beta_w = SamplingConfig(sampling_mode = "normal", is_decay=True, tau_mean = 30, tau_std = 5),
                a = SamplingConfig(sampling_mode = "normal", is_decay=False, tau_mean = 3, tau_std = 2),
                b = SamplingConfig(sampling_mode = "normal", is_decay=False, tau_mean = 1, tau_std = 1)
            )
        ],
        add_control_net = False,
        use_daleslaw = False,
        use_connectivity = False,
        refractory_period = 0,
        tau_out = 20, #ms
    ),

    recording = RecordingConfig(
        record_performance = True,
        record_gradients = True,
        record_weights = True,
        record_spikes = True,
    ),
)

if __name__ == "__main__":
    import argparse

    from thesis_code import run_experiment

    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    run_experiment.run(Path(args.output_dir), args.seed, cfg)