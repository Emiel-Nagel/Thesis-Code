from pathlib import Path

from thesis_code.configs import *

cfg = Config(
    run_name = Path(__file__).stem,

    training = TrainingConfig(
        dt = 0.001,  # timesteps are now in milliseconds
        batch_size = 128,
        max_iters = 800,
        n_epochs = 50,
        lr = 1e-3,
        separate_timesteps = False,
        n_trials = 5,

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
        )
    ),

    srnn = SRNNConfig(
        ns_hidden = [256],
        percent_pv = 25.0,
        percent_som = 25.0,
        add_control_net = False,
        use_synapses = False,
        use_daleslaw = False,
        use_connectivity = False,
        refractory_period = 5,

        decays = DecayConfig(
            # beta_sampling_mode = "single_value"
            # tau_single_values = [10] #ms

            beta_sampling_mode = "normal",
            tau_means = [10], #ms
            tau_stds = [1], #ms

            # beta_sampling_mode = "uniform"
            # tau_uppers = [11] #ms
            # tau_lowers = [9] #ms

            tau_out = 20, #ms
        )
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