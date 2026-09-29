import torch

class NeuronRecorder:
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.num_iterations = 0
        self.clear()

    def clear(self) -> None:
        self.recordings = {i: {"spk_recs": [], "mem_recs": []} for i in range(self.num_layers)}

    def increment_iterations(self) -> None:
        self.num_iterations += 1

    def add_layer_recordings(self, layer_i: int, spk_rec: torch.Tensor, mem_rec: torch.Tensor) -> None:
        self.recordings[layer_i]["spk_recs"].append(spk_rec.detach().cpu())
        self.recordings[layer_i]["mem_recs"].append(mem_rec.detach().cpu())

    def get_recordings(self, layer_i: int, iteration_i: int) -> tuple[torch.Tensor, torch.Tensor]:
        spk_rec = self.recordings[layer_i]["spk_recs"][iteration_i]
        mem_rec = self.recordings[layer_i]["mem_recs"][iteration_i]
        return spk_rec, mem_rec

class GradRecorder:
    pass # implement later