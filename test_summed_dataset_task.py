import torch

from thesis_code import tasks

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")
task = tasks.ClassifySummedSHD(
    batch_size=128,
    n_timepoints=200,
    device=device,
)

task.download_dataset()

for data, targets in task.trainloader:
    print(data.shape)
    print(targets.shape)

print(targets[0])
print(targets[1])
print(targets[2])