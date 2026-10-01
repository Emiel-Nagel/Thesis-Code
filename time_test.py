import torch, time
print(torch.__version__, torch.version.cuda, torch.get_num_threads())
a = torch.randn(4000, 4000)
t0 = time.time()
for _ in range(5): a @ a
print("matmul", time.time() - t0)