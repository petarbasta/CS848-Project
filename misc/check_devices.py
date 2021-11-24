import os
import torch

# Check for available CPUs
num_cpus = os.cpu_count()
print(f"{num_cpus} CPU threads available")

# Check for available GPUs
if torch.cuda.is_available():
    num_gpu_devices = torch.cuda.device_count()
    print(f"{num_gpu_devices} GPU devices available:")

    for i in range(num_gpu_devices):
        name = torch.cuda.get_device_name(i)
        print(f"cuda:{i} -> {name}")
else:
    print("No GPUs available")

