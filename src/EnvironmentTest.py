import platform
import sys
from importlib.metadata import distributions

print("=== Python ===")
print("Version:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())

print("\n=== Installed packages ===")
for dist in sorted(distributions(), key=lambda d: d.metadata["Name"].lower()):
    print(f"{dist.metadata['Name']}=={dist.version}")

print("\n=== CUDA / PyTorch ===")
try:
    import torch

    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("torch is not installed")
