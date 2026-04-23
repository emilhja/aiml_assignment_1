# PyTorch GPU Install Notes

## Why `torch.cuda.is_available()` was `False`

The machine has a CUDA-capable GPU and a working NVIDIA driver, but the Python virtual environment had a CPU-only PyTorch build installed.

This was confirmed by:

```python
import torch
print(torch.__version__)   # 2.11.0+cpu
print(torch.version.cuda)  # None
```

`+cpu` means the installed PyTorch build does not include CUDA support.

## Why CPU PyTorch was installed

PyTorch does not use separate package names like `torch` and `torch-gpu`.
There is only one package name:

```bash
torch
```

What changes is the build variant:

- CPU build: `torch 2.11.0+cpu`
- CUDA build: `torch 2.11.0+cu128`

If you run a plain install such as:

```bash
pip install torch torchvision torchaudio
```

`pip` may install the CPU build, depending on which wheels it finds from the configured package index and what matches the current Python environment.

Important: `pip` does not detect your GPU and automatically choose a CUDA build just because the computer has NVIDIA hardware.

## Why there is no `torch-gpu` package

PyTorch keeps the package name as `torch` for both CPU and GPU versions.

That means:

- same import: `import torch`
- same API
- different binary build underneath

So GPU support is selected by installing the correct wheel, not by using a different package name.

## What was true in this project

- GPU detected by the system: yes
- NVIDIA driver installed: yes
- CUDA-capable hardware: yes
- PyTorch installed in `venv`: yes
- Installed PyTorch build had CUDA support: no

That is why this returned `False`:

```python
torch.cuda.is_available()
```

## Install a CUDA-enabled PyTorch build in this project

From the project directory:

```bash
./venv/Scripts/python.exe -m pip uninstall torch torchvision torchaudio
./venv/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Verify after install

```bash
./venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Expected result after a successful CUDA build install:

- version contains something like `+cu128`
- `torch.version.cuda` is not `None`
- `torch.cuda.is_available()` returns `True`
