@echo off
cd /d "%~dp0"
echo Installing PyTorch with CUDA (first run downloads ~3 GB)...
.venv\Scripts\python -m pip install -q -r requirements.txt
.venv\Scripts\python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul || .venv\Scripts\python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -q onnx
.venv\Scripts\python -c "import torch;print('CUDA available:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(update your NVIDIA driver if this is False)')"
.venv\Scripts\python train_ai.py %*
pause
