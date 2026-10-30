# SPDX-License-Identifier: GPL-3.0-or-later
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
$env:PYTHONUTF8 = '1'
Push-Location $projectRoot
try {
    # Restore the handed-off, locked Python 3.12 environment first.
    & uv sync --frozen --inexact --no-install-package torch
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed; keep the output for diagnosis.' }
    $projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
    & $projectPython -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda); raise SystemExit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        # This CUDA/PyTorch version is already verified on this PC in Python 3.14.
        # Install it only in the project venv; preserve the cross-platform lockfile.
        & uv pip install --python $projectPython --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0'
        if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    }
    & $projectPython -c "import torch, tensorboard, pesq, pystoi, pyworld; assert torch.cuda.is_available(); x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); print(torch.__version__, torch.cuda.get_device_name(0), 'CUDA computation OK')"
    if ($LASTEXITCODE -ne 0) { throw 'GPU or dependency verification failed.' }
    & uv pip freeze --python $projectPython | Set-Content -Encoding UTF8 'windows-environment.txt'
    Write-Host 'Setup complete. Use .venv/Scripts/python.exe directly (or uv run --no-sync).'
} finally {
    Pop-Location
}
