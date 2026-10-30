# SPDX-License-Identifier: GPL-3.0-or-later
$PythonArgs = $args
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$scratchRoot = Join-Path $projectRoot '.tmp'
New-Item -ItemType Directory -Force $scratchRoot | Out-Null
$env:TEMP = $scratchRoot
$env:TMP = $scratchRoot
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
Push-Location $projectRoot
try {
    & (Join-Path $projectRoot '.venv/Scripts/python.exe') @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
