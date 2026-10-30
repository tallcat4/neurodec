# SPDX-License-Identifier: GPL-3.0-or-later
param(
    [Parameter(Mandatory=$true)][string]$Job,        # job directory name under neurodec/runs
    [Parameter(Mandatory=$true)][string]$Module,     # e.g. scripts.probe_frontend or scripts.run_stage
    [Parameter(Mandatory=$true)][string[]]$Arguments # arguments for the module (comma-separated is fine)
)
$ErrorActionPreference = 'Stop'
# `powershell -File` passes "a,b,c" as one element; split so each argument reaches Python separately.
$Arguments = @($Arguments | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' })
$projectRoot = $PSScriptRoot
$jobRoot = Join-Path $projectRoot ('runs/' + $Job)
if (Test-Path -LiteralPath $jobRoot) { throw ('Job already exists: ' + $Job + '; inspect it before launching again.') }
New-Item -ItemType Directory $jobRoot | Out-Null
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:TEMP = Join-Path $projectRoot '.tmp'
$env:TMP = $env:TEMP
$worker = Start-Process -FilePath (Join-Path $projectRoot '.venv/Scripts/python.exe') `
    -WorkingDirectory $projectRoot `
    -ArgumentList (@('-u', '-m', $Module) + $Arguments) `
    -RedirectStandardOutput (Join-Path $jobRoot 'training.log') `
    -RedirectStandardError (Join-Path $jobRoot 'errors.log') `
    -WindowStyle Hidden -PassThru
@{ pid=$worker.Id; module=$Module; arguments=$Arguments; launched_at=(Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $jobRoot 'process.json')
Write-Output ('Started ' + $Job + ' PID ' + $worker.Id)
