$ErrorActionPreference = 'Stop'
$labPython = Join-Path $PSScriptRoot '.venv-hi5\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) { throw 'Run setup-trajectory-lab.ps1 first.' }
$env:JUPYTER_CONFIG_DIR = Join-Path $PSScriptRoot '.jupyter-local\config'
$env:JUPYTER_DATA_DIR = Join-Path $PSScriptRoot '.jupyter-local\data'
$env:JUPYTER_RUNTIME_DIR = Join-Path $PSScriptRoot '.jupyter-local\runtime'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.jupyter-local\matplotlib'
Set-Location (Join-Path $PSScriptRoot 'HI5-Learning\python\TrajectoryPrediction')
& $labPython -m jupyterlab trajectory_prediction.ipynb
