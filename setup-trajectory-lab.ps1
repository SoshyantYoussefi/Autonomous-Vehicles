param([string]$Python312 = 'py')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:TEMP = Join-Path $PSScriptRoot '.setup-tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$labPython = Join-Path $PSScriptRoot '.venv-hi5\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) {
    if ($Python312 -eq 'py') { & $Python312 -3.12 -m venv .venv-hi5 }
    else { & $Python312 -m venv .venv-hi5 }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Pass -Python312 with its full path.' }
}
& $labPython -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'Wrong Python version.' }
& $labPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
& $labPython -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $labPython -m pip install -r HI5-Learning/python/TrajectoryPrediction/requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
& $labPython -m ipykernel install --sys-prefix --name trajectory-hi5 --display-name 'Python (HI5 Trajectory)'
if ($LASTEXITCODE -ne 0) { throw 'Kernel installation failed.' }
& $labPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Package validation failed.' }
Write-Host 'Environment ready. Start with .\start-trajectory-lab.ps1'
